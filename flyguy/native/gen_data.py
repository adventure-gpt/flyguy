# NOTE (public repository): this script drives the training harness flyguy was made with - a Rocket League
# simulator, a league of opponents and a PPO trainer (the package `cw` in the imports below), which are NOT
# part of this repository. It is here as the record of how flyguy was made and does not run on its own.
"""flyguy native trainer: the data. Our own V5 teacher plays in the usual training environment (league and all) and
every car's observation is written with the teacher's action logits for it, as 16-bit floats, in chunks the native
trainer reads (flyguy_train.cu).

  python flyguy/native/gen_data.py CONFIG.toml TEACHER.pt OUT_DIR [--rollouts 2] [--keep 1] [--ring 8] [--every 240]
        [--student NET.bin PARAMS.bin TEMPLATE.pt --student-share 0.5]

CONFIG: a trainer config with the TEACHER's model section (V5's) - the environment, league and reset mix are taken
from it. A chunk = the samples of --rollouts rollouts (about 0.5M each); with --keep below 1 only that share of the
moments is written and more rollouts are played for the same chunk (consecutive frames of a car are near copies of
each other: 10-09). Files chunk_0000.bin ... are written round a ring of --ring names (the oldest is overwritten),
one every --every seconds at most (0: without a pause). The process runs below normal priority.
With --student: flyguy itself (its newest native weights, re-read for every chunk) drives that share of the cars -
the states its own mistakes lead to, labelled by the teacher (the correct-me stage). A file OUT_DIR/SHARE holding one
number changes the share from the next chunk on (0 = the teacher drives every car again).
A file OUT_DIR/STOP ends it."""
from __future__ import annotations

import argparse
import struct
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from cw import _sim  # noqa: E402,F401
from cw.config import load  # noqa: E402
from cw.train import Trainer  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("config"); ap.add_argument("teacher"); ap.add_argument("out")
ap.add_argument("--rollouts", type=int, default=2)
ap.add_argument("--ring", type=int, default=8)
ap.add_argument("--every", type=float, default=240.0)
ap.add_argument("--student", nargs=3, metavar=("NET", "PARAMS", "TEMPLATE"))
ap.add_argument("--student-share", type=float, default=0.5)
ap.add_argument("--device", default="", help="cpu: record without the GPU (the teacher's network on the CPU too)")
ap.add_argument("--wild", type=float, default=0.0, help="share of the cars driven by the teacher at --wild-temp")
ap.add_argument("--wild-temp", type=float, default=2.0)
ap.add_argument("--keep", type=float, default=1.0, help="share of the recorded moments written (the chunk keeps its size: more game time per chunk)")
args = ap.parse_args()
if sys.platform == "win32":                                 # the simulation threads stay behind the trainer's GPU-feeding thread
    import ctypes
    ctypes.windll.kernel32.SetPriorityClass(ctypes.c_void_p(-1), 0x00004000)
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
cfg = load(args.config)
cfg.env.use_graphs = False
if args.device:                                             # "cpu": the whole recording off the GPU - two processes sharing one
    cfg.run.device = args.device                            # GPU under Windows halve each other (10-09: trainer 9,600 -> 5,000/s)
on_gpu = cfg.run.device != "cpu" and torch.cuda.is_available()
tr = Trainer(cfg, out / "_gen_run", resume=False)             # the environment, league and opponents of a normal run
tck = torch.load(args.teacher, map_location=tr.device, weights_only=False)
tr.ac.load_state_dict(tck["model"])
tr.ac.eval()
teacher = tr.ac
student = None
if args.student:
    from flyguy.model import build_model
    from flyguy.native.export import read_params, structure
    tpl = torch.load(args.student[2], map_location="cpu", weights_only=False)
    student = build_model(tpl["config"]["model"]).to(tr.device).eval()
    student.load_state_dict(tpl["model"])
    s_struct = structure(build_model(tpl["config"]["model"]).actor)


def refresh_student() -> int:
    step, (w, gain, theta, in_w, in_b, out_w, out_b) = read_params(Path(args.student[1]), s_struct)
    a = student.actor
    with torch.no_grad():
        a.w[torch.as_tensor(s_struct["wmap"], device=tr.device)] = torch.as_tensor(w, device=tr.device)
        a.gain.copy_(torch.as_tensor(gain)); a.theta.copy_(torch.as_tensor(theta)); a.in_w.copy_(torch.as_tensor(in_w))
        a.in_b.copy_(torch.as_tensor(in_b)); a.out.weight.copy_(torch.as_tensor(out_w).view_as(a.out.weight)); a.out.bias.copy_(torch.as_tensor(out_b))
    return step


class MixActor(torch.nn.Module):
    """Who decides while a chunk is recorded: flyguy for a fixed share of the cars - the same cars for the whole chunk,
    so those cars' play is its own from kickoff to goal - and the teacher for the rest. flyguy is run for its own cars
    only, in slices (a multiple of 32 cars, for the GPU routines), so the recording stays small beside the trainer.
    `wild`: a further share of the cars is driven by the teacher at temperature `wild_temp` - looser play that leaves
    the teacher's usual lines and costs no GPU time (flyguy itself answers about 9,000 cars a second there, as much
    GPU as learning from as many samples). The label written for every moment is the teacher's own answer."""

    def __init__(self, student, teacher, share: float, wild: float = 0.0, wild_temp: float = 2.0, piece: int = 256):
        super().__init__()
        self.student, self.teacher, self.share, self.piece = student, teacher, float(share) if student is not None else 0.0, piece
        self.wild, self.wild_temp = float(wild), float(wild_temp)
        self.memory = teacher.memory

    def logits(self, obs: torch.Tensor) -> torch.Tensor:
        out = self.teacher.logits(obs)
        i = torch.arange(obs.shape[0], device=obs.device, dtype=torch.int64)
        u = ((i * 2654435761) % 4294967296).double() / 4294967296.0
        if self.wild > 0:                                   # the next share of the cars: the teacher at a higher temperature
            w = (u >= self.share) & (u < self.share + self.wild)
            out = torch.where(w[:, None], out / self.wild_temp, out)
        mine = (u < self.share).nonzero().squeeze(1)
        if len(mine) == 0:
            return out
        pad = (-len(mine)) % 32
        idx = torch.cat([mine, mine[-1:].expand(pad)]) if pad else mine
        s = torch.cat([self.student.logits(obs[idx[j:j + self.piece]]) for j in range(0, len(idx), self.piece)])[:len(mine)]
        return out.index_copy(0, mine, s.to(out.dtype))

    def value(self, obs: torch.Tensor, extra: torch.Tensor | None = None) -> torch.Tensor:
        return self.teacher.value(obs, extra)


def share_now() -> float:
    """flyguy's share of the cars: OUT_DIR/SHARE (one number, re-read for every chunk) or --student-share."""
    try:
        return min(1.0, max(0.0, float((out / "SHARE").read_text().strip())))
    except (OSError, ValueError):
        return args.student_share


if student is not None:
    from flyguy import fly_cuda
    if fly_cuda.available():
        student.actor.kernel = "cuda"
n = tr.env.n_agents
T = cfg.ppo.rollout_steps
obs_dim = tr.env.obs_builder.obs_dim
print(f"flyguy data: {n} cars x {T} steps a rollout = {n * T:,} samples; chunk = {args.rollouts} rollouts; ring of {args.ring} in {out}", flush=True)
k = 0
while not (out / "STOP").exists():
    t0 = time.time()
    actor = teacher
    note = "teacher plays"
    share = share_now()
    if student is not None and share > 0 and Path(args.student[1]).exists():
        step = refresh_student()
        actor = MixActor(student, teacher, share, args.wild, args.wild_temp).to(tr.device)
        note = f"flyguy (native step {step:,}) drives {share:.0%} of the cars" + (f", the teacher at temperature {args.wild_temp:g} {args.wild:.0%}" if args.wild > 0 else "")
    elif args.wild > 0:
        actor = MixActor(None, teacher, 0.0, args.wild, args.wild_temp).to(tr.device)
        note = f"teacher plays, at temperature {args.wild_temp:g} in {args.wild:.0%} of the cars"
    tmp = out / f"chunk_{k % args.ring:04d}.tmp"
    if on_gpu:
        torch.cuda.reset_peak_memory_stats()
    want, got, played = n * T * args.rollouts, 0, 0
    with open(tmp, "wb") as fh:
        fh.write(b"FLYD"); fh.write(struct.pack("<qii", 0, obs_dim, 90))
        logit_parts = []
        while got < want and not (out / "STOP").exists():
            if tr.opponents and tr.iteration % cfg.league.swap_iters == 0:
                tr._swap_opponents()
            tr.iteration += 1
            played += 1
            with torch.no_grad():
                tr.env.collect(actor, tr.buf, tr.opponents)
                o = tr.buf.obs.view(T * n, -1)
                for i in range(0, T * n, 65536):            # piece by piece: no second copy of the rollout on the GPU
                    part = o[i:i + 65536]
                    if args.keep < 1.0:                     # a random share of the moments: neighbouring frames are near copies
                        part = part[torch.rand(part.shape[0], device=part.device) < args.keep]
                    part = part.float()
                    # the teacher's answers 8,192 cars at a time: a large network's activations for 65,536 at once are
                    # gigabytes (10-10: with V6 as the teacher the generator's peak went from 2.3 to 4.8 GB, and with
                    # the trainer and a game round beside it the card's 8 GB overflowed - a chunk took ten minutes)
                    logit_parts.append(torch.cat([teacher.logits(part[j:j + 8192]) for j in range(0, part.shape[0], 8192)]).half().cpu().numpy())
                    fh.write(part.half().cpu().numpy().tobytes())
                    got += part.shape[0]
        for a in logit_parts:
            fh.write(a.tobytes())
        fh.seek(4); fh.write(struct.pack("<q", got))
    if got < want:                                          # stopped in the middle: no half chunk
        break
    final = out / f"chunk_{k % args.ring:04d}.bin"
    for attempt in range(60):                               # the trainer may be reading the old file of this name
        try:
            tmp.replace(final)
            break
        except PermissionError:
            time.sleep(1)
    del actor
    where = "on the CPU"
    if on_gpu:
        peak = torch.cuda.max_memory_allocated() / 2 ** 30
        torch.cuda.empty_cache()                            # what the recording needed goes back to the trainer's side
        where = f"GPU peak {peak:.1f} GB, held {torch.cuda.memory_reserved() / 2 ** 30:.1f} GB"
    print(f"{time.strftime('%H:%M:%S')} {final.name}: {got:,} samples of {played} rollouts in {time.time() - t0:.0f} s | {note} | {where}", flush=True)
    k += 1
    while time.time() - t0 < args.every and not (out / "STOP").exists():
        time.sleep(2)
print("stopped", flush=True)
import os  # noqa: E402
os._exit(0)                                                 # (10-09: the interpreter hung in the GPU teardown after "stopped")
