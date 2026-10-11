"""flyguy: teach a fly network to choose what another network chooses - the small, all-PyTorch version (2026-10-11).

  python flyguy/teach.py --data MOMENTS.bin [MORE.bin ...] [--teacher bot/model.pt] [--start fresh | CKPT.pt]
        [--graph flyguy/graphs/male_cns_min2_3hop.npz] [--hops 3] [--steps 3000] [--batch 128] [--lr 4e-4]
        [--temp 0.5] [--holdout 0.05] [--eval-every 250] [--out runs/teach/student.pt] [--device cuda] [--kernel torch]

This is how flyguy began: a teacher scores the 90 actions for a game moment, and the student - a fly network - is
moved toward giving the same answer (the loss is the distance between the two answers, the teacher's sharpened by
--temp). Here that is one file and plain PyTorch, so it can be read and changed.

DATA: game moments as the bot sees them, any mix of
  a "FLYD" file      observations as 16-bit floats (flyguy/graphs/viz/demo_moments.bin; the larger set of the
                     published flyguy playing itself is on the repository's releases page)
  a "FLYR" file      a recording of the bot in the real game: run it with FLYGUY_RECORD=1 and it writes what it saw
                     at every decision to runs/flyguy_live/live_*.bin
The teacher's answers are worked out here, not read from a file, so any network this code can load can teach.
The last --holdout share of every file is never learned from: "unseen" in the log is the share of those moments on
which the student's first choice is the teacher's. Moments next to each other in a game are near copies, which is
why the end of each file is held out and not moments picked at random.

--start fresh: the student starts as the connectome gives it - every strength its synapse count with the sign of
the sending neuron's transmitter. Or give a checkpoint to go on from. --graph / --hops: the wiring; a network of
another shape than the teacher's is fine, only the game moments have to be the same 517 numbers.
What is saved (--out) is the running average of the weights, as a checkpoint the bot plays:
    python bot/run.py runs/teach/student.pt --selfcheck          then give run.py that path in bot/bot.toml

Speed: on a GPU about 2,000 moments a second at --batch 128 (--kernel cuda, with CuPy installed and a batch that
is a multiple of 32: about twice that). On a CPU it runs, at --batch 16 and a few moments a second: enough to see
that it works, not to teach anything. flyguy itself was taught with native/flyguy_train.cu on about a thousand
times the 250,000 moments of the published set, most of them fresh from a simulator - so this makes a network that
visibly copies its teacher, not another flyguy.
"""
from __future__ import annotations

import argparse
import struct
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flyguy.model import build_model  # noqa: E402
from flyguy.obs import ObsBuilder  # noqa: E402


def moments(path: Path) -> np.ndarray:
    """[n, obs size] float16: the observations of a FLYD file or of a recording of the bot (FLYR)."""
    with open(path, "rb") as fh:
        kind = fh.read(4)
        if kind == b"FLYD":
            n, od, _ = struct.unpack("<qii", fh.read(16))
            return np.memmap(path, np.float16, "r", 20, (n, od))
        if kind == b"FLYR":
            od, extra = struct.unpack("<ii", fh.read(8))
            rec = 8 + extra + 2 * od
            n = (path.stat().st_size - 12) // rec
            raw = np.memmap(path, np.uint8, "r", 12, (n, rec))
            return np.ascontiguousarray(raw[:, 8 + extra:]).view(np.float16)
    raise SystemExit(f"{path}: neither a FLYD file nor a recording of the bot")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", nargs="+", type=Path, required=True)
    ap.add_argument("--teacher", type=Path, default=ROOT / "bot" / "model.pt")
    ap.add_argument("--start", default="fresh")
    ap.add_argument("--graph", default="")
    ap.add_argument("--hops", type=int, default=0)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=4e-4, help="for the connection strengths and the plugs; the neurons' gains and thresholds get 0.3 of it")
    ap.add_argument("--temp", type=float, default=0.5)
    ap.add_argument("--average", type=float, default=0.999, help="the running average of the weights that is saved and scored")
    ap.add_argument("--holdout", type=float, default=0.05)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--out", type=Path, default=ROOT / "runs" / "teach" / "student.pt")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--kernel", default="torch", choices=["torch", "cuda"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    dev = torch.device(args.device)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    # ---- the teacher, and the student
    tck = torch.load(args.teacher, map_location="cpu", weights_only=False)
    tcfg = dict(tck["config"]["model"])
    teacher = build_model(tcfg).eval().to(dev)
    teacher.load_state_dict(tck["model"], strict=False)
    od = ObsBuilder("cpu", ball_pred=tcfg.get("ball_pred", False), opp_inputs=tcfg.get("opp_inputs", False)).obs_dim
    if args.start == "fresh":
        cfg = dict(tcfg, fly_graph=args.graph or tcfg.get("fly_graph") or "flyguy/graphs/male_cns_min2_3hop.npz",
                   fly_hops=args.hops or tcfg.get("fly_hops", 3), fly_kernel=args.kernel)
        net, step0 = build_model(cfg), 0
    else:
        sck = torch.load(args.start, map_location="cpu", weights_only=False)
        cfg, step0 = dict(sck["config"]["model"], fly_kernel=args.kernel), int(sck.get("taught_steps", 0))
        net = build_model(cfg)
        net.load_state_dict(sck["model"], strict=False)
    assert getattr(net, "fly", False), "the student has to be a fly network (a config with fly_graph)"
    student = net.actor.to(dev)
    shown = build_model(cfg).actor.to(dev).eval()                # the running average: what is scored and saved
    shown.load_state_dict(student.state_dict())
    print(student.describe(), flush=True)

    # ---- the game moments
    seen, unseen = [], []
    for p in args.data:
        m = moments(p)
        assert m.shape[1] == od, f"{p}: moments of {m.shape[1]} numbers, this teacher reads {od}"
        cut = len(m) - max(1, int(len(m) * args.holdout))
        seen.append(m[:cut]); unseen.append(m[cut:])
        print(f"{p.name}: {len(m):,} moments, the last {len(m) - cut:,} held out", flush=True)
    seen = torch.from_numpy(np.concatenate(seen))
    unseen = torch.from_numpy(np.concatenate(unseen)[:: max(1, sum(len(u) for u in unseen) // 4096)].copy())      # (4,096 of them are enough)

    def answers(net_, x):
        return net_.logits(x) if hasattr(net_, "logits") else net_(x)

    @torch.no_grad()
    def score() -> tuple[float, float]:
        same = kl = 0.0
        for i in range(0, len(unseen), 512):
            x = unseen[i:i + 512].to(dev).float()
            t, s = answers(teacher, x).float(), shown(x).float()
            same += float((t.argmax(-1) == s.argmax(-1)).sum())
            target = torch.softmax(t / args.temp, -1)
            kl += float((target * (torch.log(target.clamp_min(1e-12)) - torch.log_softmax(s, -1))).sum())
        return same / len(unseen), kl / len(unseen)

    plugs = [p for k, p in student.named_parameters() if k not in ("w", "gain", "theta")]
    opt = torch.optim.Adam([{"params": [student.w], "lr": args.lr},
                            {"params": [student.gain, student.theta], "lr": 0.3 * args.lr},
                            {"params": plugs, "lr": args.lr}])
    base = [g["lr"] for g in opt.param_groups]
    a0, k0 = score()
    print(f"start | unseen: same first choice {a0:.3f}, distance {k0:.3f} | {len(seen):,} moments to learn from, "
          f"{args.steps * args.batch / len(seen):.1f} passes over them", flush=True)
    t0, run = time.time(), 0.0
    for step in range(1, args.steps + 1):
        for g, b in zip(opt.param_groups, base):                 # the rate falls to a tenth by the end
            g["lr"] = b * 0.1 ** (step / args.steps)
        x = seen[torch.as_tensor(np.sort(rng.integers(0, len(seen), args.batch)))].to(dev).float()
        with torch.no_grad():
            target = torch.softmax(answers(teacher, x).float() / args.temp, -1)
        loss = (target * (torch.log(target.clamp_min(1e-12)) - torch.log_softmax(student(x).float(), -1))).sum(-1).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        keep = min(args.average, step / (step + 9.0))            # (a young average follows closely)
        with torch.no_grad():
            for a, b in zip(shown.parameters(), student.parameters()):
                a.lerp_(b, 1.0 - keep)
        run = loss.item() if step == 1 else 0.98 * run + 0.02 * loss.item()
        if step % args.eval_every == 0 or step == args.steps:
            a, k = score()
            print(f"step {step:,} | distance {run:.3f} | unseen: same first choice {a:.3f}, distance {k:.3f} | "
                  f"{step * args.batch / (time.time() - t0):,.0f} moments a second", flush=True)
            args.out.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": {"actor." + k_: v.detach().cpu() for k_, v in shown.state_dict().items()},
                        "steps": step0 + step, "taught_steps": step0 + step, "config": {"model": dict(cfg, fly_kernel="torch")},
                        "taught_by": args.teacher.name, "unseen_same_first_choice": a}, args.out)
    print(f"saved {args.out} (the running average; the teacher's first choice on {a:.3f} of unseen moments, {a0:.3f} at the start)")


if __name__ == "__main__":
    main()
