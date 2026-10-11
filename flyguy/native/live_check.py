"""flyguy: is the LIVE side right? The Flyguy bot records what it saw and chose in the real game
(runs/flyguy_live/live_*.bin, bot/run.py); this asks the teacher what it would have done on those very observations.

  uv run python flyguy/native/live_check.py [LIVE.bin ...] [--model bot/model.pt]
        [--teacher TEACHER.pt] [--sim flyguy/graphs/viz/demo_moments.bin]

Default: the newest recording. If the live observations are what flyguy was trained on, it picks the teacher's first
choice about as often as on simulator moments, the teacher's own answers look as they do in the simulator, and a
decision comes every 8 frames. A much lower agreement, a teacher that suddenly looks undecided or one-sided, or
uneven frame gaps point at the execution side (observations, timing), not at the network."""
import argparse
import struct
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from flyguy.model import build_model  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("live", nargs="*")
ap.add_argument("--model", default=str(ROOT / "bot" / "model.pt"))
ap.add_argument("--teacher", required=True)
ap.add_argument("--sim", default=str(ROOT / "flyguy" / "graphs" / "viz" / "demo_moments.bin"))
args = ap.parse_args()


def load(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    ac = build_model(ck["config"]["model"]).eval()
    ac.load_state_dict(ck["model"], strict=False)
    return ac


def answers(ac, obs):
    with torch.no_grad():
        return torch.cat([ac.logits(obs[i:i + 256]).float() for i in range(0, len(obs), 256)])


def describe(name, fly_act, t_logits):
    lp = torch.log_softmax(t_logits, -1)
    ent = float(-(lp.exp() * lp).sum(-1).mean())
    top = Counter(t_logits.argmax(-1).tolist()).most_common(3)
    agree = float((fly_act == t_logits.argmax(-1)).float().mean())
    rank = (t_logits.argsort(-1, descending=True) == fly_act[:, None]).float().argmax(-1)       # where flyguy's choice sits in the teacher's order
    print(f"   {name}: flyguy takes the teacher's first choice {agree:.3f} of the time, one of its first three {float((rank < 3).float().mean()):.3f}; "
          f"teacher entropy {ent:.2f}, its three commonest choices {[(a, round(c / len(fly_act), 3)) for a, c in top]}")
    return agree


fly, teacher = load(args.model), load(args.teacher)
files = [Path(f) for f in args.live] or sorted((ROOT / "runs" / "flyguy_live").glob("live_*.bin"), key=lambda f: f.stat().st_mtime)[-1:]
sim_obs = None
if Path(args.sim).exists():
    with open(args.sim, "rb") as fh:
        assert fh.read(4) == b"FLYD"
        n, od, na = struct.unpack("<qii", fh.read(16))
        sim_obs = torch.as_tensor(np.frombuffer(fh.read(n * od * 2), np.float16).reshape(n, od).astype(np.float32))
        sim_tl = torch.as_tensor(np.frombuffer(fh.read(n * na * 2), np.float16).reshape(n, na).astype(np.float32))
    print(f"simulator reference ({n:,} recorded moments of the training recording, all modes):")
    describe("simulator", answers(fly, sim_obs).argmax(-1), sim_tl)
for f in files:
    raw = f.read_bytes()
    assert raw[:4] == b"FLYR", f"{f} is not a live recording"
    od, extra = struct.unpack_from("<ii", raw, 4)          # extra: bytes of match clock etc. in every record (0 in the first recordings)
    rec = 8 + extra + 2 * od
    n = (len(raw) - 12) // rec
    if n < 30:
        print(f"{f.name}: only {n} decisions - play a little longer"); continue
    a = np.frombuffer(raw, np.uint8, n * rec, 12).reshape(n, rec)
    head = a[:, :8].copy().view(np.int32)
    frames, act = head[:, 0].astype(np.int64), torch.as_tensor(head[:, 1].astype(np.int64))
    obs = torch.as_tensor(a[:, 8 + extra:].copy().view(np.float16).astype(np.float32))
    gaps = np.diff(frames)
    play = gaps[(gaps > 0) & (gaps < 60)]                              # (larger: goal replays, kickoff countdowns)
    print(f"{f.name}: {n:,} decisions over {int((frames[-1] - frames[0]) / 120)} s of game")
    print(f"   frames between decisions: {float((play == 8).mean()):.3f} exactly 8, mean {play.mean():.2f}, 95% within {int(np.percentile(play, 95))}, worst {int(play.max())}")
    again = answers(fly, obs).argmax(-1)
    print(f"   the installed network on the recorded observations picks the recorded action {float((again == act).float().mean()):.3f} of the time (1.000 = the recording is what this network played)")
    describe("live game", act, answers(teacher, obs))
    if sim_obs is not None:                                            # where the live observations leave the range the recording covered
        lo, hi = sim_obs.min(0).values, sim_obs.max(0).values
        span = (hi - lo).clamp_min(1e-3)
        out = ((obs < lo - 0.25 * span) | (obs > hi + 0.25 * span)).float().mean(0)
        bad = torch.nonzero(out > 0.05).squeeze(1)
        print(f"   observation values outside the simulator's range (by more than a quarter of it) in over 5% of the decisions: {len(bad)} of {od}"
              + (f" - numbers {bad[:12].tolist()} (shares {[round(float(out[i]), 2) for i in bad[:12]]})" if len(bad) else ""))
        dead = torch.nonzero((obs.std(0) < 1e-6) & (sim_obs.std(0) > 1e-3)).squeeze(1)
        print(f"   values that never change in the live game but do in the simulator: {len(dead)} of {od}" + (f" - numbers {dead[:16].tolist()}" if len(dead) else "")
              + " (a 1v1 game leaves the slots of cars that are not there empty)")
