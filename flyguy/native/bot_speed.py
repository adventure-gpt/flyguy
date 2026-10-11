"""flyguy: how long does the bot's decision take on this PC? The network alone, compiled and through PyTorch, for
one car, on real game moments if a demo file is there (random inputs fire differently).

  uv run python flyguy/native/bot_speed.py [MODEL.pt] [--moments flyguy/graphs/viz/demo_moments.bin] [--threads 6]

RLBot's performance figure counts the game ticks answered in time; a tick is 8.3 ms."""
import argparse
import os
import struct
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from flyguy.model import build_model  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("model", nargs="?", default=str(ROOT / "bot" / "model.pt"))
ap.add_argument("--moments", default=str(ROOT / "flyguy" / "graphs" / "viz" / "demo_moments.bin"))
ap.add_argument("--threads", type=int, default=6)
ap.add_argument("--n", type=int, default=300)
args = ap.parse_args()
torch.set_num_threads(args.threads)
ck = torch.load(args.model, map_location="cpu", weights_only=False)
ac = build_model(ck["config"]["model"]).eval()
ac.load_state_dict(ck["model"], strict=False)
if Path(args.moments).exists():
    with open(args.moments, "rb") as fh:
        assert fh.read(4) == b"FLYD"
        n, od, na = struct.unpack("<qii", fh.read(16))
        obs = torch.as_tensor(np.frombuffer(fh.read(min(n, args.n) * od * 2), np.float16).reshape(-1, od).astype(np.float32))
    what = "recorded game moments"
else:
    obs = torch.randn(args.n, 517) * 0.5
    what = "random inputs"
res = {}
for name, env in (("compiled", "1"), ("PyTorch", "0")):
    os.environ["FLYGUY_NATIVE"] = env
    ac.actor._nat = None
    with torch.no_grad():
        ac.logits(obs[:1])
        if name == "compiled" and ac.actor._native() is None:
            print("   compiled: flyguy/native/fly_infer.dll is not there (build_infer.bat)")
            continue
        ts = []
        outs = []
        for i in range(len(obs)):
            t0 = time.perf_counter()
            outs.append(ac.logits(obs[i:i + 1]))
            ts.append(time.perf_counter() - t0)
    ts = np.array(ts) * 1000
    res[name] = torch.cat(outs)
    print(f"   {name:9s} {np.median(ts):6.2f} ms a decision (median; 95% under {np.percentile(ts, 95):.2f}, worst {ts.max():.2f}) on {what}, {args.threads} threads")
if len(res) == 2:
    d = (res["compiled"] - res["PyTorch"]).abs().max()
    same = (res["compiled"].argmax(-1) == res["PyTorch"].argmax(-1)).float().mean()
    print(f"   largest difference between the two answers {float(d):.2e}; same action chosen in {float(same):.1%} of {len(obs)} moments")
