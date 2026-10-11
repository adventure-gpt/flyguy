"""flyguy: how fast does the generator's flyguy (the torch-side network on the GPU) answer? The correct-me stage runs
it for its share of the cars at every step of the recording.

  python flyguy/native/student_speed.py TEMPLATE.pt [--cars 1068] [--piece 1024] [--reps 5]"""
import argparse
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from flyguy import fly_cuda  # noqa: E402
from flyguy.model import build_model  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("template")
ap.add_argument("--cars", type=int, default=1068)
ap.add_argument("--piece", type=int, default=1024)
ap.add_argument("--reps", type=int, default=5)
args = ap.parse_args()
ck = torch.load(args.template, map_location="cpu", weights_only=False)
ac = build_model(ck["config"]["model"]).to("cuda").eval()
print(f"kernel setting in the checkpoint: {ac.actor.kernel!r}; our GPU routines available: {fly_cuda.available()}")
obs = torch.randn(args.cars, ck["config"]["model"].get("obs_dim", 517) if False else 517, device="cuda") * 0.5
for kernel in ("cuda", "torch"):
    ac.actor.kernel = kernel
    for piece in (args.piece, 256):
        pad = (-args.cars) % 32
        x = torch.cat([obs, obs[-1:].expand(pad, -1)]) if pad else obs
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            ac.logits(x[:piece])                                 # (first call: compiles / builds)
            torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(args.reps):
                out = torch.cat([ac.logits(x[j:j + piece]) for j in range(0, len(x), piece)])
            torch.cuda.synchronize()
        dt = (time.time() - t0) / args.reps
        print(f"   {kernel:5s} pieces of {piece:4d}: {dt * 1000:7.1f} ms for {args.cars} cars = {args.cars / dt:8.0f} cars/s | GPU peak {torch.cuda.max_memory_allocated() / 2 ** 30:.2f} GB")
