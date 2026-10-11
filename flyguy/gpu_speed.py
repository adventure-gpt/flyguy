"""flyguy: how fast does the fly-wired actor learn and act on this GPU?

  uv run python flyguy/gpu_speed.py GRAPH.npz [GRAPH.npz ...] [--hops 3 4] [--batches 256 1024 4096]

Per graph and hop count: samples a second for one learning step (forward, backward, Adam) and for acting (forward
only), peak memory, for both ways of computing a hop (gather / scatter, and a sparse matrix product)."""
import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flyguy.fly import FlyActor  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("graphs", nargs="+")
ap.add_argument("--hops", type=int, nargs="+", default=[3, 4])
ap.add_argument("--batches", type=int, nargs="+", default=[256, 1024, 4096])
ap.add_argument("--obs", type=int, default=517)
args = ap.parse_args()
dev = torch.device("cuda")
print(torch.cuda.get_device_name(0), f"{torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GB")
for gpath in args.graphs:
    for hops in args.hops:
        f = FlyActor(gpath, args.obs, 90, hops=hops).to(dev)
        opt = torch.optim.Adam(f.parameters(), lr=1e-4, eps=1e-8)
        print(f"\n{Path(gpath).stem}, {hops} hops: {f.used:,} connection steps a decision; neurons per hop {f.sizes}")
        for b in args.batches:
            x = torch.randn(b, args.obs, device=dev)
            t = torch.randn(b, 90, device=dev)
            for path, limit in (("gather", 10**18), ("sparse", 0)):
                f.gather_limit = limit
                try:
                    torch.cuda.reset_peak_memory_stats()

                    def learn():
                        lp = torch.log_softmax(f(x), -1)
                        loss = (torch.softmax(t, -1) * (torch.log_softmax(t, -1) - lp)).sum(-1).mean()
                        opt.zero_grad(set_to_none=True)
                        loss.backward()
                        opt.step()
                    for _ in range(2):
                        learn()
                    torch.cuda.synchronize(); t0 = time.perf_counter()
                    n = 6
                    for _ in range(n):
                        learn()
                    torch.cuda.synchronize(); dl = (time.perf_counter() - t0) / n
                    peak = torch.cuda.max_memory_allocated() / 2**30
                    with torch.no_grad():
                        f(x); torch.cuda.synchronize(); t0 = time.perf_counter()
                        for _ in range(n):
                            f(x)
                        torch.cuda.synchronize(); da = (time.perf_counter() - t0) / n
                    print(f"   batch {b:5d} {path:6s}: learn {dl * 1e3:7.1f} ms = {b / dl:9,.0f} samples/s, peak {peak:4.1f} GB | act {da * 1e3:6.1f} ms = {b / da:9,.0f} samples/s", flush=True)
                except RuntimeError as e:
                    print(f"   batch {b:5d} {path:6s}: {str(e)[:70]}", flush=True)
                    torch.cuda.empty_cache()
        del f, opt
        torch.cuda.empty_cache()
