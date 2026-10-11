"""flyguy: does the custom hop (sparse product with its own backward) give the same outputs and gradients as the
plain gather path, and how fast is it? usage: python flyguy/hop_check.py GRAPH.npz [--hops 3] [--batches 256 1024]"""
import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flyguy.fly import FlyActor  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("graph")
ap.add_argument("--hops", type=int, default=3)
ap.add_argument("--batches", type=int, nargs="+", default=[256, 1024, 2048])
ap.add_argument("--device", default="cuda")
args = ap.parse_args()
dev = torch.device(args.device)
f = FlyActor(args.graph, 517, 90, hops=args.hops).to(dev)
print(f.describe())
x = torch.randn(8, 517, device=dev)
t = torch.randn(8, 90, device=dev)


def grads(limit):
    f.gather_limit = limit
    f.zero_grad()
    out = f(x)
    ((out - t) ** 2).mean().backward()
    return out.detach().clone(), {k: p.grad.detach().clone() for k, p in f.named_parameters() if p.grad is not None}


o1, g1 = grads(10 ** 18)
o2, g2 = grads(0)
print("outputs equal:", bool(torch.allclose(o1, o2, atol=1e-5, rtol=1e-4)), float((o1 - o2).abs().max()))
for k in g1:
    d = float((g1[k] - g2[k]).abs().max()); s = float(g1[k].abs().max())
    print(f"   grad {k:10s}: max diff {d:.2e} (max {s:.2e})  {'OK' if d <= 1e-4 * max(s, 1e-12) + 1e-7 else 'DIFFERENT'}")
for kern in (("cuda", "cuda16") if dev.type == "cuda" else ()):
    f.kernel, f.gather_limit = kern, 0
    x32, t32 = torch.randn(32, 517, device=dev), torch.randn(32, 90, device=dev)
    f.zero_grad(); o_k = f(x32); ((o_k - t32) ** 2).mean().backward()
    g_k = {k: p.grad.detach().clone() for k, p in f.named_parameters() if p.grad is not None}
    f.kernel = "torch"
    f.zero_grad(); o_t = f(x32); ((o_t - t32) ** 2).mean().backward()
    worst = max(float((g_k[k] - p.grad).abs().max() / p.grad.abs().max().clamp_min(1e-12)) for k, p in f.named_parameters() if p.grad is not None)
    print(f"kernel {kern}: outputs max diff {float((o_k - o_t).abs().max()):.2e} (spread {float(o_t.std()):.2e}); worst gradient diff relative to its largest entry {worst:.2e}")
for kern in (("torch", "cuda", "cuda16") if dev.type == "cuda" else ()):
    f.kernel = kern
    opt = torch.optim.Adam(f.parameters(), lr=1e-4, eps=1e-8)
    f.gather_limit = 0
    print(f"-- kernel {kern}")
    for b in args.batches:
        xb, tb = torch.randn(b, 517, device=dev), torch.randn(b, 90, device=dev)
        try:
            torch.cuda.reset_peak_memory_stats()

            def learn():
                lp = torch.log_softmax(f(xb), -1)
                loss = (torch.softmax(tb, -1) * (torch.log_softmax(tb, -1) - lp)).sum(-1).mean()
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            for _ in range(2):
                learn()
            torch.cuda.synchronize(); t0 = time.perf_counter()
            for _ in range(6):
                learn()
            torch.cuda.synchronize(); dl = (time.perf_counter() - t0) / 6
            with torch.no_grad():
                f(xb); torch.cuda.synchronize(); t0 = time.perf_counter()
                for _ in range(6):
                    f(xb)
                torch.cuda.synchronize(); da = (time.perf_counter() - t0) / 6
            print(f"   batch {b:5d}: learn {dl * 1e3:7.1f} ms = {b / dl:8,.0f} samples/s, peak {torch.cuda.max_memory_allocated() / 2**30:4.1f} GB | act {b / da:8,.0f} samples/s", flush=True)
        except RuntimeError as e:
            print(f"   batch {b:5d}: {str(e)[:90]}", flush=True)
