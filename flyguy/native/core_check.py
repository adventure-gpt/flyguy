"""flyguy: is the native library (flyguy_core.dll, flyguy/fly_native.py) the same network as flyguy/fly.py's?

  uv run python flyguy/native/core_check.py [MODEL.pt] [--batch 160] [--max-batch 64]

On a GPU: the logits of the library against the PyTorch implementation's on the CPU, the gradient of a made-up loss
through both (all seven weight arrays), the same again with the batch split into pieces (--batch above --max-batch:
the pieces' gradients must add up), and one optimiser step (the weights PyTorch sees afterwards are the library's).
Small on purpose: it can run beside a trainer."""
import argparse
import copy
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from flyguy.fly_native import NativeFlyActor, grad_arrays  # noqa: E402
from flyguy.model import build_model  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("model", nargs="?", default=str(ROOT / "bot" / "model.pt"))
ap.add_argument("--batch", type=int, default=160)
ap.add_argument("--max-batch", type=int, default=64)
args = ap.parse_args()
ck = torch.load(args.model, map_location="cpu", weights_only=False)
cfg = dict(ck["config"]["model"])
ref = build_model({**cfg, "fly_kernel": "torch"}).eval()
ref.load_state_dict(ck["model"], strict=False)
nat = build_model({**cfg, "fly_kernel": "native"}).eval()
nat.load_state_dict(ck["model"], strict=False)
assert isinstance(nat.actor, NativeFlyActor)
nat.actor.max_batch = args.max_batch
nat = nat.cuda()
torch.manual_seed(0)
obs = torch.randn(args.batch, 517) * 0.5
R = torch.randn(args.batch, 90)
ok = True
for n in sorted({min(args.max_batch, args.batch), args.batch}):
    x, r = obs[:n], R[:n]
    ref.zero_grad()
    lr_ = ref.actor(x)                                         # (with gradients: the PyTorch path, not the bot's compiled one)
    ((lr_ * r).sum() + 0.1 * lr_.pow(2).sum()).backward()
    f = ref.actor
    want = [f.w.grad[nat.actor._wmap.cpu()] if nat.actor._core else None, f.gain.grad, f.theta.grad, f.in_w.grad, f.in_b.grad, f.out.weight.grad.reshape(-1), f.out.bias.grad]
    opt = nat.actor.native_adam(1e-3, 0.0)
    assert opt is not None, "the library did not load (flyguy_core.dll, a GPU)"
    opt.zero_grad()
    ln = nat.actor(x.cuda())
    ((ln * r.cuda()).sum() + 0.1 * ln.pow(2).sum()).backward()
    got = grad_arrays(nat.actor)
    want[0] = f.w.grad[nat.actor._wmap.cpu()]
    d_logit = float((ln.detach().cpu() - lr_.detach()).abs().max())
    print(f"batch {n} ({'one piece' if n <= args.max_batch else f'pieces of {args.max_batch}'}): largest logit difference {d_logit:.2e} (logits up to {float(lr_.abs().max()):.2f})")
    for name, w, g in zip(("strengths", "gain", "threshold", "plug in", "its bias", "plug out", "its bias"), want, got):
        w = w.detach().numpy().reshape(-1)
        g = g.reshape(-1)
        rel = float(np.abs(g - w).max() / max(np.abs(w).max(), 1e-12))
        cos = float((g * w).sum() / max(np.linalg.norm(g) * np.linalg.norm(w), 1e-30))
        ok &= rel < 2e-3 and d_logit < 1e-3
        print(f"   gradient of {name:10s} norm {np.linalg.norm(w):.4e} | largest difference / largest value {rel:.2e} | direction {cos:.6f}")
before = [t.clone() for t in (nat.actor.w.data, nat.actor.gain.data, nat.actor.out.weight.data)]
opt.step()
after = [nat.actor.w.data, nat.actor.gain.data, nat.actor.out.weight.data]
moved = [float((a - b).abs().max()) for a, b in zip(after, before)]
out_of_play = torch.ones(nat.actor.w.numel(), dtype=torch.bool, device="cuda")
out_of_play[nat.actor._wmap] = False
still = float((nat.actor.w.data - before[0])[out_of_play].abs().max())
print(f"one Adam step at 1e-3 (neurons 0.3x): largest change of a strength {moved[0]:.2e}, a gain {moved[1]:.2e}, a read-out weight {moved[2]:.2e}; "
      f"strengths not in play moved {still:.1e} (must be 0); gradient norm the library reports {opt.grad_norm:.3e} (0 = not asked: no clipping)")
ok &= 5e-4 < moved[0] < 2e-3 and still == 0.0 and 1e-4 < moved[1] < 6e-4
sd = opt.state_dict()
opt.load_state_dict(copy.deepcopy(sd))
print(f"optimiser state: {len(sd['m'])} arrays, step count {sd['adam_t']}")
print("ALL GOOD" if ok else "MISMATCH")
