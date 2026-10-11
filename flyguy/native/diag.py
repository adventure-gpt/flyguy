"""flyguy: is the signal alive inside the fly network? Per hop, on real data: how many neurons fire, how large the
inputs are against the thresholds, how many neurons never fire in the batch.

  python flyguy/native/diag.py NET_TEMPLATE.pt PARAMS.bin CHUNK.bin [--n 2048]
NET_TEMPLATE.pt: any torch checkpoint of the fly actor (for the structure); PARAMS.bin: native weights to look at."""
import argparse
import os
import struct
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""          # CPU only: the GPU beside it belongs to the trainer
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from flyguy.model import build_model  # noqa: E402
from flyguy.native.export import read_params, structure  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("template"); ap.add_argument("params"); ap.add_argument("chunk")
ap.add_argument("--n", type=int, default=2048)
args = ap.parse_args()
ck = torch.load(args.template, map_location="cpu", weights_only=False)
f = build_model(ck["config"]["model"]).actor
s = structure(f)
step, (w, gain, theta, in_w, in_b, out_w, out_b) = read_params(Path(args.params), s)
with torch.no_grad():
    f.w[torch.as_tensor(s["wmap"])] = torch.as_tensor(w)
    f.gain.copy_(torch.as_tensor(gain)); f.theta.copy_(torch.as_tensor(theta)); f.in_w.copy_(torch.as_tensor(in_w)); f.in_b.copy_(torch.as_tensor(in_b))
    f.out.weight.copy_(torch.as_tensor(out_w).view_as(f.out.weight)); f.out.bias.copy_(torch.as_tensor(out_b))
with open(args.chunk, "rb") as fh:
    assert fh.read(4) == b"FLYD"
    n, od, na = struct.unpack("<qii", fh.read(16))
    obs = np.frombuffer(fh.read(args.n * od * 2), np.float16).reshape(args.n, od).astype(np.float32)
    fh.seek(20 + n * od * 2)
    tl = np.frombuffer(fh.read(args.n * na * 2), np.float16).reshape(args.n, na).astype(np.float32)
obs, tl = torch.as_tensor(obs), torch.as_tensor(tl)
print(f"native step {step:,}; {args.n} samples of {Path(args.chunk).name}; strengths |w| mean {np.abs(w).mean():.4f} max {np.abs(w).max():.3f}; "
      f"gain mean {gain.mean():.3f} [{gain.min():.2f}, {gain.max():.2f}]; threshold mean {np.log1p(np.exp(theta)).mean():.4f}")
with torch.no_grad():
    thr = torch.nn.functional.softplus(f.theta)
    drive = torch.zeros(args.n, f.sizes[0]).index_add_(1, f.in_r, obs[:, f.in_c] * f.in_w) + f.in_b
    z = f.gain[f.n0] * drive
    h = torch.relu(z - thr[f.n0])
    rows = [("hop 0 (sensory)", z, h)]
    for k in range(1, f.hops + 1):
        nk = getattr(f, f"n{k}")
        zz = torch.zeros(args.n, len(nk)).index_add_(1, getattr(f, f"r{k}"), h[:, getattr(f, f"c{k}")] * f.w[getattr(f, f"e{k}")])
        s_ = getattr(f, f"s{k}")
        if len(s_):
            zz = zz.index_add(1, s_, drive[:, getattr(f, f"d{k}")])
        z = f.gain[nk] * zz
        h = torch.relu(z - thr[nk])
        rows.append((f"hop {k}", z, h))
    for name, z, h in rows:
        act = (h > 0).float()
        print(f"   {name:16s} {z.shape[1]:6d} neurons | firing {act.mean():.3f} of the time | never in this batch {float((act.sum(0) == 0).float().mean()):.3f} | "
              f"always {float((act.mean(0) > 0.999).float().mean()):.3f} | input rms {float(z.pow(2).mean().sqrt()):.4f} | output rms {float(h.pow(2).mean().sqrt()):.4f}")
    logits = f.out(h[:, f.dn_rows])
    lp, tp = torch.log_softmax(logits, -1), torch.log_softmax(tl, -1)
    print(f"   logits spread {float(logits.std()):.3f} (teacher's {float(tl.std()):.3f}); KL {float((tp.exp() * (tp - lp)).sum(-1).mean()):.4f}; "
          f"same top action {float((logits.argmax(-1) == tl.argmax(-1)).float().mean()):.3f}; student entropy {float(-(lp.exp() * lp).sum(-1).mean()):.2f} teacher {float(-(tp.exp() * tp).sum(-1).mean()):.2f}")
