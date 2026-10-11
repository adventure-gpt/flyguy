"""flyguy: bring silent neurons back without changing what the network does.

  python flyguy/native/revive.py TEMPLATE.pt PARAMS_IN.bin CHUNK.bin PARAMS_OUT.bin [--samples 512] [--seed 0]

10-09: with strengths started from the transmitter signs, 40% of the neurons of hops 1 and 2 never fired - inputs that
are mostly inhibitory keep a rectifying neuron silent, a silent neuron passes no gradient, so it stays silent and its
connections never learn. For every neuron (outside the sensory layer) that fires for none of the samples in any hop:
its incoming strengths keep their relative sizes and get a fresh random sign (so its input is positive about half the
time) and one common factor that brings its input to the size the live neurons have, its threshold and gain go back
to their starting values, and its OUTGOING strengths to live neurons (and, for a descending neuron, its weights onto
the actions) are set to zero - it was sending nothing before and sends nothing now, so the network's output is
unchanged, but from here on the neuron fires and its outgoing strengths (and through them everything behind it) can
learn. Reports the firing share per hop and the loss before and after."""
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
from flyguy.native.export import read_params, structure, write_params  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("template"); ap.add_argument("params"); ap.add_argument("chunk"); ap.add_argument("out")
ap.add_argument("--samples", type=int, default=512)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--passes", type=int, default=3, help="rounds of bringing the revived neurons' inputs to --target")
ap.add_argument("--target", type=float, default=0.4, help="input size (rms) given to a revived neuron; the live ones sit at 0.4-0.5")
args = ap.parse_args()
ck = torch.load(args.template, map_location="cpu", weights_only=False)
f = build_model(ck["config"]["model"]).actor
s = structure(f)
step, P = read_params(Path(args.params), s)
wmap = torch.as_tensor(s["wmap"])
g = np.load(ROOT / ck["config"]["model"]["fly_graph"])
pre, post = g["pre"][s["wmap"]].astype(np.int64), g["post"][s["wmap"]].astype(np.int64)     # of the in-play connections


def load(P):
    w, gain, theta, in_w, in_b, out_w, out_b = P
    with torch.no_grad():
        f.w.zero_(); f.w[wmap] = torch.as_tensor(w)
        f.gain.copy_(torch.as_tensor(gain)); f.theta.copy_(torch.as_tensor(theta)); f.in_w.copy_(torch.as_tensor(in_w)); f.in_b.copy_(torch.as_tensor(in_b))
        f.out.weight.copy_(torch.as_tensor(out_w).view_as(f.out.weight)); f.out.bias.copy_(torch.as_tensor(out_b))


with open(args.chunk, "rb") as fh:
    assert fh.read(4) == b"FLYD"
    n, od, na = struct.unpack("<qii", fh.read(16))
    obs = torch.as_tensor(np.frombuffer(fh.read(args.samples * od * 2), np.float16).reshape(args.samples, od).astype(np.float32))
    fh.seek(20 + n * od * 2)
    tl = torch.as_tensor(np.frombuffer(fh.read(args.samples * na * 2), np.float16).reshape(args.samples, na).astype(np.float32))


@torch.no_grad()
def survey():
    """Firing counts per neuron (over every hop it takes part in), its largest input rms over those hops, firing share
    per hop, KL to the teacher."""
    fired = torch.zeros(f.n_neurons)
    seen = torch.zeros(f.n_neurons, dtype=torch.bool)
    sq = [torch.zeros(int(n)) for n in f.sizes]
    share = [0.0] * (f.hops + 1)
    kl = 0.0
    thr = torch.nn.functional.softplus(f.theta)
    for i in range(0, args.samples, 64):
        x = obs[i:i + 64]
        drive = torch.zeros(len(x), f.sizes[0]).index_add_(1, f.in_r, x[:, f.in_c] * f.in_w) + f.in_b
        h = torch.relu(f.gain[f.n0] * drive - thr[f.n0])
        share[0] += float((h > 0).float().mean()) * len(x)
        for k in range(1, f.hops + 1):
            nk = getattr(f, f"n{k}")
            z = torch.zeros(len(x), len(nk)).index_add_(1, getattr(f, f"r{k}"), h[:, getattr(f, f"c{k}")] * f.w[getattr(f, f"e{k}")])
            s_ = getattr(f, f"s{k}")
            if len(s_):
                z = z.index_add(1, s_, drive[:, getattr(f, f"d{k}")])
            sq[k] += z.pow(2).sum(0)
            h = torch.relu(f.gain[nk] * z - thr[nk])
            fired[nk] += (h > 0).float().sum(0)
            seen[nk] = True
            share[k] += float((h > 0).float().mean()) * len(x)
        lp, tp = torch.log_softmax(f.out(h[:, f.dn_rows]), -1), torch.log_softmax(tl[i:i + 64], -1)
        kl += float((tp.exp() * (tp - lp)).sum(-1).sum())
    rms = torch.zeros(f.n_neurons)
    for k in range(1, f.hops + 1):
        nk = getattr(f, f"n{k}")
        rms[nk] = torch.maximum(rms[nk], (sq[k] / args.samples).sqrt())
    return fired, seen, rms.numpy(), [v / args.samples for v in share], kl / args.samples


load(P)
fired, seen, _, share, kl0 = survey()
is_sens = np.zeros(f.n_neurons, bool); is_sens[g["sensory"]] = True
dead = (seen & (fired == 0)).numpy() & ~is_sens
print(f"native step {step:,}, {args.samples} samples: firing share per hop {[round(v, 3) for v in share]}; KL {kl0:.4f}")
print(f"silent neurons (never fire in any hop): {int(dead.sum()):,} of {int((seen.numpy() & ~is_sens).sum()):,} in play outside the sensory layer")
rng = np.random.default_rng(args.seed)
w = P[0].copy()
inc, outg = dead[post], dead[pre]
w[inc] = np.abs(w[inc]) * rng.choice([-1.0, 1.0], int(inc.sum())).astype(np.float32)
w[outg & ~inc] = 0.0                                             # silent -> live only: a connection between two silent neurons stays
gain, theta = P[1].copy(), P[2].copy()
gain[dead], theta[dead] = 1.0, -4.0
out_w = P[5].copy().reshape(na, -1)
dn_ids = getattr(f, f"n{f.hops}")[f.dn_rows].numpy()
out_w[:, dead[dn_ids]] = 0.0                                     # a silent descending neuron sent nothing to the actions either
P2 = [w, gain, theta, P[3], P[4], out_w.reshape(P[5].shape), P[6]]
print(f"re-signed {int(inc.sum()):,} incoming strengths, zeroed {int((outg & ~inc).sum()):,} outgoing ones and the action weights of {int(dead[dn_ids].sum())} descending neurons")
for it in range(args.passes + 1):                                # hop after hop: bring every revived neuron's input to the target size
    load(P2)
    fired2, _, rms, share2, kl1 = survey()
    dead2 = (seen & (fired2 == 0)).numpy() & ~is_sens
    rate = (fired2.numpy()[dead] / args.samples).mean() if dead.any() else 0.0
    print(f"   pass {it}: firing share per hop {[round(v, 3) for v in share2]}; still silent {int(dead2.sum()):,}; the revived fire {rate:.3f} of the time; KL {kl1:.4f}")
    assert abs(kl1 - kl0) < 1e-4 * max(1.0, abs(kl0)), "the surgery changed the network's output"
    if it == args.passes:
        break
    scale = np.ones(f.n_neurons, np.float32)
    m = dead & (rms > 0)
    scale[m] = np.clip(args.target / rms[m], 0.02, 200.0)
    w[inc] *= scale[post[inc]]
write_params(P2, Path(args.out), step)
print("wrote", args.out)
