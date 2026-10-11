"""flyguy visualizer, step 1: where every neuron of the graph sits, and which of them take part in a decision.

  uv run --with pyarrow python -I flyguy/viz/build_layout.py DATA_DIR [--graph flyguy/graphs/male_cns_min2.npz]
        [--out flyguy/graphs/viz] [--hops 3] [--edges 12000]

DATA_DIR holds the male-cns v1.0 Feather files (the body annotations carry a soma position for most neurons).
A neuron without one (the sensory neurons: their cell bodies are outside the CNS) is put at the synapse-weighted
centre of the neurons it connects to, with a little scatter. The cloud is turned so its long axis (brain to nerve
cord) stands upright with the brain on top. Writes, for flyguy/viz/index.html:
  layout.bin  "FLYL", version, N, layers, 4 layer sizes, edge count; float32 xyz per neuron (in -1..1); a class code
              and a flag byte per neuron (1 in play, 2 measured soma position, 4 descending, 8 sensory); for every
              layer of a decision (sensory, hop 1, 2, 3) the neuron of each row; the strongest connections of each hop
              as (from neuron, to neuron, relative size)
  meta.json   class names, counts and colours, the credit line
The files are derived from Janelia FlyEM male CNS v1.0 (CC-BY): they stay in the git-ignored graphs folder."""
import argparse
import json
import struct
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))                    # (python -I: our own code is named explicitly; nothing is run from DATA_DIR)

ap = argparse.ArgumentParser()
ap.add_argument("data")
ap.add_argument("--graph", default=str(ROOT / "flyguy" / "graphs" / "male_cns_min2.npz"))
ap.add_argument("--out", default=str(ROOT / "flyguy" / "graphs" / "viz"))
ap.add_argument("--hops", type=int, default=3)
ap.add_argument("--fan", type=int, default=16)
ap.add_argument("--edges", type=int, default=12000, help="connections drawn per hop (the ones with the most synapses)")
args = ap.parse_args()

import pyarrow as pa  # noqa: E402
import pyarrow.ipc as ipc  # noqa: E402

g = np.load(args.graph)
body, cls, pre, post, syn = g["body"], g["cls"], g["pre"].astype(np.int64), g["post"].astype(np.int64), g["syn"].astype(np.float64)
N = len(body)
f = next(Path(args.data).glob("body-annotations-*.feather"))
with pa.memory_map(str(f), "r") as src:
    ann = ipc.open_file(src).read_all()
soma = dict(zip(ann.column("bodyId").to_pylist(), ann.column("somaLocation").to_pylist()))
pos = np.full((N, 3), np.nan)
for i, b in enumerate(body.tolist()):
    s = soma.get(b)
    if s is not None and len(s) == 3:
        pos[i] = s
known = ~np.isnan(pos[:, 0])
measured = known.copy()
print(f"{N:,} neurons; soma position measured for {int(known.sum()):,}")
for it in range(6):                              # the rest: centre of their partners, weighted by synapses
    acc, wsum = np.zeros((N, 3)), np.zeros(N)
    for a, b in ((pre, post), (post, pre)):
        m = known[a]
        wsum += np.bincount(b[m], weights=syn[m], minlength=N)
        for d in range(3):
            acc[:, d] += np.bincount(b[m], weights=syn[m] * pos[a[m], d], minlength=N)
    fill = ~known & (wsum > 0)
    if not fill.any():
        break
    pos[fill] = acc[fill] / wsum[fill, None]
    known |= fill
    print(f"   round {it}: placed {int(fill.sum()):,} by their partners")
if (~known).any():
    pos[~known] = np.nanmean(pos, 0)
    print(f"   {int((~known).sum()):,} without any placed partner: at the centre")
rng = np.random.default_rng(0)
ext = np.nanmax(pos, 0) - np.nanmin(pos, 0)
pos[~measured] += rng.normal(0, 0.008, (int((~measured).sum()), 3)) * ext.max()
pos -= pos[measured].mean(0)
u, s_, vt = np.linalg.svd(pos[measured] - pos[measured].mean(0), full_matrices=False)
q = pos @ vt.T                                   # the cloud's own axes
lo, hi = np.percentile(q[measured], 0.2, axis=0), np.percentile(q[measured], 99.8, axis=0)     # (stray positions must not count)
order = np.argsort(-(hi - lo))                   # the longest stretch is brain - nerve cord (most NEURONS lie across, in the optic lobes)
axes = [order[1], order[0], order[2]]            # x = width, y = long axis, z = depth
p = (q[:, axes] - ((lo + hi) / 2)[axes]) / ((hi - lo).max() / 2)
sign = np.linalg.det(vt[axes])
top, bottom = p[measured & (p[:, 1] > 0)], p[measured & (p[:, 1] <= 0)]
if top[:, 0].std() < bottom[:, 0].std():         # the brain (with the optic lobes) is the wide end: on top
    p[:, 1] *= -1
    sign = -sign
if sign < 0:
    p[:, 2] *= -1                                # keep it a rotation, not a mirror image
p = np.clip(p, -1.1, 1.1)
print(f"   extent after turning: x {p[:, 0].min():.2f}..{p[:, 0].max():.2f}, y {p[:, 1].min():.2f}..{p[:, 1].max():.2f}, z {p[:, 2].min():.2f}..{p[:, 2].max():.2f}")

from flyguy.fly import FlyActor  # noqa: E402

fly = FlyActor(args.graph, 517, 90, hops=args.hops, fan=args.fan)
layers = [fly.n0.numpy().astype(np.uint32)] + [getattr(fly, f"n{k}").numpy().astype(np.uint32) for k in range(1, args.hops + 1)]
in_play = np.zeros(N, bool)
for l in layers:
    in_play[l] = True
flag = in_play.astype(np.uint8) + 2 * measured.astype(np.uint8)
flag[g["descending"]] += 4
flag[g["sensory"]] += 8
names = [k for k, _ in Counter(cls.tolist()).most_common()]
code = np.array([names.index(c) for c in cls.tolist()], np.uint8)
fixed = {"sensory": [0.25, 0.9, 1.0], "descending": [1.0, 0.45, 0.15], "motor": [1.0, 0.85, 0.2], "ascending": [0.45, 1.0, 0.5]}
wheel = [[0.62, 0.55, 1.0], [0.85, 0.5, 1.0], [0.4, 0.62, 1.0], [1.0, 0.55, 0.8], [0.75, 0.8, 1.0], [0.55, 0.9, 0.85], [1.0, 0.7, 0.55], [0.8, 0.95, 0.6]]
classes, w = [], 0
for nme in names:
    col = next((v for k, v in fixed.items() if k in nme), None)
    if col is None:
        col = wheel[w % len(wheel)]
        w += 1
    classes.append({"name": nme or "unclassified", "count": int((cls == nme).sum()), "in_play": int((in_play & (cls == nme)).sum()), "color": col})
e_pre, e_post, e_w = [], [], []
for k in range(1, args.hops + 1):                # the connections of each hop with the most synapses
    c, r, e = getattr(fly, f"c{k}").numpy(), getattr(fly, f"r{k}").numpy(), getattr(fly, f"e{k}").numpy()
    sy = syn[e]
    top_e = np.argsort(-sy)[:args.edges]
    e_pre.append(layers[k - 1][c[top_e]]); e_post.append(layers[k][r[top_e]]); e_w.append((sy[top_e] / sy[top_e].max()).astype(np.float32))
e_pre, e_post, e_w = np.concatenate(e_pre).astype(np.uint32), np.concatenate(e_post).astype(np.uint32), np.concatenate(e_w)
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
with open(out / "layout.bin", "wb") as fh:
    fh.write(b"FLYL")
    fh.write(struct.pack("<3I", 1, N, len(layers)))
    fh.write(struct.pack("<4I", *([len(l) for l in layers] + [0] * 4)[:4]))
    fh.write(struct.pack("<I", len(e_pre)))
    fh.write(p.astype(np.float32).tobytes())
    blob = code.tobytes() + flag.tobytes()
    fh.write(blob + b"\0" * ((-len(blob)) % 4))
    for l in layers:
        fh.write(l.tobytes())
    fh.write(e_pre.tobytes()); fh.write(e_post.tobytes()); fh.write(e_w.tobytes())
meta = {"n": N, "in_play": int(in_play.sum()), "layers": [int(len(l)) for l in layers], "hops": args.hops, "classes": classes,
        "connections": int(len(pre)), "connections_in_play": int(fly.in_play), "edges_drawn": int(len(e_pre)),
        "credit": "Wiring: Janelia FlyEM male CNS v1.0 (CC-BY)"}
(out / "meta.json").write_text(json.dumps(meta, indent=1))
print(f"wrote {out / 'layout.bin'} ({(out / 'layout.bin').stat().st_size / 1e6:.1f} MB) and meta.json: {int(in_play.sum()):,} of {N:,} neurons in play, layers {meta['layers']}, {len(e_pre):,} connections drawn")
for c in classes:
    print(f"   {c['name']:26s} {c['count']:7,}  in play {c['in_play']:7,}")
