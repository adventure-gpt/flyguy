"""flyguy step 1: the male fly CNS connectome (Janelia FlyEM male-cns v1.0, CC-BY) as a sparse graph.

  uv run --no-project --with pyarrow --with numpy python -I flyguy/build_graph.py DATA_DIR OUT.npz [--min-syn 5]

DATA_DIR holds the three Feather files from male-cns.janelia.org/download. Reads the annotations (which bodies are
neurons, their class and side), streams the 152M-row connection table once, keeps connections between annotated
neurons with at least --min-syn synapses, and writes:
  body      [N] int64   body id of each kept neuron (sorted)
  cls       [N] <U      superclass ('' if none)
  nt        [N] <U      consensus neurotransmitter ('' if none)
  pre, post [E] int32   connection i goes from neuron pre[i] to neuron post[i] (indices into body)
  syn       [E] int32   its synapse count
  sensory, descending, motor, ascending  int32 index lists by superclass
Prints the counts by class, connections kept at several thresholds, and how many neurons a signal entering at the
sensory neurons reaches hop by hop (and when it first reaches the descending neurons).
Runs at below-normal priority: a trainer has this machine."""
import argparse
import ctypes
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc

try:
    ctypes.windll.kernel32.SetPriorityClass(ctypes.c_void_p(-1), 0x00004000)   # below normal
except Exception:
    pass
ap = argparse.ArgumentParser()
ap.add_argument("data"); ap.add_argument("out")
ap.add_argument("--min-syn", type=int, default=5)
args = ap.parse_args()
d = Path(args.data)


def table(name: str) -> pa.Table:
    f = next(d.glob(name))
    with pa.memory_map(str(f), "r") as src:
        return ipc.open_file(src).read_all()


ann = table("body-annotations-*.feather")
body = np.asarray(ann.column("bodyId").to_numpy())
cls = np.array([c or "" for c in ann.column("superclass").to_pylist()])
status = np.array([str(s or "") for s in ann.column("statusLabel").to_pylist()])
print(f"annotated bodies: {len(body):,}; status labels: {dict(Counter(status).most_common(6))}")
keep = cls != ""                                                # a neuron = an annotated body with a class
body, cls = body[keep], cls[keep]
order = np.argsort(body)
body, cls = body[order], cls[order]
n = len(body)
print(f"neurons (bodies with a superclass): {n:,}")
for k, v in Counter(cls).most_common(30):
    print(f"   {k:28s} {v:7,}")
nt_t = table("body-neurotransmitters-*.feather")
nt_map = dict(zip(nt_t.column("body").to_pylist(), nt_t.column("consensus_nt").to_pylist()))
nt = np.array([nt_map.get(int(b)) or "" for b in body])
print("transmitters:", dict(Counter(nt).most_common(8)))

f = next(d.glob("connectome-weights-*.feather"))
pre_l, post_l, syn_l = [], [], []
hist = Counter()
rows = 0
with pa.memory_map(str(f), "r") as src:
    r = ipc.open_file(src)
    for i in range(r.num_record_batches):
        b = r.get_batch(i)
        a, c, w = (b.column(k).to_numpy() for k in ("body_pre", "body_post", "weight"))
        rows += len(w)
        ia, ic = np.searchsorted(body, a), np.searchsorted(body, c)
        ok = (ia < n) & (ic < n)
        ok[ok] &= (body[ia[ok]] == a[ok]) & (body[ic[ok]] == c[ok])
        for t in (1, 2, 3, 5, 10, 20):
            hist[t] += int((ok & (w >= t)).sum())
        m = ok & (w >= args.min_syn) & (ia != ic)
        pre_l.append(ia[m].astype(np.int32)); post_l.append(ic[m].astype(np.int32)); syn_l.append(w[m].astype(np.int32))
pre, post, syn = np.concatenate(pre_l), np.concatenate(post_l), np.concatenate(syn_l)
print(f"connection rows read: {rows:,}; between two neurons with at least k synapses: " + ", ".join(f"k={t}: {hist[t]:,}" for t in sorted(hist)))
print(f"kept (>= {args.min_syn} synapses, no self-connections): {len(pre):,} connections; mean in-degree {len(pre) / n:.1f}")
groups = {k: np.nonzero(np.char.find(cls, k) >= 0)[0].astype(np.int32) for k in ("sensory", "descending", "motor", "ascending")}
for k, v in groups.items():
    print(f"   {k}: {len(v):,} neurons; classes {sorted(set(cls[v]))[:6]}")
# hops: who can a signal entering at the sensory neurons have reached after 1, 2, ... synapses?
reached = np.zeros(n, bool); reached[groups["sensory"]] = True
front = reached.copy()
dn = groups["descending"]
for hop in range(1, 9):
    nxt = np.zeros(n, bool)
    nxt[post[front[pre]]] = True
    front = nxt & ~reached
    reached |= nxt
    print(f"   after {hop} hops: {int(reached.sum()):,} neurons reached ({reached.mean():.1%}); descending neurons reached {int(reached[dn].sum()):,} of {len(dn):,}")
    if not front.any():
        break
np.savez_compressed(args.out, body=body, cls=cls, nt=nt, pre=pre, post=post, syn=syn, min_syn=args.min_syn, **groups)
print("wrote", args.out)
