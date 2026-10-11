"""flyguy in a fly's body, step 1: the way from the brain to the muscles in the connectome.

  uv run --no-project --with pyarrow --with numpy python -I flyguy/body/build_motor_map.py DATA_DIR
        [--graph flyguy/graphs/male_cns_min2.npz] [--out flyguy/graphs/viz]

flyguy's decision is read off the descending neurons - the neurons that carry commands from a fly's brain to its body.
In the fly those commands go on to the MOTOR neurons, each of which drives one muscle. The graph has 815 motor neurons
and 17,955 direct connections from descending neurons onto 782 of them; the annotations (DATA_DIR, the male-cns v1.0
Feather files) name each motor neuron's muscle, body part and side. Writes
  motor_bridge.npz    mn [815] and dn [1332] neuron indices of the graph; for every descending -> motor connection its
                      index in the graph's connection list (= flyguy's strength for it), its descending neuron and its
                      motor neuron (position in mn)
  motor_neurons.json  per motor neuron: body id, type (the muscle), subclass (fl / ml / hl leg, wm wing, nm neck,
                      pm proboscis, am antenna, ad abdomen, hm haltere, ...), side, neuromere, nerve
Derived from Janelia FlyEM male CNS v1.0 (CC-BY): stays in the git-ignored graphs folder."""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc

ROOT = Path(__file__).resolve().parents[2]
ap = argparse.ArgumentParser()
ap.add_argument("data")
ap.add_argument("--graph", default=str(ROOT / "flyguy" / "graphs" / "male_cns_min2.npz"))
ap.add_argument("--out", default=str(ROOT / "flyguy" / "graphs" / "viz"))
args = ap.parse_args()
g = np.load(args.graph)
body, cls, pre, post, syn = g["body"], g["cls"], g["pre"], g["post"], g["syn"]
N = len(body)
mn = np.nonzero(np.char.find(cls, "motor") >= 0)[0].astype(np.int32)
dn = g["descending"].astype(np.int32)
where = np.full(N, -1, np.int32)
where[mn] = np.arange(len(mn))
is_d = np.zeros(N, bool)
is_d[dn] = True
e = np.nonzero(is_d[pre] & (where[post] >= 0))[0]
f = next(Path(args.data).glob("body-annotations-*.feather"))
with pa.memory_map(str(f), "r") as src:
    ann = ipc.open_file(src).read_all()
cols = {c: ann.column(c).to_pylist() for c in ("bodyId", "type", "subclass", "exitNerve", "somaNeuromere", "somaSide", "superclass")}
row = {b: i for i, b in enumerate(cols["bodyId"])}
neurons = []
for k, i in enumerate(mn.tolist()):
    r = row[int(body[i])]
    neurons.append({"k": k, "body": int(body[i]), "type": cols["type"][r] or "", "subclass": cols["subclass"][r] or "", "side": cols["somaSide"][r] or "",
                    "neuromere": cols["somaNeuromere"][r] or "", "nerve": cols["exitNerve"][r] or "", "superclass": cols["superclass"][r] or ""})
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
np.savez(out / "motor_bridge.npz", mn=mn, dn=dn, e_idx=e.astype(np.int64), e_pre=pre[e].astype(np.int32), e_post=where[post[e]], e_syn=syn[e].astype(np.int32))
(out / "motor_neurons.json").write_text(json.dumps({"credit": "Janelia FlyEM male CNS v1.0 (CC-BY)", "neurons": neurons}, indent=0))
print(f"{len(mn)} motor neurons, {len(dn)} descending neurons, {len(e):,} descending -> motor connections ({int(syn[e].sum()):,} synapses) onto {len(np.unique(post[e]))} motor neurons")
print("by body part:", dict(Counter(n["subclass"] for n in neurons).most_common()))
print("wrote", out / "motor_bridge.npz", "and motor_neurons.json")
