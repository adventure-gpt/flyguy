"""flyguy in a fly's body: the brain-to-muscle step as plain numbers, for the live body (body_server.py has no torch).

  uv run python flyguy/body/export_bridge.py [MODEL.pt] [--motor runs/flyguy_body/X_motor.npz]

MODEL.pt: the network that plays (default: the one installed in the Flyguy RLBot entry). Writes
flyguy/graphs/viz/motor_weights.npz: M [motor neurons, descending neurons] = flyguy's strengths of the fly's direct
descending -> motor connections, in the order of the network's last layer (the order of the visualizer feed's last
1,332 bytes); every motor neuron's gain and threshold; `scale` = each motor neuron's 99th-percentile firing over a
recorded game (--motor: brain_to_motor.py's output; default the newest one) - the unit its firing is shown in."""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from flyguy.model import build_model  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("model", nargs="?", default=str(ROOT / "bot" / "model.pt"))
ap.add_argument("--bridge", default=str(ROOT / "flyguy" / "graphs" / "viz" / "motor_bridge.npz"))
ap.add_argument("--motor", default="")
ap.add_argument("--out", default=str(ROOT / "flyguy" / "graphs" / "viz" / "motor_weights.npz"))
args = ap.parse_args()
ck = torch.load(args.model, map_location="cpu", weights_only=False)
ac = build_model(ck["config"]["model"]).eval()
ac.load_state_dict(ck["model"], strict=False)
f = ac.actor
assert getattr(f, "whole_graph", True), ("this network was cut down to the connections it uses (flyguy/trim_graph.py); the fly's descending -> "
                                         "motor connections are not among them. Use a checkpoint of the whole graph.")
b = np.load(args.bridge)
last = getattr(f, f"n{f.hops}").numpy()
row_of = np.full(f.n_neurons, -1, np.int64)
row_of[last] = np.arange(len(last))
src = row_of[b["e_pre"]]
ok = src >= 0
M = np.zeros((len(b["mn"]), len(last)), np.float32)
with torch.no_grad():
    np.add.at(M, (b["e_post"][ok].astype(np.int64), src[ok]), f.w.detach().numpy()[b["e_idx"][ok]])
    mn = torch.as_tensor(b["mn"].astype(np.int64))
    gain = f.gain.detach()[mn].numpy()
    thr = torch.nn.functional.softplus(f.theta.detach())[mn].numpy()
motor = Path(args.motor) if args.motor else max((ROOT / "runs" / "flyguy_body").glob("*_motor.npz"), key=lambda p: p.stat().st_mtime, default=None)
scale = np.full(len(b["mn"]), 0.5, np.float32)
note = "no recorded game: every neuron in units of 0.5"
if motor is not None and motor.exists():
    scale = np.percentile(np.load(motor, allow_pickle=True)["mn"].astype(np.float32), 99, axis=0).astype(np.float32)
    note = f"units from {motor.name}"
np.savez(args.out, M=M, gain=gain.astype(np.float32), thr=thr.astype(np.float32), scale=scale, sizes=np.array(f.sizes, np.int64), steps=int(ck.get("steps", 0)))
print(f"{args.out}: {int((M != 0).sum()):,} connections onto {int((np.abs(M).sum(1) > 0).sum())} motor neurons from the network at step {int(ck.get('steps', 0)):,}; {note}")
