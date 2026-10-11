"""flyguy in a fly's body, step 2: what the motor neurons would have done during a real game.

  uv run python flyguy/body/brain_to_motor.py LIVE.bin MODEL.pt [--out runs/flyguy_body/NAME_motor.npz]

LIVE.bin: a recording of the Flyguy bot (runs/flyguy_live/live_*.bin: what it saw at every decision). MODEL.pt: the
network that PLAYED that game (the check line says how many of the recorded actions it reproduces - 1.000 or it is not
the one). For every decision the network is run again exactly as in the game, the firing of its 1,332 descending
neurons is kept, and it is carried one synapse further: through the fly's own 17,955 connections onto the motor
neurons (flyguy's strengths for them - they were never part of its training, so they are the connectome's starting
values: synapse count with the sign of the transmitter), with each motor neuron's own gain and threshold.
Output: time of every decision (s from the first), descending and motor firing, the action taken."""
import argparse
import os
import struct
import sys
from pathlib import Path

os.environ["FLYGUY_NATIVE"] = "0"                    # the PyTorch path: its tap hands out the firing itself, not bytes
import numpy as np  # noqa: E402
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from flyguy.model import build_model  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("live"); ap.add_argument("model")
ap.add_argument("--bridge", default=str(ROOT / "flyguy" / "graphs" / "viz" / "motor_bridge.npz"))
ap.add_argument("--weights", default=str(ROOT / "flyguy" / "graphs" / "viz" / "motor_weights.npz"),
                help="for a network of a trimmed graph (flyguy/trim_graph.py), which does not carry the descending -> motor strengths itself")
ap.add_argument("--out", default="")
args = ap.parse_args()
raw = Path(args.live).read_bytes()
assert raw[:4] == b"FLYR", "not a live recording"
od, extra = struct.unpack_from("<ii", raw, 4)          # extra: bytes of match clock etc. in every record (0 in the first recordings)
rec = 8 + extra + 2 * od
T = (len(raw) - 12) // rec
a = np.frombuffer(raw, np.uint8, T * rec, 12).reshape(T, rec)
head = a[:, :8].copy().view(np.int32)
frames, act = head[:, 0].astype(np.int64), head[:, 1].astype(np.int64)
obs = torch.as_tensor(a[:, 8 + extra:].copy().view(np.float16).astype(np.float32))
ck = torch.load(args.model, map_location="cpu", weights_only=False)
ac = build_model(ck["config"]["model"]).eval()
ac.load_state_dict(ck["model"], strict=False)
f = ac.actor
b = np.load(args.bridge)
last = getattr(f, f"n{f.hops}").numpy()               # the neuron of every row of the last layer (the descending neurons)
row_of = np.full(f.n_neurons, -1, np.int64)
row_of[last] = np.arange(len(last))
src = row_of[b["e_pre"]]
ok = src >= 0                                         # (a descending neuron the signal cannot reach in this many hops sends nothing)
if getattr(f, "whole_graph", True):
    with torch.no_grad():
        w = f.w.detach()[torch.as_tensor(b["e_idx"][ok])]
        M = torch.sparse_coo_tensor(torch.as_tensor(np.stack([b["e_post"][ok].astype(np.int64), src[ok]])), w, (len(b["mn"]), len(last))).coalesce()
        gain = f.gain.detach()[torch.as_tensor(b["mn"].astype(np.int64))]
        thr = torch.nn.functional.softplus(f.theta.detach())[torch.as_tensor(b["mn"].astype(np.int64))]
else:                                                 # those connections are outside the hops the file was trimmed for:
    mw = np.load(args.weights)                        # export_bridge.py wrote them down from the whole network
    assert mw["M"].shape == (len(b["mn"]), len(last)), "motor_weights.npz does not belong to this network"
    M = torch.as_tensor(mw["M"]).to_sparse().coalesce()
    gain, thr = torch.as_tensor(mw["gain"]), torch.as_tensor(mw["thr"])
kept = []
f.tap = lambda acts: kept.append(acts[-1].clone())
torch.set_num_threads(4)
again = np.zeros(T, np.int64)
with torch.no_grad():
    for i in range(T):
        again[i] = int(ac.logits(obs[i:i + 1]).argmax(-1))
    dn = torch.stack(kept)                            # [T, descending]
    mn = torch.relu(gain * torch.sparse.mm(M, dn.t()).t() - thr)
same = float((again == act).mean())
t = (frames - frames[0]) / 120.0
out = Path(args.out) if args.out else ROOT / "runs" / "flyguy_body" / (Path(args.live).stem + "_motor.npz")
out.parent.mkdir(parents=True, exist_ok=True)
np.savez_compressed(out, t=t, frames=frames, action=act, dn=dn.numpy().astype(np.float16), mn=mn.numpy().astype(np.float32), mn_idx=b["mn"],
                    model=str(args.model), same=same)
fire = (mn > 0).float()
print(f"{Path(args.live).name}: {T:,} decisions over {t[-1]:.0f} s; this network reproduces {same:.3f} of the recorded actions")
print(f"   descending neurons firing per decision: {float((dn > 0).float().sum(1).mean()):.0f} of {dn.shape[1]}; connections used {int(ok.sum()):,} of {len(ok):,}")
print(f"   motor neurons firing per decision: {float(fire.sum(1).mean()):.0f} of {mn.shape[1]}; ever firing {int((fire.sum(0) > 0).sum())}; "
      f"firing level: median of the active ones {float(mn[mn > 0].median()):.3f}, 95% under {float(mn[mn > 0].quantile(0.95)):.3f}")
print("wrote", out)
