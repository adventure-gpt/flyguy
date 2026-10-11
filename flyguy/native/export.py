"""flyguy native trainer: write the network's structure and weights as flat binaries, and read weights back.

  python flyguy/native/export.py net   GRAPH.npz NET.bin [--hops 3] [--fan 16] [--obs 517]
  python flyguy/native/export.py params STATE.pt PARAMS.bin        torch checkpoint (fly actor) -> native weights
  python flyguy/native/export.py torch  PARAMS.bin TEMPLATE.pt OUT.pt   native weights -> a torch checkpoint like TEMPLATE
  python flyguy/native/export.py batch  STATE.pt BATCH.bin [--n 64] [--teacher T.pt]   a fixed batch + what the PyTorch
        implementation computes on it (loss, logits, gradient norms) - for `flyguy_train --check`

NET.bin (little endian, int32 unless noted): "FLY1", obs_dim, n_act, hops, n_neurons, n_w, n_s, fan, n_dn,
sizes[hops+1], E[hops]; plug: in_crow[n_s+1], in_col[E_in], in_tcrow[obs_dim+1], in_tcol[E_in], in_tperm[E_in];
n0[n_s]; per hop: nid[N_k], crow[N_k+1], col[E_k], widx[E_k], tcrow[N_{k-1}+1], tcol[E_k], twidx[E_k], drow[N_k];
dn_rows[n_dn]; wmap[n_w] (which connection of the graph each native strength is).
Only the n_w connections that can affect a decision are parameters there (the others never get a gradient).
PARAMS.bin: "FLYP", step (int64), float32: W[n_w], gain[n_neurons], theta[n_neurons], in_w[E_in], in_b[n_s],
out_w[n_act * n_dn] (row = action), out_b[n_act].
"""
from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from flyguy.fly import FlyActor  # noqa: E402


def structure(f: FlyActor) -> dict:
    """The index arrays of the native trainer, from the actor's own buffers (so the two cannot disagree)."""
    K = f.hops
    used = torch.unique(torch.cat([getattr(f, f"e{k}") for k in range(1, K + 1)])).cpu().numpy()        # in-play connections
    comp = np.full(f.n_conn, -1, np.int64)
    comp[used] = np.arange(len(used))
    s = {"obs_dim": f.obs_dim, "n_act": f.out.out_features, "hops": K, "n_neurons": f.n_neurons, "n_w": len(used),
         "n_s": f.sizes[0], "fan": f.fan, "n_dn": len(f.dn_rows), "sizes": list(f.sizes), "wmap": used,
         "in_crow": f.in_crow.cpu().numpy(), "in_col": f.in_ci.cpu().numpy(), "in_tcrow": f.in_tcrow.cpu().numpy(), "in_tcol": f.in_tci.cpu().numpy(),
         "in_tperm": f.in_tperm.cpu().numpy(), "n0": f.n0.cpu().numpy(), "dn_rows": f.dn_rows.cpu().numpy(), "hop": []}
    for k in range(1, K + 1):
        g = lambda name: getattr(f, f"{name}{k}").cpu().numpy()
        widx = comp[g("e")]
        drow = np.full(f.sizes[k], -1, np.int64)
        drow[g("s")] = g("d")
        s["hop"].append({"nid": g("n"), "crow": g("crow"), "col": g("ci"), "widx": widx, "tcrow": g("tcrow"),
                         "tcol": g("tci"), "twidx": widx[g("tperm")], "drow": drow})
    return s


def write_net(f: FlyActor, out: Path) -> dict:
    s = structure(f)
    i32 = lambda a: np.ascontiguousarray(a, dtype=np.int32).tobytes()
    with open(out, "wb") as fh:
        fh.write(b"FLY1")
        fh.write(struct.pack("<8i", s["obs_dim"], s["n_act"], s["hops"], s["n_neurons"], s["n_w"], s["n_s"], s["fan"], s["n_dn"]))
        fh.write(i32(s["sizes"])); fh.write(i32([len(h["col"]) for h in s["hop"]]))
        for k in ("in_crow", "in_col", "in_tcrow", "in_tcol", "in_tperm", "n0"):
            fh.write(i32(s[k]))
        for h in s["hop"]:
            for k in ("nid", "crow", "col", "widx", "tcrow", "tcol", "twidx", "drow"):
                fh.write(i32(h[k]))
        fh.write(i32(s["dn_rows"])); fh.write(i32(s["wmap"]))
    return s


def params_from_torch(f: FlyActor, s: dict) -> list[np.ndarray]:
    return [f.w.detach().cpu().numpy()[s["wmap"]], f.gain.detach().cpu().numpy(), f.theta.detach().cpu().numpy(), f.in_w.detach().cpu().numpy(),
            f.in_b.detach().cpu().numpy(), f.out.weight.detach().cpu().numpy().reshape(-1), f.out.bias.detach().cpu().numpy()]


def write_params(arrs: list[np.ndarray], out: Path, step: int = 0) -> None:
    with open(out, "wb") as fh:
        fh.write(b"FLYP"); fh.write(struct.pack("<q", step))
        for a in arrs:
            fh.write(np.ascontiguousarray(a, dtype=np.float32).tobytes())


def read_params(path: Path, s: dict) -> tuple[int, list[np.ndarray]]:
    raw = Path(path).read_bytes()
    assert raw[:4] == b"FLYP"
    step = struct.unpack("<q", raw[4:12])[0]
    sizes = [s["n_w"], s["n_neurons"], s["n_neurons"], s["n_s"] * s["fan"], s["n_s"], s["n_act"] * s["n_dn"], s["n_act"]]
    a, off = [], 12
    for n in sizes:
        a.append(np.frombuffer(raw, np.float32, n, off).copy()); off += 4 * n
    return step, a


def actor_of(state: Path) -> tuple[dict, FlyActor]:
    from flyguy.model import build_model
    ck = torch.load(state, map_location="cpu", weights_only=False)
    ac = build_model(ck["config"]["model"])
    ac.load_state_dict(ck["model"], strict=False)
    return ck, ac.actor


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["net", "params", "torch", "batch"])
    ap.add_argument("a"); ap.add_argument("b"); ap.add_argument("c", nargs="?")
    ap.add_argument("--hops", type=int, default=3); ap.add_argument("--fan", type=int, default=16)
    ap.add_argument("--obs", type=int, default=517); ap.add_argument("--n", type=int, default=64)
    args = ap.parse_args()
    if args.cmd == "net":
        f = FlyActor(args.a, args.obs, 90, hops=args.hops, fan=args.fan)
        s = write_net(f, Path(args.b))
        write_params(params_from_torch(f, s), Path(args.b).with_suffix(".init.bin"))
        print(f"{args.b}: {s['n_w']:,} strengths in play, neurons per hop {s['sizes']}, connection steps {sum(len(h['col']) for h in s['hop']):,}")
    elif args.cmd == "params":
        ck, f = actor_of(Path(args.a))
        write_params(params_from_torch(f, structure(f)), Path(args.b), 0)     # the native trainer counts its own steps
        print(f"{args.b}: weights of {args.a} (its count: {int(ck.get('steps', 0)):,}); native step counter starts at 0")
    elif args.cmd == "torch":
        ck, f = actor_of(Path(args.b))
        s = structure(f)
        step, (w, gain, theta, in_w, in_b, out_w, out_b) = read_params(Path(args.a), s)
        sd = dict(ck["model"])
        full = sd["actor.w"].clone()
        full[torch.as_tensor(s["wmap"])] = torch.as_tensor(w)
        sd.update({"actor.w": full, "actor.gain": torch.as_tensor(gain), "actor.theta": torch.as_tensor(theta),
                   "actor.in_w": torch.as_tensor(in_w), "actor.in_b": torch.as_tensor(in_b),
                   "actor.out.weight": torch.as_tensor(out_w).view(s["n_act"], s["n_dn"]), "actor.out.bias": torch.as_tensor(out_b)})
        torch.save({"model": sd, "steps": step, "config": ck["config"], "git": ck.get("git"), "note": f"flyguy native weights {args.a}"}, args.c)
        print(f"{args.c}: native weights at step {step:,} as a torch checkpoint")
    else:                                                       # a fixed batch and the PyTorch answer on it
        ck, f = actor_of(Path(args.a))
        torch.manual_seed(0)
        n = args.n
        x = torch.randn(n, f.obs_dim) * 0.7
        t = torch.randn(n, 90) * 2.0
        f.zero_grad()
        logits = f(x)
        lp, tp = torch.log_softmax(logits, -1), torch.log_softmax(t, -1)
        loss = (tp.exp() * (tp - lp)).sum(-1).mean()
        loss.backward()
        s = structure(f)
        g = [f.w.grad.numpy()[s["wmap"]], f.gain.grad.numpy(), f.theta.grad.numpy(), f.in_w.grad.numpy(), f.in_b.grad.numpy(),
             f.out.weight.grad.numpy().reshape(-1), f.out.bias.grad.numpy()]
        with open(args.b, "wb") as fh:
            fh.write(b"FLYB"); fh.write(struct.pack("<i", n))
            fh.write(x.numpy().astype(np.float32).tobytes()); fh.write(t.numpy().astype(np.float32).tobytes())
        print(f"{args.b}: {n} samples | PyTorch: loss {float(loss):.6f} agree {float((logits.argmax(-1) == t.argmax(-1)).float().mean()):.4f}")
        print("   logits[0][:5] " + " ".join(f"{v:.6f}" for v in logits[0, :5].tolist()))
        for name, a in zip(("W", "gain", "theta", "in_w", "in_b", "out_w", "out_b"), g):
            print(f"   grad {name:6s} norm {float(np.linalg.norm(a)):.6e} sum {float(a.astype(np.float64).sum()):+.6e}")


if __name__ == "__main__":
    main()
