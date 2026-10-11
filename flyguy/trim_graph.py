"""flyguy: the part of the wiring a network of N hops uses, as a graph file of its own (2026-10-11, for sharing).

  python flyguy/trim_graph.py GRAPH.npz OUT.npz [--hops 3] [--weights IN.pt OUT.pt]

A decision crosses `hops` synapses on its way from the sensory neurons to the descending neurons (flyguy/fly.py), so a
connection only matters if its source can already carry the signal at some hop and its target can still reach a
descending neuron in the hops left. At 3 hops that is 3.0 million of the 15.3 million connections of the graph with
2+ synapses, and a file with only those is a fifth of the size. OUT holds
  every neuron exactly as before (body ids, classes, transmitters, the sensory / descending / motor / ascending
  lists: nothing is renumbered), the kept connections in their old order,
  in_total  every neuron's incoming synapse count in the WHOLE graph (the starting strengths are normalised by it,
            so a new network built from OUT starts exactly like one built from GRAPH),
  hops      the hop count it was cut for (a network of that many hops or fewer can be built from it; more cannot).
A network built from OUT is the same network: the same neurons at every hop, the connections in the same order.
--weights IN.pt OUT.pt: a checkpoint trained on GRAPH with its strengths cut down to the kept connections and its
config pointing at OUT. Its answers are compared with the original's on random game states before it is written.
Connections that are not kept never took part in a decision, so they were never trained either: nothing learned is
lost. (The fly's descending -> motor connections are among them: flyguy/body/export_bridge.py needs the whole graph
and a checkpoint of the whole graph.)
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flyguy.fly import FlyActor  # noqa: E402


def in_play(graph: Path, hops: int) -> np.ndarray:
    """True for every connection of the graph that some hop of a `hops`-hop network computes."""
    f = FlyActor(graph, 16, 2, hops=hops, fan=16)
    keep = np.zeros(f.n_conn, bool)
    for k in range(1, hops + 1):
        keep[getattr(f, f"e{k}").numpy()] = True
    assert int(keep.sum()) == f.in_play
    return keep


def trim(graph: Path, out: Path, hops: int) -> np.ndarray:
    keep = in_play(graph, hops)
    g = np.load(graph)
    assert "in_total" not in g.files, "this graph is trimmed already"
    total = np.zeros(len(g["body"]), np.int64)
    np.add.at(total, g["post"].astype(np.int64), g["syn"].astype(np.int64))
    arrays = {k: g[k] for k in g.files}
    for k in ("pre", "post", "syn"):
        arrays[k] = g[k][keep]
    arrays["in_total"] = total
    arrays["hops"] = np.int64(hops)
    arrays["whole_connections"] = np.int64(len(keep))
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **arrays)
    a, b = FlyActor(graph, 16, 2, hops=hops), FlyActor(out, 16, 2, hops=hops)      # the same network from either file
    new_index = np.cumsum(keep) - 1
    assert a.sizes == b.sizes and torch.equal(a.w.detach()[torch.as_tensor(keep)], b.w.detach())
    for k in range(hops + 1):
        assert torch.equal(getattr(a, f"n{k}"), getattr(b, f"n{k}"))
        if k:
            assert np.array_equal(new_index[getattr(a, f"e{k}").numpy()], getattr(b, f"e{k}").numpy())
            assert torch.equal(getattr(a, f"c{k}"), getattr(b, f"c{k}")) and torch.equal(getattr(a, f"r{k}"), getattr(b, f"r{k}"))
    print(f"{out}: {int(keep.sum()):,} of {len(keep):,} connections ({hops} hops), {out.stat().st_size / 2**20:.1f} MB "
          f"(was {graph.stat().st_size / 2**20:.1f}); neurons per hop {b.sizes}")
    return keep


def trim_weights(src: Path, dst: Path, keep: np.ndarray, graph_name: str) -> None:
    from flyguy.model import build_model
    ck = torch.load(src, map_location="cpu", weights_only=False)
    assert ck["model"]["actor.w"].numel() == len(keep), "this checkpoint was not trained on that graph"
    cfg = dict(ck["config"]["model"])
    new = dict(ck)
    new["model"] = dict(ck["model"])
    new["model"]["actor.w"] = ck["model"]["actor.w"][torch.as_tensor(keep)].clone()
    new["config"] = dict(ck["config"])
    new["config"]["model"] = dict(cfg, fly_graph=graph_name)
    whole, cut = build_model(cfg).eval(), build_model(new["config"]["model"]).eval()
    whole.load_state_dict(ck["model"], strict=False)
    cut.load_state_dict(new["model"], strict=False)
    torch.manual_seed(0)
    obs = torch.randn(64, whole.actor.obs_dim)
    with torch.no_grad():
        a, b = whole.actor(obs), cut.actor(obs)
    worst = float((a - b).abs().max())
    assert torch.equal(a.argmax(-1), b.argmax(-1)) and worst <= 1e-4 * float(a.abs().max()), worst
    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(new, dst)
    print(f"{dst}: {new['model']['actor.w'].numel():,} strengths, {dst.stat().st_size / 2**20:.1f} MB (was "
          f"{src.stat().st_size / 2**20:.1f}); largest difference in 64 x 90 action scores {worst:.2g}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("graph", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--hops", type=int, default=3)
    ap.add_argument("--weights", nargs=2, type=Path, metavar=("IN.pt", "OUT.pt"))
    args = ap.parse_args()
    keep = trim(args.graph, args.out, args.hops)
    if args.weights:
        try:
            name = args.out.resolve().relative_to(ROOT).as_posix()
        except ValueError:
            name = str(args.out.resolve())
        trim_weights(args.weights[0], args.weights[1], keep, name)
