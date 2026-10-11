"""flyguy: an actor wired like the fruit fly (a side project, 2026-10-09).

The network IS the male fly CNS connectome (Janelia FlyEM male-cns v1.0, CC-BY; graph file from
flyguy/build_graph.py): one unit per neuron, one trainable strength per connection, nothing else between them.
The only parts that are not fly are two thin plugs: each sensory neuron listens to a handful of observation
values (`fan` of them, fixed at random, one trainable weight each), and one linear map takes the descending
neurons to the action logits. All the processing in between happens in fly neurons over fly connections. It is
a network with a fly's wiring, taught by Whiskers; not a simulated fly.

A decision: the observation drives the sensory neurons, the signal crosses `hops` synapses, the action is read
off the descending neurons. A neuron fires relu(gain * input - threshold) with a threshold that cannot go below
zero, so a neuron the signal has not reached yet is silent - and therefore a connection is only worth computing
at hop k if its source can already carry the signal (within k-1 synapses of a sensory neuron) and its target
can still reach a descending neuron in the hops that are left. Skipping the rest changes nothing in the result
and removes most of the work. Every connection has ONE strength, shared by all the hops it is used in. Initial
strengths: proportional to the synapse count, normalised per receiving neuron, with the sign of the sending
neuron's transmitter (acetylcholine excites; GABA, glutamate and histamine inhibit) - a starting point only.

Speed: a hop is a sparse matrix times the batch of neuron states. PyTorch's own gradient for that product builds
the full dense targets x sources table (7 GB a step at 3 hops) - so the hop has its own backward here: the
transposed sparse product for the states, and the product sampled at the existing connections for the strengths.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
INHIBITORY = ("gaba", "glutamate", "histamine")


def _reach(start: np.ndarray, a: np.ndarray, b: np.ndarray, n: int, hops: int) -> list[np.ndarray]:
    """out[k][i]: neuron i is in `start` or within k steps of it along a -> b."""
    out = [np.zeros(n, bool)]
    out[0][start] = True
    for _ in range(hops):
        nxt = out[-1].copy()
        nxt[b[out[-1][a]]] = True
        out.append(nxt)
    return out


def _csr(rows: np.ndarray, n_rows: int) -> np.ndarray:
    """Row pointer of a CSR matrix whose entries are sorted by row."""
    return np.concatenate([[0], np.cumsum(np.bincount(rows, minlength=n_rows))]).astype(np.int64)


class _Hop(torch.autograd.Function):
    """z[b, target] = sum over connections (source -> target) of strength * h[b, source]."""

    @staticmethod
    def forward(ctx, w, h, crow, col, t_crow, t_col, t_perm):
        n_rows = crow.numel() - 1
        a = torch.sparse_csr_tensor(crow, col, w, size=(n_rows, h.shape[1]))
        ctx.save_for_backward(w, h, crow, col, t_crow, t_col, t_perm)
        return torch.sparse.mm(a, h.t()).t()

    @staticmethod
    def backward(ctx, gz):
        w, h, crow, col, t_crow, t_col, t_perm = ctx.saved_tensors
        n_rows = crow.numel() - 1
        gw = gh = None
        if ctx.needs_input_grad[0]:                                  # the product, only where a connection exists
            pat = torch.sparse_csr_tensor(crow, col, torch.zeros_like(w), size=(n_rows, h.shape[1]))
            gw = torch.sparse.sampled_addmm(pat, gz.t().contiguous(), h, beta=0.0, alpha=1.0).values()
        if ctx.needs_input_grad[1]:
            at = torch.sparse_csr_tensor(t_crow, t_col, w[t_perm], size=(h.shape[1], n_rows))
            gh = torch.sparse.mm(at, gz.t()).t()
        return gw, gh, None, None, None, None, None


class FlyActor(nn.Module):
    memory = 0

    def __init__(self, graph: str | Path, obs_dim: int, n_actions: int, hops: int = 3, fan: int = 16,
                 out_gain: float = 0.01, seed: int = 0):
        super().__init__()
        p = Path(graph)
        g = np.load(p if p.is_absolute() else ROOT / p)
        n = len(g["body"])
        pre, post, syn = g["pre"].astype(np.int64), g["post"].astype(np.int64), g["syn"].astype(np.float32)
        sens, dn = g["sensory"].astype(np.int64), g["descending"].astype(np.int64)
        self.n_neurons, self.n_conn, self.hops, self.obs_dim, self.fan = n, len(pre), hops, obs_dim, fan
        self.whole_graph = "in_total" not in g.files       # False: a trimmed file (flyguy/trim_graph.py), connections renumbered
        self.gather_limit = 60_000_000         # cars x connection steps up to which the plain gather path is used
        self.kernel = "torch"                  # "cuda" / "cuda16": the hand-written kernels of flyguy/fly_cuda.py (32-bit /
        #                                        16-bit states); set by whoever builds the network (model.fly_kernel)
        fwd = _reach(sens, pre, post, n, hops)                  # can carry the signal after k hops
        bwd = _reach(dn, post, pre, n, hops)                    # within j hops upstream of a descending neuron
        sets = [np.nonzero(fwd[k] & bwd[hops - k])[0] for k in range(hops + 1)]    # the neurons that matter after hop k
        self.sizes = [len(s) for s in sets]
        pos = []
        for s in sets:
            q = np.full(n, -1, np.int64)
            q[s] = np.arange(len(s))
            pos.append(q)
        is_sens = np.zeros(n, bool)
        is_sens[sens] = True
        self.used = 0
        in_play = np.zeros(len(pre), bool)

        def buf(name, a):
            self.register_buffer(name, torch.as_tensor(np.ascontiguousarray(a)), persistent=False)
        for k in range(1, hops + 1):
            m = np.nonzero((pos[k - 1][pre] >= 0) & (pos[k][post] >= 0))[0]
            r, c = pos[k][post[m]], pos[k - 1][pre[m]]
            o = np.lexsort((c, r))                              # by target, then source: CSR order
            m, r, c = m[o], r[o], c[o]
            t = np.lexsort((r, c))                              # by source, then target: the transposed matrix
            buf(f"e{k}", m); buf(f"r{k}", r); buf(f"c{k}", c)
            buf(f"crow{k}", _csr(r, len(sets[k]))); buf(f"tcrow{k}", _csr(c[t], len(sets[k - 1])))
            buf(f"tcol{k}", r[t]); buf(f"tperm{k}", t)
            buf(f"ci{k}", c.astype(np.int32)); buf(f"tci{k}", r[t].astype(np.int32))      # for the CUDA kernels
            buf(f"n{k}", sets[k])
            buf(f"s{k}", np.nonzero(is_sens[sets[k]])[0])                       # sensory neurons among this hop's
            buf(f"d{k}", pos[0][sets[k][is_sens[sets[k]]]])                     # ... and their place in hop 0
            self.used += len(m)
            in_play[m] = True
        self.in_play = int(in_play.sum())
        buf("n0", sets[0])
        ok = pos[hops][dn] >= 0
        assert ok.any(), "no descending neuron can be reached in this many hops"
        buf("dn_rows", pos[hops][dn[ok]])
        # the fly: one strength per connection, a gain and a threshold per neuron
        if "in_total" in g.files:                               # a trimmed graph (flyguy/trim_graph.py): only the connections
            #                                                     a network of g["hops"] hops uses, with every neuron's
            #                                                     incoming synapse count in the whole graph kept beside them
            assert hops <= int(g["hops"]), f"this graph file holds the connections of a {int(g['hops'])}-hop network only"
            total = g["in_total"].astype(np.float32)
        else:
            total = np.zeros(n, np.float32)
            np.add.at(total, post, syn)
        sign = np.where(np.isin(g["nt"][pre], INHIBITORY), -1.0, 1.0).astype(np.float32)
        self.w = nn.Parameter(torch.as_tensor(1.5 * sign * syn / np.maximum(total[post], 1.0)))
        self.gain = nn.Parameter(torch.ones(n))
        self.theta = nn.Parameter(torch.full((n,), -4.0))          # threshold = softplus(theta) >= 0 (0.018 at the start)
        # plug in: each sensory neuron in play listens to `fan` observation values (which ones: fixed at random)
        rng = np.random.default_rng(seed)
        n_s = self.sizes[0]
        cols = np.stack([np.sort(rng.choice(obs_dim, size=fan, replace=False)) for _ in range(n_s)]).reshape(-1)
        rows = np.repeat(np.arange(n_s), fan)
        t = np.lexsort((rows, cols))
        buf("in_r", rows); buf("in_c", cols); buf("in_crow", _csr(rows, n_s)); buf("in_tcrow", _csr(cols[t], obs_dim))
        buf("in_tcol", rows[t]); buf("in_tperm", t)
        buf("in_ci", cols.astype(np.int32)); buf("in_tci", rows[t].astype(np.int32))
        self.in_w = nn.Parameter(torch.randn(n_s * fan) / fan ** 0.5)
        self.in_b = nn.Parameter(torch.zeros(n_s))
        # plug out: descending neurons -> action logits
        self.out = nn.Linear(len(self.dn_rows), n_actions)
        nn.init.orthogonal_(self.out.weight, out_gain)
        nn.init.zeros_(self.out.bias)

    def describe(self) -> str:
        plug = self.in_w.numel() + self.in_b.numel() + sum(p.numel() for p in self.out.parameters())
        return (f"fly actor: {self.n_neurons:,} neurons and {self.n_conn:,} connections in the graph; {self.hops} hops: "
                f"{self.in_play:,} connections and {sum(self.sizes) - 0:,} neuron states in play (per hop "
                f"{self.sizes}), {self.used:,} connection steps a decision; learned outside the fly: {plug:,} numbers "
                f"(each sensory neuron hears {self.fan} observation values; {len(self.dn_rows):,} descending neurons -> actions)")

    def _spmm(self, k_name: str, w: torch.Tensor, h: torch.Tensor, n_rows: int, small: bool) -> torch.Tensor:
        r, c = getattr(self, f"{k_name}r" if k_name == "in_" else f"r{k_name}"), getattr(self, f"{k_name}c" if k_name == "in_" else f"c{k_name}")
        if small:
            return torch.zeros(h.shape[0], n_rows, dtype=h.dtype, device=h.device).index_add_(1, r, h[:, c] * w)
        g = (lambda s: getattr(self, f"in_{s}")) if k_name == "in_" else (lambda s: getattr(self, f"{s}{k_name}"))
        return _Hop.apply(w, h, g("crow"), c, g("tcrow"), g("tcol"), g("tperm"))

    def _forward_cuda(self, obs: torch.Tensor) -> torch.Tensor:
        """The same computation with states as [neurons, batch] and the kernels of flyguy/fly_cuda.py."""
        from .fly_cuda import HopCuda
        half = self.kernel == "cuda16"
        x = obs.t().contiguous()
        drive = HopCuda.apply(self.in_w, x, self.in_crow, self.in_ci, self.in_tcrow, self.in_tci, self.in_tperm, False) + self.in_b[:, None]
        thr = nn.functional.softplus(self.theta)
        h = torch.relu(self.gain[self.n0][:, None] * drive - thr[self.n0][:, None])
        for k in range(1, self.hops + 1):
            rows, g = getattr(self, f"n{k}"), (lambda name: getattr(self, f"{name}{k}"))
            z = HopCuda.apply(self.w[g("e")], h, g("crow"), g("ci"), g("tcrow"), g("tci"), g("tperm"), half)
            s_ = g("s")
            if len(s_):
                z = z.index_add(0, s_, drive[g("d")])
            h = torch.relu(self.gain[rows][:, None] * z - thr[rows][:, None])
        return self.out(h[self.dn_rows].t())

    def _native(self):
        """The compiled forward pass for playing (flyguy/native/fly_infer.dll: a hop walks only the connections of the
        neurons that fire), set up from the current weights. None when the DLL is not there or FLYGUY_NATIVE=0 - then
        the PyTorch path below plays. 10-09: RLBot counts the ticks a bot answers in time; a decision has to fit in
        one tick of 8.3 ms with everything else the bot does, and the PyTorch path alone took 5."""
        ver = (self.w._version, self.in_w._version, self.gain._version, self.theta._version, self.in_b._version,
               self.out.weight._version, self.out.bias._version)
        nat = getattr(self, "_nat", None)
        if nat is not None and nat["ver"] == ver:
            return nat if nat["h"] else None
        import ctypes
        import os
        if nat is not None and nat["h"]:
            nat["dll"].fly_free(nat["h"])
        nat = {"ver": ver, "h": None}
        self._nat = nat
        path = ROOT / "flyguy" / "native" / "fly_infer.dll"
        if os.environ.get("FLYGUY_NATIVE", "1") == "0" or not path.exists():
            return None
        try:
            dll = ctypes.CDLL(str(path))
        except OSError:
            return None
        vp, ci = ctypes.c_void_p, ctypes.c_int
        dll.fly_create.restype, dll.fly_create.argtypes = vp, [ci, ci, ci]
        dll.fly_set_plug.restype, dll.fly_set_plug.argtypes = None, [vp, ci, ci] + [vp] * 6
        dll.fly_set_hop.restype, dll.fly_set_hop.argtypes = None, [vp, ci, ci, ci, ci] + [vp] * 6
        dll.fly_set_out.restype, dll.fly_set_out.argtypes = None, [vp, ci, vp, vp, vp]
        dll.fly_forward.restype, dll.fly_forward.argtypes = None, [vp, vp, ci, vp, vp]
        dll.fly_free.restype, dll.fly_free.argtypes = None, [vp]

        def i32(t):
            return np.ascontiguousarray(t.detach().numpy() if torch.is_tensor(t) else t, dtype=np.int32)

        def f32(t):
            return np.ascontiguousarray(t.detach().numpy(), dtype=np.float32)

        def ptr(a):
            return a.ctypes.data_as(vp)
        with torch.no_grad():
            thr = nn.functional.softplus(self.theta)
            h = dll.fly_create(self.obs_dim, self.out.out_features, self.hops)
            a = [i32(self.in_crow), i32(self.in_ci), f32(self.in_w), f32(self.in_b), f32(self.gain[self.n0]), f32(thr[self.n0])]
            dll.fly_set_plug(h, self.sizes[0], len(a[1]), *map(ptr, a))
            for k in range(1, self.hops + 1):
                rows = getattr(self, f"n{k}")
                drow = np.full(len(rows), -1, np.int32)
                drow[getattr(self, f"s{k}").numpy()] = getattr(self, f"d{k}").numpy()
                a = [i32(getattr(self, f"tcrow{k}")), i32(getattr(self, f"tci{k}")), f32(self.w[getattr(self, f"e{k}")[getattr(self, f"tperm{k}")]]),
                     f32(self.gain[rows]), f32(thr[rows]), drow]              # (the connections by SOURCE neuron)
                dll.fly_set_hop(h, k - 1, len(rows), self.sizes[k - 1], len(a[1]), *map(ptr, a))
            a = [i32(self.dn_rows), f32(self.out.weight), f32(self.out.bias)]
            dll.fly_set_out(h, len(a[0]), *map(ptr, a))
        nat.update(h=h, dll=dll, ptr=ptr, shade=np.zeros(int(sum(self.sizes)), np.uint8))
        return nat

    def _forward_native(self, obs: torch.Tensor, nat: dict) -> torch.Tensor:
        x = np.ascontiguousarray(obs.detach().numpy(), dtype=np.float32)
        out = np.empty((x.shape[0], self.out.out_features), np.float32)
        tap = getattr(self, "tap", None)                            # the visualizer: every neuron's firing for the first car
        nat["dll"].fly_forward(nat["h"], nat["ptr"](x), x.shape[0], nat["ptr"](out), nat["ptr"](nat["shade"]) if tap is not None else None)
        if tap is not None:
            tap(nat["shade"])
        return torch.from_numpy(out)

    @torch.no_grad()
    def _forward_cpu(self, obs: torch.Tensor) -> torch.Tensor:
        """Playing, not learning, on the CPU (the RLBot bot): the hops as sparse matrices built ONCE from the current
        strengths. 10-09: through the gather path a decision took 52 ms of the bot's 67 and RLBot showed it at 50%."""
        nat = self._native()
        if nat is not None:
            return self._forward_native(obs, nat)
        ver = (self.w._version, self.in_w._version, self.gain._version, self.theta._version, self.in_b._version)
        c = getattr(self, "_cpu", None)
        if c is None or c["ver"] != ver:
            thr = nn.functional.softplus(self.theta)
            c = {"ver": ver, "a_in": torch.sparse_csr_tensor(self.in_crow, self.in_c, self.in_w.detach().clone(), size=(self.sizes[0], self.obs_dim)),
                 "g0": self.gain[self.n0][:, None].clone(), "t0": thr[self.n0][:, None].clone(), "hop": []}
            for k in range(1, self.hops + 1):
                rows = getattr(self, f"n{k}")
                a = torch.sparse_csr_tensor(getattr(self, f"crow{k}"), getattr(self, f"c{k}"), self.w.detach()[getattr(self, f"e{k}")],
                                            size=(len(rows), self.sizes[k - 1]))
                c["hop"].append((a, self.gain[rows][:, None].clone(), thr[rows][:, None].clone(), getattr(self, f"s{k}"), getattr(self, f"d{k}")))
            self._cpu = c
        drive = torch.sparse.mm(c["a_in"], obs.t().contiguous()) + self.in_b[:, None]
        h = torch.relu(c["g0"] * drive - c["t0"])
        tap = getattr(self, "tap", None)                            # the visualizer: every layer's firing for the first car
        acts = [h[:, 0]] if tap is not None else None
        for a, g, t, s_, d_ in c["hop"]:
            z = torch.sparse.mm(a, h)
            if len(s_):
                z[s_] += drive[d_]
            h = torch.relu(g * z - t)
            if acts is not None:
                acts.append(h[:, 0])
        if tap is not None:
            tap(acts)
        return self.out(h[self.dn_rows].t())

    def forward(self, obs: torch.Tensor, extra: torch.Tensor | None = None) -> torch.Tensor:
        obs = obs.float()
        if not obs.is_cuda and not torch.is_grad_enabled():
            return self._forward_cpu(obs)
        small = obs.shape[0] * self.used <= self.gather_limit       # few cars (the RLBot bot, tests): plain gather / scatter
        if self.kernel != "torch" and obs.is_cuda and not small and obs.shape[0] % 32 == 0:
            from .fly_cuda import available
            if available():
                return self._forward_cuda(obs)
        drive = self._spmm("in_", self.in_w, obs, self.sizes[0], small) + self.in_b      # [B, sensory in play]
        thr = nn.functional.softplus(self.theta)
        h = torch.relu(self.gain[self.n0] * drive - thr[self.n0])   # hop 0: the sensory neurons answer the game state
        for k in range(1, self.hops + 1):
            rows = getattr(self, f"n{k}")
            z = self._spmm(str(k), self.w[getattr(self, f"e{k}")], h, len(rows), small)
            s = getattr(self, f"s{k}")
            if len(s):                                              # sensory neurons stay driven by the game state
                z = z.index_add(1, s, drive[:, getattr(self, f"d{k}")])
            h = torch.relu(self.gain[rows] * z - thr[rows])
        return self.out(h[:, self.dn_rows])


class VizTap:
    """The visualizer's window into a playing flyguy (flyguy/viz): after every decision the firing of every neuron
    in play, layer by layer, goes into a small shared file as one byte a neuron, with the chosen action. Whoever
    wants to draw it (flyguy/viz/server.py) reads the file; nothing waits for a reader.

    File: "FLYV", version, seq (odd while being written), tick, time (f64), action, 8 controls (f32), layer count,
    4 layer sizes, 90 action probabilities (f32) from byte 80; firing bytes from byte 512, the layers one after
    another (sensory first), 255 * tanh(1.5 * firing)."""

    HEADER = 512

    def __init__(self, sizes: list[int], path: str | Path | None = None):
        import mmap
        import struct
        import tempfile
        self.struct = struct
        self.sizes = [int(s) for s in sizes]
        self.path = Path(path) if path else Path(tempfile.gettempdir()) / "flyguy_viz.bin"
        total = self.HEADER + sum(self.sizes)
        # (10-09: a file of the right size is taken as it is - writing a new one fails on Windows while the
        # visualizer's server has the old one mapped, and the bot then played without its tap)
        if not (self.path.exists() and self.path.stat().st_size == total):
            with open(self.path, "wb") as fh:
                fh.write(b"\0" * total)
        self.fh = open(self.path, "r+b")
        self.mm = mmap.mmap(self.fh.fileno(), total)
        self.mm[0:8] = b"FLYV" + struct.pack("<I", 1)
        self.mm[60:80] = struct.pack("<5I", len(self.sizes), *(self.sizes + [0] * 4)[:4])
        self.seq = 0
        self.tick = 0
        self.acts = None

    def __call__(self, acts: list[torch.Tensor]) -> None:
        self.acts = acts

    def publish(self, action: int, controls, probs, clock: float = -1.0, frame: int = 0, overtime: bool = False) -> None:
        """clock: the match clock in seconds (what the scoreboard counts down), frame: the game's frame number - at
        byte 440 (f32, u32, i32), so a recording of the page can be lined up with a recording of the game."""
        if self.acts is None:
            return
        self.mm[440:452] = self.struct.pack("<fIi", float(clock), int(frame) & 0xFFFFFFFF, int(bool(overtime)))
        import time
        pack = self.struct.pack
        self.seq += 1
        self.mm[8:12] = pack("<I", self.seq)
        at = self.HEADER
        if isinstance(self.acts, np.ndarray):                      # from the compiled forward pass: the bytes as they go out
            self.mm[at:at + len(self.acts)] = self.acts.tobytes()
        else:
            for a in self.acts:
                b = (torch.tanh(a * 1.5) * 255.0).to(torch.uint8).numpy().tobytes()
                self.mm[at:at + len(b)] = b
                at += len(b)
        self.tick += 1
        self.mm[12:28] = pack("<Idi", self.tick, time.time(), int(action))
        self.mm[28:60] = np.asarray(controls, np.float32)[:8].tobytes()
        p = np.asarray(probs, np.float32)[:90]
        self.mm[80:80 + 4 * len(p)] = p.tobytes()
        self.seq += 1
        self.mm[8:12] = pack("<I", self.seq)
        self.acts = None
