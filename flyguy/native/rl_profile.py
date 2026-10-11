# NOTE (public repository): this script drives the training harness flyguy was made with - a Rocket League
# simulator, a league of opponents and a PPO trainer (the package `cw` in the imports below), which are NOT
# part of this repository. It is here as the record of how flyguy was made and does not run on its own.
"""flyguy RL: where an update's time goes. Builds the ordinary trainer from a run config, wraps the parts of an
iteration with timers (the GPU is waited for before and after each, so the figures are per part, slightly above the
unwatched run's) and runs a few iterations.

  python flyguy/native/rl_profile.py CONFIG.toml [--iters 2] [--set section.key=value ...] [--name NAME]

One line per part and iteration: seconds, calls. `fly forward (acting)` and `fly forward (learning)` are the native
library's forward pass, `fly backward` its backward pass, `fly step` Adam; the rest is the environment in PyTorch
(`obs`, `opponents`, `outside bots`, `critic`), the simulator (`wait for the simulator`) and what is left of each
phase (`acting: rest`, `learning: rest`)."""
import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from cw.config import load  # noqa: E402
from cw.train import Trainer  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("config")
ap.add_argument("--iters", type=int, default=2)
ap.add_argument("--set", nargs="*", default=[])
ap.add_argument("--name", default="flyguy_rl_profile")
args = ap.parse_args()
cfg = load(args.config)
for s in args.set:                                             # numbers and booleans only: enough for the sizes tried here
    k, v = s.split("=", 1)
    node = cfg
    *path, key = k.split(".")
    for p in path:
        node = getattr(node, p)
    old = getattr(node, key)
    setattr(node, key, type(old)(v) if not isinstance(old, bool) else v.lower() == "true")
cfg.run.name = args.name
tr = Trainer(cfg, ROOT / "runs" / args.name, resume=False)
T: dict[str, float] = defaultdict(float)
N: dict[str, int] = defaultdict(int)
sync = torch.cuda.synchronize if tr.device.type == "cuda" else (lambda: None)


def timed(obj, name: str, label):
    """Replace obj.name by a timed call; label is a string or a function of the arguments (for one that depends on
    whether gradients are on)."""
    f = getattr(obj, name)

    def g(*a, **k):
        sync()
        t = time.perf_counter()
        try:
            return f(*a, **k)
        finally:
            sync()
            key = label(*a, **k) if callable(label) else label
            T[key] += time.perf_counter() - t
            N[key] += 1
    try:
        setattr(obj, name, g)
    except Exception:  # noqa: BLE001
        object.__setattr__(obj, name, g)


timed(tr.env, "collect", "ACTING (whole)")
timed(tr.learner, "update", "LEARNING (whole)")
for h in tr.env.halves:
    timed(h, "run_act", "acting: a step's decisions (whole)")
    timed(h, "run_post", "acting: rewards and bookkeeping")
    timed(h, "pull", "acting: wait for the simulator")
    timed(h, "apply_resets", "acting: resets")
    for name, ext in list(h.external.items()):
        timed(ext, "controls", "outside bots")
if not cfg.env.use_graphs:                                    # (inside a captured graph nothing may wait for the GPU)
    timed(tr.env.obs_builder, "build", "obs")
actor = tr.ac.actor
if hasattr(actor, "_piece"):
    actor._ready()
    timed(actor, "_piece", lambda *a, **k: "fly forward (learning)" if torch.is_grad_enabled() or N["LEARNING (now)"] else "fly forward (acting)")
    timed(actor._lib, "fc_backward", "fly backward")
    timed(actor._lib, "fc_step", "fly step (Adam)")
if not cfg.env.use_graphs:
    timed(tr.ac, "value", lambda *a, **k: "critic (learning)" if N["LEARNING (now)"] else "critic")
for o in tr.opponents or []:
    timed(o, "logits", "opponents")
upd = tr.learner.update


def update(*a, **k):                                           # (the forward of a later piece runs inside backward(), without gradients)
    N["LEARNING (now)"] = 1
    try:
        return upd(*a, **k)
    finally:
        N["LEARNING (now)"] = 0


tr.learner.update = update
for it in range(args.iters):
    T.clear()
    keep = N.pop("LEARNING (now)", 0)
    N.clear()
    t0 = time.perf_counter()
    stats = tr.iterate()
    sync()
    total = time.perf_counter() - t0
    n = tr.buf.T * tr.buf.N
    print(f"--- iteration {it + 1}: {total:.1f} s for {n:,} samples = {n / total:,.0f} a second (fly batch {cfg.model.fly_batch}, "
          f"minibatch {cfg.ppo.minibatch}, {len(tr.env.halves)} halves of {tr.env.halves[0].n_agents} cars)", flush=True)
    act_parts = sum(v for k, v in T.items() if k in ("obs", "fly forward (acting)", "opponents", "outside bots", "critic"))
    T["acting: rest of the decisions"] = T["acting: a step's decisions (whole)"] - act_parts
    T["acting: rest"] = T["ACTING (whole)"] - sum(v for k, v in T.items() if k.startswith("acting: ") and "rest" not in k)
    T["learning: rest"] = T["LEARNING (whole)"] - sum(T[k] for k in ("fly forward (learning)", "fly backward", "fly step (Adam)"))
    for k in sorted(T, key=lambda k: (not k.isupper() and not k.startswith("A"), k)):
        if k != "LEARNING (now)":
            print(f"   {k:38s} {T[k]:7.2f} s {N.get(k, 0):7d} calls", flush=True)
    print("   " + " | ".join(f"{k} {float(v):.4f}" for k, v in stats.items() if k in ("entropy", "kl", "ev", "distill_kl", "distill_agree")), flush=True)
tr.env.close()
import os  # noqa: E402

os._exit(0)
