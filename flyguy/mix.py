"""A point between two saved versions of flyguy: every weight of the actor = (1 - T) * A + T * B.
usage: python flyguy/mix.py A.pt B.pt T OUT.pt

Why (2026-10-11): the public copy of flyguy has to sit just under the RLBot bot pack's skill limit (a public bot may
not reach 42 goals before Nexto reaches 28), and no saved version does - the running average of the teaching run was
kept at native steps 170,414 (0.548 of the goals against Nexto) and 253,176 (0.643), nothing in between. Both are
running averages of the same run, so a point on the line between them is close to what the average was at a moment
in between; how it plays is measured, not assumed. Everything except the actor is taken from A.
"""
import sys
from pathlib import Path

import torch

a_path, b_path, t, out = Path(sys.argv[1]), Path(sys.argv[2]), float(sys.argv[3]), Path(sys.argv[4])
a = torch.load(a_path, map_location="cpu", weights_only=False)
b = torch.load(b_path, map_location="cpu", weights_only=False)
assert a["config"]["model"] == b["config"]["model"], "two different networks"
model = dict(a["model"])
for k, v in a["model"].items():
    if k.startswith("actor.") and v.is_floating_point():
        model[k] = torch.lerp(v.float(), b["model"][k].float(), t).to(v.dtype)
steps = round((1 - t) * int(a["steps"]) + t * int(b["steps"]))
ck = {"model": model, "steps": steps, "config": a["config"], "git": a.get("git"),
      "mix": {"a": a_path.name, "b": b_path.name, "t": t, "a_steps": int(a["steps"]), "b_steps": int(b["steps"])},
      "note": f"{1 - t:.2f} of {a_path.name} + {t:.2f} of {b_path.name}"}
out.parent.mkdir(parents=True, exist_ok=True)
torch.save(ck, out)
print(out, f"steps {steps:,} ({ck['note']})")
