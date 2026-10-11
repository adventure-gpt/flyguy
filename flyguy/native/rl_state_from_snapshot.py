# NOTE (public repository): this script drives the training harness flyguy was made with - a Rocket League
# simulator, a league of opponents and a PPO trainer (the package `cw` in the imports below), which are NOT
# part of this repository. It is here as the record of how flyguy was made and does not run on its own.
"""A run's full state (latest.pt) rebuilt around one of its step snapshots, so the trainer can resume from it.

  python flyguy/native/rl_state_from_snapshot.py FULL.pt SNAPSHOT.pt OUT.pt --iteration N

A step snapshot (runs/<run>/checkpoints/step_*.pt) holds the weights, their running averages and the step count -
not the optimisers, the return scale or the league's records, and `cw train --resume` needs those (10-10: going back
to flyguy's 140M save failed on exactly that). This takes them from a full state of the SAME run (the newest one,
set aside) and puts the snapshot's weights, averages and step count in. The fly actor's own optimiser state is left
out, so its Adam starts fresh on the old weights (its moments belonged to weights 70M samples further on); the
critic's optimiser, the return scale and the league's records are the later ones, which is harmless."""
import argparse

import torch

ap = argparse.ArgumentParser()
ap.add_argument("full"); ap.add_argument("snapshot"); ap.add_argument("out")
ap.add_argument("--iteration", type=int, required=True, help="the update number the snapshot was saved at (the run's log)")
args = ap.parse_args()
full = torch.load(args.full, map_location="cpu", weights_only=False)
snap = torch.load(args.snapshot, map_location="cpu", weights_only=False)
assert set(full["model"]) == set(snap["model"]), "not the same network"
was = (full["steps"], full["iteration"])
full["model"] = snap["model"]
full["steps"], full["iteration"] = int(snap["steps"]), args.iteration
if "avg" in snap:
    full["avg"] = {n: {k: v.to(full["avg"][n][k].dtype) if n in full.get("avg", {}) and k in full["avg"][n] else v for k, v in sd.items()} for n, sd in snap["avg"].items()}
opt = full["optim"]
fresh = []
for k, v in list(opt.items()) if isinstance(opt, dict) else []:
    if isinstance(v, dict) and v.get("native"):
        opt[k] = {"native": True, "adam_t": 0}
        fresh.append(k)
torch.save(full, args.out)
print(f"{args.out}: weights and averages of {args.snapshot} at {full['steps']:,} steps (update {full['iteration']}); the rest from the state at {was[0]:,} (update {was[1]}); "
      f"optimiser parts: {sorted(opt) if isinstance(opt, dict) else type(opt).__name__}; started fresh: {fresh or 'none found'}")
