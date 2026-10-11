"""A small copy of a training state for the RLBot bot: weights, step count and config only.
usage: python flyguy/slim.py STATE.pt OUT.pt"""
import sys
import time
from pathlib import Path

import torch

src, out = Path(sys.argv[1]), Path(sys.argv[2])
ck = torch.load(src, map_location="cpu", weights_only=False)
slim = {"model": ck["model"], "steps": ck["steps"], "config": ck["config"], "git": ck.get("git"),
        "note": "slim copy of " + src.name + " saved " + time.strftime("%m-%d %H:%M", time.localtime(src.stat().st_mtime))}
out.parent.mkdir(parents=True, exist_ok=True)
torch.save(slim, out)
print(out, f"{out.stat().st_size / 2**20:.0f} MB, steps {int(ck['steps']):,}")
