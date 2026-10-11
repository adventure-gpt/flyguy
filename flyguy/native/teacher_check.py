# NOTE (public repository): this script drives the training harness flyguy was made with - a Rocket League
# simulator, a league of opponents and a PPO trainer (the package `cw` in the imports below), which are NOT
# part of this repository. It is here as the record of how flyguy was made and does not run on its own.
"""flyguy: will the data generator take this teacher? No GPU: the generator's config is loaded, its network built and
the teacher's file loaded into it, and a few made-up observations go through it and through the league's
older-architecture network (league.legacy_model) if the config names one.

  python flyguy/native/teacher_check.py GEN_CONFIG.toml TEACHER.pt

Before a teacher swap (10-10: V6 in V5's place) - the generator itself only says so after the teaching has stopped."""
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from cw.config import load, to_dict  # noqa: E402
from flyguy.model import build_model  # noqa: E402
from flyguy.obs import ObsBuilder  # noqa: E402

cfg = load(sys.argv[1])
ck = torch.load(sys.argv[2], map_location="cpu", weights_only=False)
m = build_model(to_dict(cfg)["model"]).eval()
m.load_state_dict(ck["model"])
obs_dim = ObsBuilder("cpu", ball_pred=cfg.model.ball_pred, opp_inputs=cfg.model.opp_inputs).obs_dim
with torch.no_grad():
    out = m.logits(torch.randn(8, obs_dim) * 0.3)
print(f"teacher {Path(sys.argv[2]).name}: {int(ck.get('steps', 0)):,} steps, actor {sum(p.numel() for p in m.actor.parameters()) / 1e6:.1f}M weights, "
      f"reads {obs_dim} inputs, answers {tuple(out.shape)}")
if cfg.league.legacy_model:
    lk = torch.load(ROOT / cfg.league.legacy_model, map_location="cpu", weights_only=False)
    old = build_model(lk["config"]["model"]).eval()
    old.load_state_dict(lk["model"])
    print(f"league's older network {Path(cfg.league.legacy_model).name}: actor {sum(p.numel() for p in old.actor.parameters()) / 1e6:.1f}M weights")
elif to_dict(cfg)["model"] != ck["config"]["model"]:
    print("NOTE: no league.legacy_model")
print("OK")
