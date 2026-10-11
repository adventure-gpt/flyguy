"""Scripted speed-flip kickoffs (the approach top bots use), with parameters optimized in RocketSim.

A kickoff is a per-tick control schedule for the car that goes for the ball:
  1. boost straight for `t_straight` ticks, steering `steer` for the last `t_steer` of them
     (angles the car so the diagonal flip points at the ball)
  2. jump for `t_jump` ticks, then release for `t_gap` ticks (still boosting)
  3. dodge: jump with (pitch=-1 fwd, yaw=`flip_yaw`) for 2 ticks  -> diagonal front flip
  4. flip cancel: pitch=+1 (and roll=`roll`) for `t_cancel` ticks, boosting
  5. drive to the ball: steer at it, full throttle and boost, until first touch
Parameters are stored per spawn class for a blue car on the right-hand side (x >= 0); left-side
spawns mirror steer/yaw/roll. Orange cars see the same field rotated (team-canonical).
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

PARAMS_FILE = Path(__file__).resolve().parent / "kickoffs.json"

# Canonical (blue) spawns: class -> (x >= 0 representative, y)
SPAWNS = {"diagonal": (2048.0, -2560.0), "offcenter": (256.0, -3840.0), "center": (0.0, -4608.0)}


@dataclass
class KickoffParams:
    t_straight: int = 40
    steer: float = -0.5
    t_steer: int = 10
    t_jump: int = 6
    t_gap: int = 3
    flip_yaw: float = -0.7
    t_cancel: int = 40
    roll: float = 0.0


def spawn_class(x: float, y: float) -> tuple[str, float]:
    """Classify a (team-canonical, blue-side) spawn; returns (class, mirror sign for x)."""
    ax = abs(x)
    if ax > 1000:
        cls = "diagonal"
    elif ax > 100:
        cls = "offcenter"
    else:
        cls = "center"
    return cls, (1.0 if x >= 0 else -1.0)


def schedule(p: KickoffParams, mirror: float) -> list[np.ndarray]:
    """Per-tick controls [throttle, steer, pitch, yaw, roll, jump, boost, handbrake] up to the drive phase."""
    out = []
    for t in range(p.t_straight):
        steer = p.steer * mirror if t >= p.t_straight - p.t_steer else 0.0
        out.append(np.array([1, steer, 0, 0, 0, 0, 1, 0], np.float32))
    for _ in range(p.t_jump):
        out.append(np.array([1, 0, 0, 0, 0, 1, 1, 0], np.float32))
    for _ in range(p.t_gap):
        out.append(np.array([1, 0, 0, 0, 0, 0, 1, 0], np.float32))
    for _ in range(2):
        out.append(np.array([1, 0, -1, p.flip_yaw * mirror, 0, 1, 1, 0], np.float32))
    for _ in range(p.t_cancel):
        out.append(np.array([1, 0, 1, 0, p.roll * mirror, 0, 1, 0], np.float32))
    return out


def drive_to_ball(pos: np.ndarray, fwd: np.ndarray, up: np.ndarray, ball: np.ndarray) -> np.ndarray:
    right = np.cross(up, fwd)
    to = ball - pos
    ang = math.atan2(float(np.dot(to, right)), float(np.dot(to, fwd)))
    return np.array([1, float(np.clip(ang * 3, -1, 1)), 0, 0, 0, 0, 1, 0], np.float32)


def load_params() -> dict[str, KickoffParams]:
    if not PARAMS_FILE.exists():
        return {k: KickoffParams() for k in SPAWNS}
    raw = json.loads(PARAMS_FILE.read_text())
    return {k: KickoffParams(**v["params"]) for k, v in raw.items()}


def save_params(best: dict[str, tuple[KickoffParams, float]]) -> None:
    PARAMS_FILE.write_text(json.dumps({k: {"params": asdict(p), "time_to_ball": t} for k, (p, t) in best.items()}, indent=2))


class KickoffRunner:
    """Plays one scripted kickoff for one car, tick by tick (team-canonical inputs)."""

    def __init__(self, params: dict[str, KickoffParams] | None = None, max_ticks: int = 420):
        self.params = params or load_params()
        self.max_ticks = max_ticks
        self.active = False

    def start(self, canon_pos: np.ndarray) -> None:
        cls, mirror = spawn_class(float(canon_pos[0]), float(canon_pos[1]))
        self.sched = schedule(self.params[cls], mirror)
        self.tick = 0
        self.active = True

    def step(self, canon_pos: np.ndarray, fwd: np.ndarray, up: np.ndarray, ball: np.ndarray) -> np.ndarray | None:
        """Control for the next tick, or None once the script is over (hand back to the policy)."""
        if not self.active or self.tick >= self.max_ticks:
            self.active = False
            return None
        c = self.sched[self.tick] if self.tick < len(self.sched) else drive_to_ball(canon_pos, fwd, up, ball)
        self.tick += 1
        return c

    def stop(self) -> None:
        self.active = False
