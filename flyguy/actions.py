"""Discrete action table.

Controls layout: throttle, steer, pitch, yaw, roll, jump, boost, handbrake.

Ground actions vary throttle, steer, boost and handbrake (boosting implies full throttle).
Air actions vary pitch, yaw, roll, jump and boost; yaw is dropped while jumping (roll covers
side flips) and the all-neutral air action is dropped because it duplicates a ground action.
Handbrake accompanies directional jumps so wavedashes/half-flips remain reachable.
Result: 24 ground + 66 air = 90 actions.
"""
from __future__ import annotations

import numpy as np

CONTROL_NAMES = ("throttle", "steer", "pitch", "yaw", "roll", "jump", "boost", "handbrake")


def make_table() -> np.ndarray:
    rows: list[list[float]] = []
    for throttle in (-1, 0, 1):
        for steer in (-1, 0, 1):
            for boost in (0, 1):
                for handbrake in (0, 1):
                    if boost == 1 and throttle != 1:
                        continue
                    rows.append([throttle, steer, 0, steer, 0, 0, boost, handbrake])
    for pitch in (-1, 0, 1):
        for yaw in (-1, 0, 1):
            for roll in (-1, 0, 1):
                for jump in (0, 1):
                    for boost in (0, 1):
                        if jump == 1 and yaw != 0:
                            continue
                        if pitch == 0 and roll == 0 and jump == 0:
                            continue
                        handbrake = float(jump == 1 and (pitch != 0 or yaw != 0 or roll != 0))
                        rows.append([boost, yaw, pitch, yaw, roll, jump, boost, handbrake])
    table = np.asarray(rows, dtype=np.float32)
    assert table.shape == (90, 8), table.shape
    return table


TABLE = make_table()
NUM_ACTIONS = len(TABLE)
