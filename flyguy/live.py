"""Live-game adapter: RLBot v5 GamePacket -> canonical state arrays.

This is the only code that differs between training and live play. Everything downstream
(observations, policy, action table) is the exact code used in training.

Convention differences handled here (see docs/DESIGN.md section 4):
  - pad timer: RLBot reports seconds SINCE pickup; the canonical state stores seconds UNTIL
    available (cooldown - elapsed; 10 s big, 4 s small).
  - pad order: RLBot FieldInfo order is matched to the canonical order by coordinates.
  - dodge window: RLBot dodge_timeout (-1 on ground / expired, may exceed 1.25 s while the
    jump is held) -> can_dodge + dodge_time_left clamped to 1.25 s, matching RocketSim's
    HasFlipOrJump(): a car that left the ground without jumping keeps its flip indefinitely.
  - demo: demolished_timeout is -1 when not demolished.
  - orientation: Rotator (pitch, yaw, roll) -> forward/up axes (same convention as RocketSim).
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from .state import A, ARENA_DIM, C, CAR_DIM, DODGE_WINDOW, MAX_CARS, NUM_PADS, PAD_COOLDOWN, PAD_POSITIONS

AIR_ON_GROUND, AIR_JUMPING, AIR_DOUBLE_JUMPING, AIR_DODGING, AIR_IN_AIR = 0, 1, 2, 3, 4


def rotator_axes(pitch: float, yaw: float, roll: float) -> tuple[np.ndarray, np.ndarray]:
    """Forward and up unit vectors for a Rocket League rotator (radians)."""
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    cr, sr = math.cos(roll), math.sin(roll)
    forward = np.array([cp * cy, cp * sy, sp], np.float32)
    up = np.array([-cr * cy * sp - sr * sy, -cr * sy * sp + sr * cy, cp * cr], np.float32)
    return forward, up


def pad_mapping(field_pad_locations: list[tuple[float, float, float]]) -> np.ndarray:
    """canonical index for each RLBot pad index (matched by x, y)."""
    mapping = np.empty(len(field_pad_locations), dtype=np.int64)
    for i, (x, y, _) in enumerate(field_pad_locations):
        d = np.abs(PAD_POSITIONS[:, 0] - x) + np.abs(PAD_POSITIONS[:, 1] - y)
        j = int(np.argmin(d))
        if d[j] > 10.0:
            raise ValueError(f"RLBot pad {i} at ({x}, {y}) has no canonical match")
        mapping[i] = j
    if len(set(mapping.tolist())) != NUM_PADS:
        raise ValueError("RLBot pad layout does not match the canonical 34-pad soccar layout")
    return mapping


def _v(v: Any) -> tuple[float, float, float]:
    return (v.x, v.y, v.z)


class PacketAdapter:
    """Builds canonical (arena [1, AD], cars [1, 6, CD]) arrays from RLBot v5 packets.

    Slots follow the training convention: blue players (by packet index) then orange.
    """

    def __init__(self, field_info: Any):
        self.pad_map = pad_mapping([_v(p.location) for p in field_info.boost_pads])
        self.arena = np.zeros((1, ARENA_DIM), np.float32)
        self.cars = np.zeros((1, MAX_CARS, CAR_DIM), np.float32)
        self.slot_of_index: dict[int, int] = {}

    def slots(self, packet: Any) -> dict[int, int]:
        blue = [i for i, p in enumerate(packet.players) if p.team == 0]
        orange = [i for i, p in enumerate(packet.players) if p.team == 1]
        if len(blue) > MAX_CARS // 2 or len(orange) > MAX_CARS // 2:
            raise ValueError("more than 3 cars per team is not supported")
        return {idx: s for s, idx in enumerate(blue + orange)}

    def update(self, packet: Any) -> tuple[np.ndarray, np.ndarray]:
        arena, cars = self.arena, self.cars
        arena.fill(0.0)
        cars.fill(0.0)
        ball = packet.balls[0].physics
        arena[0, A.BALL_POS] = _v(ball.location)
        arena[0, A.BALL_VEL] = _v(ball.velocity)
        arena[0, A.BALL_ANGVEL] = _v(ball.angular_velocity)
        active = np.zeros(NUM_PADS, np.float32)
        left = np.zeros(NUM_PADS, np.float32)
        for i, pad in enumerate(packet.boost_pads):
            j = self.pad_map[i]
            active[j] = 1.0 if pad.is_active else 0.0
            left[j] = 0.0 if pad.is_active else max(0.0, float(PAD_COOLDOWN[j]) - pad.timer)
        arena[0, A.PAD_ACTIVE] = active
        arena[0, A.PAD_TIME_LEFT] = left

        self.slot_of_index = self.slots(packet)
        for idx, s in self.slot_of_index.items():
            p = packet.players[idx]
            c = cars[0, s]
            ph = p.physics
            c[C.PRESENT] = 1.0
            c[C.TEAM] = float(p.team)
            c[C.POS] = _v(ph.location)
            c[C.VEL] = _v(ph.velocity)
            c[C.ANGVEL] = _v(ph.angular_velocity)
            fwd, up = rotator_axes(ph.rotation.pitch, ph.rotation.yaw, ph.rotation.roll)
            c[C.FORWARD], c[C.UP] = fwd, up
            c[C.BOOST] = p.boost
            air = int(p.air_state)
            on_ground = air == AIR_ON_GROUND
            c[C.ON_GROUND] = float(on_ground)
            c[C.HAS_JUMPED] = float(p.has_jumped)
            c[C.HAS_DOUBLE_JUMPED] = float(p.has_double_jumped)
            c[C.HAS_FLIPPED] = float(p.has_dodged)
            c[C.IS_JUMPING] = float(air == AIR_JUMPING)
            c[C.IS_FLIPPING] = float(air == AIR_DODGING)
            if p.has_jumped:
                can = (not on_ground) and not p.has_dodged and not p.has_double_jumped and p.dodge_timeout > 0
                time_left = min(max(p.dodge_timeout, 0.0), DODGE_WINDOW)
            else:
                can = (not on_ground) and not p.has_dodged and not p.has_double_jumped
                time_left = DODGE_WINDOW
            c[C.CAN_DODGE] = float(can)
            c[C.DODGE_TIME_LEFT] = time_left if can else 0.0
            c[C.IS_SUPERSONIC] = float(p.is_supersonic)
            demoed = p.demolished_timeout >= 0
            c[C.IS_DEMOED] = float(demoed)
            c[C.DEMO_TIME_LEFT] = max(p.demolished_timeout, 0.0) if demoed else 0.0
        return arena, cars
