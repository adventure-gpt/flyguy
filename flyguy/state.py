"""Canonical game state: the contract between the live-game adapter, the observation builder and the policy.
Field layouts as constants (the training code reads them from its simulator core).
All tensors are world frame: blue defends -y, orange defends +y.
"""
from __future__ import annotations

import numpy as np

ARENA_FIELDS: tuple[str, ...] = ('ball_pos0', 'ball_pos1', 'ball_pos2', 'ball_vel0', 'ball_vel1', 'ball_vel2', 'ball_angvel0', 'ball_angvel1', 'ball_angvel2', 'pad_active0', 'pad_active1', 'pad_active2', 'pad_active3', 'pad_active4', 'pad_active5', 'pad_active6', 'pad_active7', 'pad_active8', 'pad_active9', 'pad_active10', 'pad_active11', 'pad_active12', 'pad_active13', 'pad_active14', 'pad_active15', 'pad_active16', 'pad_active17', 'pad_active18', 'pad_active19', 'pad_active20', 'pad_active21', 'pad_active22', 'pad_active23', 'pad_active24', 'pad_active25', 'pad_active26', 'pad_active27', 'pad_active28', 'pad_active29', 'pad_active30', 'pad_active31', 'pad_active32', 'pad_active33', 'pad_time_left0', 'pad_time_left1', 'pad_time_left2', 'pad_time_left3', 'pad_time_left4', 'pad_time_left5', 'pad_time_left6', 'pad_time_left7', 'pad_time_left8', 'pad_time_left9', 'pad_time_left10', 'pad_time_left11', 'pad_time_left12', 'pad_time_left13', 'pad_time_left14', 'pad_time_left15', 'pad_time_left16', 'pad_time_left17', 'pad_time_left18', 'pad_time_left19', 'pad_time_left20', 'pad_time_left21', 'pad_time_left22', 'pad_time_left23', 'pad_time_left24', 'pad_time_left25', 'pad_time_left26', 'pad_time_left27', 'pad_time_left28', 'pad_time_left29', 'pad_time_left30', 'pad_time_left31', 'pad_time_left32', 'pad_time_left33', 'time_since_touch', 'episode_time')
CAR_FIELDS: tuple[str, ...] = ('present', 'team', 'pos0', 'pos1', 'pos2', 'vel0', 'vel1', 'vel2', 'angvel0', 'angvel1', 'angvel2', 'forward0', 'forward1', 'forward2', 'up0', 'up1', 'up2', 'boost', 'on_ground', 'has_jumped', 'has_double_jumped', 'has_flipped', 'is_jumping', 'is_flipping', 'can_dodge', 'dodge_time_left', 'is_supersonic', 'is_demoed', 'demo_time_left')
EVENT_FIELDS: tuple[str, ...] = ('touched', 'demo_inflicted', 'demoed', 'bump_inflicted')
ARENA_DIM, CAR_DIM, EVENT_DIM = len(ARENA_FIELDS), len(CAR_FIELDS), len(EVENT_FIELDS)
MAX_CARS: int = 6
NUM_PADS: int = 34

PAD_POSITIONS = np.asarray([[0.0, -4240.0, 70.0], [-1792.0, -4184.0, 70.0], [1792.0, -4184.0, 70.0], [-3072.0, -4096.0, 73.0], [3072.0, -4096.0, 73.0], [-940.0, -3308.0, 70.0], [940.0, -3308.0, 70.0], [0.0, -2816.0, 70.0], [-3584.0, -2484.0, 70.0], [3584.0, -2484.0, 70.0], [-1788.0, -2300.0, 70.0], [1788.0, -2300.0, 70.0], [-2048.0, -1036.0, 70.0], [2048.0, -1036.0, 70.0], [0.0, -1024.0, 70.0], [-3584.0, 0.0, 73.0], [-1024.0, 0.0, 70.0], [1024.0, 0.0, 70.0], [3584.0, 0.0, 73.0], [0.0, 1024.0, 70.0], [-2048.0, 1036.0, 70.0], [2048.0, 1036.0, 70.0], [-1788.0, 2300.0, 70.0], [1788.0, 2300.0, 70.0], [-3584.0, 2484.0, 70.0], [3584.0, 2484.0, 70.0], [0.0, 2816.0, 70.0], [-940.0, 3308.0, 70.0], [940.0, 3308.0, 70.0], [-3072.0, 4096.0, 73.0], [3072.0, 4096.0, 73.0], [-1792.0, 4184.0, 70.0], [1792.0, 4184.0, 70.0], [0.0, 4240.0, 70.0]], dtype=np.float32).reshape(NUM_PADS, 3)
PAD_IS_BIG = np.asarray([0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
PAD_COOLDOWN = np.where(PAD_IS_BIG > 0, 10.0, 4.0).astype(np.float32)


def _mirror_permutation(pos: np.ndarray) -> np.ndarray:
    """perm[i] = index of the pad located at (-x, -y) of pad i."""
    perm = np.empty(len(pos), dtype=np.int64)
    for i, (x, y, _) in enumerate(pos):
        d = np.abs(pos[:, 0] + x) + np.abs(pos[:, 1] + y)
        j = int(np.argmin(d))
        if d[j] > 1e-3:
            raise ValueError(f"pad layout is not point-symmetric at pad {i}")
        perm[i] = j
    return perm


PAD_MIRROR = _mirror_permutation(PAD_POSITIONS)


def _vec(fields: tuple[str, ...], prefix: str) -> slice:
    i = fields.index(prefix + "0")
    return slice(i, i + 3)


class A:
    """Arena field indices."""
    BALL_POS = _vec(ARENA_FIELDS, "ball_pos")
    BALL_VEL = _vec(ARENA_FIELDS, "ball_vel")
    BALL_ANGVEL = _vec(ARENA_FIELDS, "ball_angvel")
    PAD_ACTIVE = slice(ARENA_FIELDS.index("pad_active0"), ARENA_FIELDS.index("pad_active0") + NUM_PADS)
    PAD_TIME_LEFT = slice(ARENA_FIELDS.index("pad_time_left0"), ARENA_FIELDS.index("pad_time_left0") + NUM_PADS)
    TIME_SINCE_TOUCH = ARENA_FIELDS.index("time_since_touch")
    EPISODE_TIME = ARENA_FIELDS.index("episode_time")


class C:
    """Car field indices."""
    PRESENT = CAR_FIELDS.index("present")
    TEAM = CAR_FIELDS.index("team")
    POS = _vec(CAR_FIELDS, "pos")
    VEL = _vec(CAR_FIELDS, "vel")
    ANGVEL = _vec(CAR_FIELDS, "angvel")
    FORWARD = _vec(CAR_FIELDS, "forward")
    UP = _vec(CAR_FIELDS, "up")
    BOOST = CAR_FIELDS.index("boost")
    ON_GROUND = CAR_FIELDS.index("on_ground")
    HAS_JUMPED = CAR_FIELDS.index("has_jumped")
    HAS_DOUBLE_JUMPED = CAR_FIELDS.index("has_double_jumped")
    HAS_FLIPPED = CAR_FIELDS.index("has_flipped")
    IS_JUMPING = CAR_FIELDS.index("is_jumping")
    IS_FLIPPING = CAR_FIELDS.index("is_flipping")
    CAN_DODGE = CAR_FIELDS.index("can_dodge")
    DODGE_TIME_LEFT = CAR_FIELDS.index("dodge_time_left")
    IS_SUPERSONIC = CAR_FIELDS.index("is_supersonic")
    IS_DEMOED = CAR_FIELDS.index("is_demoed")
    DEMO_TIME_LEFT = CAR_FIELDS.index("demo_time_left")


class E:
    """Per-car event indices."""
    TOUCHED = EVENT_FIELDS.index("touched")
    DEMO_INFLICTED = EVENT_FIELDS.index("demo_inflicted")
    DEMOED = EVENT_FIELDS.index("demoed")
    BUMP_INFLICTED = EVENT_FIELDS.index("bump_inflicted")


# Field geometry (uu).
FIELD_X, FIELD_Y, FIELD_Z = 4096.0, 5120.0, 2044.0
GOAL_Y = 5120.0
GOAL_CENTER_Z = 321.0
CAR_MAX_SPEED = 2300.0
BALL_MAX_SPEED = 6000.0
DODGE_WINDOW = 1.25
DEMO_TIME = 3.0
TICK_RATE = 120
