"""Observation builder: canonical state -> per-agent observation vector.

Written once, used everywhere: training calls it on the GPU for thousands of agents per
step; the live RLBot bot calls the same code on the CPU for its 1-3 cars.

Frame: team-canonical. Orange agents see the world rotated 180 degrees about z (x, y
negated, pads permuted), so every agent attacks +y. The rotation is proper, so all
vectors (positions, velocities, angular velocities, orientation axes) transform alike.

Layout (OBS_DIM = GLOBAL_DIM + OTHER_SLOTS * CAR_FEAT_DIM):
  global [273]: self kinematics/status, ball absolute + relative, goals, pads, prev controls,
                then every boost pad relative to self in the car frame (+ distance)
                [+24 with ball_pred: predicted ball position at 0.5/1/1.5/2 s, absolute and
                relative to self in the car frame]
  others [5 x 33]: one row per other car slot; column 0 = present, column 1 = is_teammate
                [+11 with opp_inputs: that car's last controls and its has-jumped / has-double-jumped /
                is-flipping flags, what a human reads from boost trails and jumps]

New blocks are only ever appended to the end of the global block, so a checkpoint trained on a
shorter layout loads with zero weights on the new inputs (see model.widen_input).
"""
from __future__ import annotations

import torch

from .state import (A, C, DEMO_TIME, DODGE_WINDOW, FIELD_X, FIELD_Y, FIELD_Z, GOAL_CENTER_Z, GOAL_Y,
                    MAX_CARS, NUM_PADS, PAD_MIRROR, PAD_POSITIONS)

BASE_GLOBAL_DIM = 137        # layout before the relative-pad block; older checkpoints expect this
GLOBAL_DIM = BASE_GLOBAL_DIM + 4 * NUM_PADS
CAR_FEAT_DIM = 33
OPP_INPUT_DIM = 11           # other cars' last controls (8) + has_jumped, has_double_jumped, is_flipping
OTHER_SLOTS = MAX_CARS - 1
OBS_DIM = GLOBAL_DIM + OTHER_SLOTS * CAR_FEAT_DIM
PRESENT_COL, TEAMMATE_COL = 0, 1
LOCAL_BALL = slice(45, 48)   # ball position relative to self, in the car frame (fwd, right, up) / DIST
SELF_TOKEN = slice(0, 33)    # self kinematics and status (entity-attention token)
BALL_TOKEN = slice(33, 55)   # ball absolute and relative to self (entity-attention token)
BALL_PRED_TIMES = (0.5, 1.0, 1.5, 2.0)
BALL_PRED_DIM = 6 * len(BALL_PRED_TIMES)
# Kinds of opponent: an input of the training-time value function only, never of the policy.
OPP_TYPES = ("kind0", "kind1", "kind2", "kind3", "kind4", "kind5", "kind6", "kind7", "kind8", "kind9")
CRITIC_EXTRA_DIM = OTHER_SLOTS * 14 + 3 + len(OPP_TYPES)

BALL_R, GRAVITY, BALL_RESTITUTION, BALL_DRAG = 92.75, 650.0, 0.6, 0.03

DIST = 5000.0
VEL = 2300.0
BALL_VEL = 6000.0
ANG = 5.5
BALL_ANG = 6.0

_SELF_FLAGS = [C.ON_GROUND, C.HAS_JUMPED, C.HAS_DOUBLE_JUMPED, C.HAS_FLIPPED, C.IS_JUMPING,
               C.IS_FLIPPING, C.CAN_DODGE, C.IS_SUPERSONIC, C.IS_DEMOED]
_OTHER_FLAGS = [C.ON_GROUND, C.CAN_DODGE, C.IS_SUPERSONIC, C.IS_DEMOED]
_JUMP_STATE = [C.HAS_JUMPED, C.HAS_DOUBLE_JUMPED, C.HAS_FLIPPED, C.IS_JUMPING, C.IS_FLIPPING]
_OPP_FLAGS = [C.HAS_JUMPED, C.HAS_DOUBLE_JUMPED, C.IS_FLIPPING]


def _fold(x: torch.Tensor, lim: float) -> torch.Tensor:
    """Reflect x into [-lim, lim] (elastic bounces off two walls at +-lim)."""
    m = torch.remainder(x + lim, 4 * lim)
    return torch.where(m < 2 * lim, m - lim, 3 * lim - m)


def ball_path(pos: torch.Tensor, vel: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
    """Closed-form ball flight: pos/vel [N, 3], times [K] -> positions [N, K, 3]. Gravity, linear air
    drag, one floor bounce (then resting), side and back walls as elastic mirrors (goals and ramps
    are ignored). A cheap hint of where the ball is going, not a simulator."""
    t = times[None, :]
    f = (1.0 - torch.exp(-BALL_DRAG * t)) / BALL_DRAG
    x = _fold(pos[:, 0:1] + vel[:, 0:1] * f, FIELD_X - BALL_R)
    y = _fold(pos[:, 1:2] + vel[:, 1:2] * f, FIELD_Y - BALL_R)
    z0 = pos[:, 2:3].clamp_min(BALL_R)
    vz = vel[:, 2:3]
    tb = (vz + torch.sqrt(vz * vz + 2.0 * GRAVITY * (z0 - BALL_R))) / GRAVITY    # first floor contact
    z_free = z0 + vz * t - 0.5 * GRAVITY * t * t
    rebound = -BALL_RESTITUTION * (vz - GRAVITY * tb)
    s = (t - tb).clamp_min(0.0)
    z = torch.where(t < tb, z_free, BALL_R + rebound * s - 0.5 * GRAVITY * s * s)
    return torch.stack([x, y, z.clamp(BALL_R, FIELD_Z - BALL_R)], dim=-1)


class ObsBuilder:
    def __init__(self, device: torch.device | str, ball_pred: bool = False, opp_inputs: bool = False):
        device = torch.device(device)
        self.device = device
        self.ball_pred, self.opp_inputs = ball_pred, opp_inputs
        self.global_dim = GLOBAL_DIM + (BALL_PRED_DIM if ball_pred else 0)
        self.car_feat_dim = CAR_FEAT_DIM + (OPP_INPUT_DIM if opp_inputs else 0)
        self.obs_dim = self.global_dim + OTHER_SLOTS * self.car_feat_dim
        self.opp_flags = torch.tensor(_OPP_FLAGS, dtype=torch.long, device=device)
        self.pred_times = torch.tensor(BALL_PRED_TIMES, device=device)
        self.jump_state = torch.tensor(_JUMP_STATE, dtype=torch.long, device=device)
        self.other_slots = torch.tensor(
            [[j for j in range(MAX_CARS) if j != s] for s in range(MAX_CARS)], dtype=torch.long, device=device)
        self.pad_mirror = torch.as_tensor(PAD_MIRROR, dtype=torch.long, device=device)
        self.pos_scale = torch.tensor([FIELD_X, FIELD_Y, FIELD_Z], device=device)
        self.mirror = torch.tensor([-1.0, -1.0, 1.0], device=device)
        self.opp_goal = torch.tensor([0.0, GOAL_Y, GOAL_CENTER_Z], device=device)
        self.own_goal = torch.tensor([0.0, -GOAL_Y, GOAL_CENTER_Z], device=device)
        self.self_flags = torch.tensor(_SELF_FLAGS, dtype=torch.long, device=device)
        self.other_flags = torch.tensor(_OTHER_FLAGS, dtype=torch.long, device=device)
        # After the orange pad permutation, pad slot i sits at PAD_POSITIONS[i] in either team's frame.
        self.pad_pos = torch.as_tensor(PAD_POSITIONS, device=device)

    @torch.no_grad()
    def build(self, arena: torch.Tensor, cars: torch.Tensor, prev_controls: torch.Tensor,
              agent_arena: torch.Tensor, agent_slot: torch.Tensor,
              car_controls: torch.Tensor | None = None) -> torch.Tensor:
        """arena [A, ARENA_DIM], cars [A, 6, CAR_DIM], prev_controls [N, 8],
        agent_arena/agent_slot [N] (long)  ->  obs [N, OBS_DIM] float32.
        car_controls [A, 6, 8]: every car's last controls for opp_inputs (default: scattered from
        prev_controls, which covers every car when all agents are passed, as in training)."""
        me = cars[agent_arena, agent_slot]                                   # [N, CD]
        ar = arena[agent_arena]                                              # [N, AD]
        oth = cars[agent_arena[:, None], self.other_slots[agent_slot]]      # [N, 5, CD]
        team = me[:, C.TEAM]
        orange = team > 0.5
        o1 = orange[:, None]
        o2 = orange[:, None, None]

        def flip(v: torch.Tensor, o: torch.Tensor) -> torch.Tensor:
            return torch.where(o, v * self.mirror, v)

        pos = flip(me[:, C.POS], o1)
        vel = flip(me[:, C.VEL], o1)
        angvel = flip(me[:, C.ANGVEL], o1)
        fwd = flip(me[:, C.FORWARD], o1)
        up = flip(me[:, C.UP], o1)
        right = torch.linalg.cross(up, fwd, dim=-1)
        basis = torch.stack([fwd, right, up], dim=1)                        # [N, 3, 3]

        def local(v: torch.Tensor) -> torch.Tensor:                          # [N, 3] or [N, K, 3]
            if v.dim() == 2:
                return torch.einsum("nij,nj->ni", basis, v)
            return torch.einsum("nij,nkj->nki", basis, v)

        bpos = flip(ar[:, A.BALL_POS], o1)
        bvel = flip(ar[:, A.BALL_VEL], o1)
        bang = flip(ar[:, A.BALL_ANGVEL], o1)
        rel = bpos - pos
        rvel = bvel - vel

        pad_active = ar[:, A.PAD_ACTIVE]
        pad_left = ar[:, A.PAD_TIME_LEFT]
        pad_active = torch.where(o1, pad_active[:, self.pad_mirror], pad_active)
        pad_left = torch.where(o1, pad_left[:, self.pad_mirror], pad_left)
        prel = self.pad_pos - pos[:, None]                                    # [N, 34, 3]

        glob = torch.cat([
            pos / self.pos_scale, vel / VEL, angvel / ANG, fwd, up,
            local(vel) / VEL, local(angvel) / ANG,
            me[:, C.BOOST:C.BOOST + 1] / 100.0,
            me[:, self.self_flags],
            me[:, C.DODGE_TIME_LEFT:C.DODGE_TIME_LEFT + 1] / DODGE_WINDOW,
            me[:, C.DEMO_TIME_LEFT:C.DEMO_TIME_LEFT + 1] / DEMO_TIME,
            bpos / self.pos_scale, bvel / BALL_VEL, bang / BALL_ANG,
            rel / DIST, local(rel) / DIST, rvel / VEL, local(rvel) / VEL,
            rel.norm(dim=-1, keepdim=True) / DIST,
            local(self.opp_goal - pos) / DIST, local(self.own_goal - pos) / DIST,
            pad_active, pad_left / 10.0,
            prev_controls,
            (local(prel) / DIST).flatten(1), prel.norm(dim=-1) / DIST,
        ], dim=-1)
        if self.ball_pred:
            path = ball_path(bpos, bvel, self.pred_times)                       # [N, K, 3] canonical
            glob = torch.cat([glob, (path / self.pos_scale).flatten(1),
                              (local(path - pos[:, None]) / DIST).flatten(1)], dim=-1)

        opos = flip(oth[..., C.POS], o2)
        ovel = flip(oth[..., C.VEL], o2)
        ofwd = flip(oth[..., C.FORWARD], o2)
        oup = flip(oth[..., C.UP], o2)
        present = oth[..., C.PRESENT]
        mate = (oth[..., C.TEAM] == team[:, None]).float() * present
        orel = opos - pos[:, None]
        orvel = ovel - vel[:, None]
        oball = bpos[:, None] - opos
        parts = [
            present[..., None], mate[..., None],
            orel / DIST, local(orel) / DIST, orvel / VEL, local(orvel) / VEL,
            ovel / VEL, ofwd, oup,
            oth[..., C.BOOST:C.BOOST + 1] / 100.0,
            oth[..., self.other_flags],
            orel.norm(dim=-1, keepdim=True) / DIST,
            oball / DIST, oball.norm(dim=-1, keepdim=True) / DIST,
        ]
        if self.opp_inputs:
            # Every car's controls from the last step (prev_controls holds one row per agent, and
            # with agent_arena / agent_slot says whose); car-relative, so unchanged by the mirror.
            if car_controls is None:
                car_ctl = torch.zeros(arena.shape[0], MAX_CARS, 8, device=arena.device, dtype=prev_controls.dtype)
                car_ctl[agent_arena, agent_slot] = prev_controls
            else:
                car_ctl = car_controls
            parts += [car_ctl[agent_arena[:, None], self.other_slots[agent_slot]].to(orel.dtype),
                      oth[..., self.opp_flags]]
        others = torch.cat(parts, dim=-1) * present[..., None]

        return torch.cat([glob, others.flatten(1)], dim=-1)


    @torch.no_grad()
    def critic_extra(self, arena: torch.Tensor, cars: torch.Tensor, prev_controls: torch.Tensor,
                     agent_arena: torch.Tensor, agent_slot: torch.Tensor, opp_type: torch.Tensor,
                     kickoff: torch.Tensor) -> torch.Tensor:
        """Privileged value-function inputs [N, CRITIC_EXTRA_DIM], never seen by the actor: every
        other car's last controls and jump/flip state, the episode clocks, and which kind of
        opponent the learner faces (opp_type [N, len(OPP_TYPES)], one-hot)."""
        car_ctl = torch.zeros(arena.shape[0], MAX_CARS, 8, device=arena.device, dtype=prev_controls.dtype)
        car_ctl[agent_arena, agent_slot] = prev_controls
        slots = self.other_slots[agent_slot]
        oth = cars[agent_arena[:, None], slots]
        per_car = torch.cat([car_ctl[agent_arena[:, None], slots], oth[..., self.jump_state],
                             oth[..., C.DODGE_TIME_LEFT:C.DODGE_TIME_LEFT + 1] / DODGE_WINDOW], dim=-1)
        per_car = per_car * oth[..., C.PRESENT:C.PRESENT + 1]
        ar = arena[agent_arena]
        clocks = torch.stack([(ar[:, A.EPISODE_TIME] / 300.0).clamp(max=1.0), ar[:, A.TIME_SINCE_TOUCH] / 10.0,
                              kickoff[agent_arena]], dim=-1)
        return torch.cat([per_car.flatten(1), clocks, opp_type], dim=-1)


def split_obs(obs: torch.Tensor, global_dim: int = GLOBAL_DIM,
              car_feat_dim: int = CAR_FEAT_DIM) -> tuple[torch.Tensor, torch.Tensor]:
    """obs [N, global_dim + 5 * car_feat_dim] -> (global [N, global_dim], others [N, 5, car_feat_dim])."""
    return obs[:, :global_dim], obs[:, global_dim:].reshape(-1, OTHER_SLOTS, car_feat_dim)


assert NUM_PADS == 34
