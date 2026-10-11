"""Set-structured actor and critic.

Each network: a shared per-car encoder applied to every other-car slot, masked mean+max
pooling separately over teammates and opponents, then an MLP trunk over
[global features, pooled teammates, pooled opponents]. One network therefore serves
1v1/2v2/3v3 and is invariant to slot order. Actor and critic share no parameters.

Optional extensions (ModelConfig flags), each added so that a checkpoint without it loads
with an unchanged function (new trunk inputs get zero weights, new residual paths start at 0):
  critic_extra  privileged value inputs (obs.critic_extra), critic only
  attention     self-attention over [self, ball, cars] tokens (actor; critic too unless
                attention_critic is off)
  memory        GRU memory in the actor between its first and second trunk layers
The trunk input is [global | critic extra | attention | pooled teammates, pooled opponents].
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .actions import NUM_ACTIONS, TABLE
from .obs import (BALL_PRED_DIM, BALL_TOKEN, CAR_FEAT_DIM, CRITIC_EXTRA_DIM, GLOBAL_DIM, LOCAL_BALL, OPP_INPUT_DIM,
                  OTHER_SLOTS, PRESENT_COL,
                  SELF_TOKEN, TEAMMATE_COL, split_obs)


class NormReLU(nn.Module):
    """LayerNorm, then ReLU - the hidden activation of a `model.norm` network (V6, 2026-10-09). It sits where the
    plain network has its nn.ReLU, so every Linear keeps its index in the Sequential. Why: in the plain network a
    unit whose input drifts negative for every state outputs 0, gets no gradient and never returns - by 101.5B a
    fifth of the actor's hidden units never fired and a third of its last hidden layer was dead
    (scripts/dormant_units.py). Normalising each layer's inputs keeps every unit's input centred, so a unit cannot
    drift out of reach."""

    def __init__(self, d: int):
        super().__init__()
        self.norm = nn.LayerNorm(d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.norm(x))


def _mlp(sizes: list[int], final_act: bool, norm: bool = False) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2 or final_act:
            layers.append(NormReLU(sizes[i + 1]) if norm else nn.ReLU())
    return nn.Sequential(*layers)


class _AttnBlock(nn.Module):
    """Pre-LayerNorm transformer block with a key-padding mask (plain SDPA: CUDA-graph friendly)."""

    def __init__(self, d: int, heads: int, ff: int):
        super().__init__()
        self.heads = heads
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.ff = nn.Sequential(nn.Linear(d, ff), nn.ReLU(), nn.Linear(ff, d))

    def forward(self, x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        n, t, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(n, t, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=valid[:, None, None, :])
        x = x + self.proj(o.transpose(1, 2).reshape(n, t, d))
        return x + self.ff(self.ln2(x))


class EntityAttention(nn.Module):
    """Self-attention over [self, ball, 5 other-car] tokens (absent cars masked out). Returns
    [self token, ball token, mean over present cars] -> [N, 3 * d]: pairwise relations (who is
    closest, who covers whom) that mean/max pooling cannot express."""

    def __init__(self, d: int, heads: int = 4, layers: int = 2, car_feat_dim: int = CAR_FEAT_DIM):
        super().__init__()
        n_self, n_ball = SELF_TOKEN.stop - SELF_TOKEN.start, BALL_TOKEN.stop - BALL_TOKEN.start
        self.self_in, self.ball_in = nn.Linear(n_self, d), nn.Linear(n_ball, d)
        self.car_in = nn.Linear(car_feat_dim, d)
        self.blocks = nn.ModuleList([_AttnBlock(d, heads, 2 * d) for _ in range(layers)])
        self.norm = nn.LayerNorm(d)

    def forward(self, glob: torch.Tensor, cars: torch.Tensor) -> torch.Tensor:
        tok = torch.cat([self.self_in(glob[:, SELF_TOKEN])[:, None], self.ball_in(glob[:, BALL_TOKEN])[:, None],
                         self.car_in(cars)], dim=1)
        present = cars[..., PRESENT_COL] > 0.5
        valid = torch.cat([torch.ones_like(present[:, :2]), present], dim=1)
        for block in self.blocks:
            tok = block(tok, valid)
        tok = self.norm(tok)
        m = present[..., None].to(tok.dtype)
        car_mean = (tok[:, 2:] * m).sum(1) / m.sum(1).clamp_min(1.0)
        return torch.cat([tok[:, 0], tok[:, 1], car_mean], dim=-1)


class SetNet(nn.Module):
    def __init__(self, enc: int, trunk: list[int], out: int, out_gain: float, global_dim: int = GLOBAL_DIM,
                 extra_dim: int = 0, attention: int = 0, memory: int = 0, car_feat_dim: int = CAR_FEAT_DIM,
                 norm: bool = False):
        super().__init__()
        self.global_dim, self.extra_dim, self.memory = global_dim, extra_dim, memory
        self.car_feat_dim = car_feat_dim
        if norm and memory:
            raise ValueError("model.norm is not written for a memory actor")
        att_dim = 3 * attention
        self.encoder = _mlp([car_feat_dim, enc, enc], final_act=True, norm=norm)
        self.trunk = _mlp([global_dim + extra_dim + att_dim + 4 * enc, *trunk, out], final_act=False, norm=norm)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, math.sqrt(2))
                nn.init.zeros_(m.bias)
        last = self.trunk[-1]
        nn.init.orthogonal_(last.weight, out_gain)
        # Optional modules are registered after encoder/trunk, so a plain network's parameter order
        # (and its optimizer state) is a prefix of the extended one's.
        self.attn = EntityAttention(attention, car_feat_dim=car_feat_dim) if attention else None
        if memory:
            if len(trunk) < 2:
                raise ValueError("memory needs at least two trunk layers")
            self.mem_in = nn.Linear(trunk[0], memory)
            self.gru = nn.GRUCell(memory, memory)
            self.mem_out = nn.Linear(memory, trunk[1])
            nn.init.zeros_(self.mem_out.weight)                    # memory starts silent: same function
            nn.init.zeros_(self.mem_out.bias)

    def _trunk_input(self, obs: torch.Tensor, extra: torch.Tensor | None) -> torch.Tensor:
        glob, cars = split_obs(obs, self.global_dim, self.car_feat_dim)
        e = self.encoder(cars)                                     # [N, 5, enc], >= 0 (ReLU)
        present = cars[..., PRESENT_COL:PRESENT_COL + 1]
        mate = cars[..., TEAMMATE_COL:TEAMMATE_COL + 1]
        pooled = []
        for m in (mate, present - mate):                           # teammates, opponents
            cnt = m.sum(1)
            mean = (e * m).sum(1) / cnt.clamp_min(1.0)
            mx = (e * m).amax(1)                                   # ReLU features: masked rows are 0
            pooled += [mean, mx]
        parts = [glob]
        if self.extra_dim:
            parts.append(extra.to(glob.dtype))
        if self.attn is not None:
            parts.append(self.attn(glob, cars).to(glob.dtype))
        return torch.cat(parts + pooled, dim=-1)

    def forward(self, obs: torch.Tensor, extra: torch.Tensor | None = None) -> torch.Tensor:
        if self.memory:
            raise RuntimeError("memory network: use step() or sequence()")
        return self.trunk(self._trunk_input(obs, extra))

    def _memory_tail(self, a1: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        z = self.trunk[2](a1) + self.mem_out(h.to(a1.dtype))
        for layer in self.trunk[3:]:
            z = layer(z)
        return z

    def step(self, obs: torch.Tensor, h: torch.Tensor, extra: torch.Tensor | None = None):
        """One decision for a memory network: (out, next hidden state)."""
        a1 = torch.relu(self.trunk[0](self._trunk_input(obs, extra)))
        h = self.gru(torch.relu(self.mem_in(a1)), h.to(a1.dtype))
        return self._memory_tail(a1, h), h

    def sequence(self, obs: torch.Tensor, h0: torch.Tensor, resets: torch.Tensor) -> torch.Tensor:
        """obs [L, B, D], h0 [B, M] (state before step 0), resets [L, B] (1 = episode starts at that
        step, clearing the state first) -> out [L, B, out]. Matches L calls of step()."""
        n_steps, b = obs.shape[:2]
        a1 = torch.relu(self.trunk[0](self._trunk_input(obs.flatten(0, 1), None)))
        x = torch.relu(self.mem_in(a1)).view(n_steps, b, -1)
        h = h0.to(x.dtype)
        hs = []
        for t in range(n_steps):
            h = h * (1.0 - resets[t]).to(h.dtype)[:, None]
            h = self.gru(x[t], h)
            hs.append(h)
        return self._memory_tail(a1, torch.stack(hs).flatten(0, 1)).view(n_steps, b, -1)


def widen_end(w: torch.Tensor, in_features: int) -> torch.Tensor:
    """Upgrade a per-car input weight (or its optimizer moment) saved with shorter car rows: the new
    per-car inputs are appended at the end of the row, so zero columns keep the function unchanged."""
    missing = in_features - w.shape[1]
    return w if missing <= 0 else torch.cat([w, w.new_zeros(w.shape[0], missing)], dim=1)


def widen_input(w: torch.Tensor, in_features: int, enc: int) -> torch.Tensor:
    """Upgrade a trunk input weight (or its optimizer moment) saved with a shorter input: new inputs
    always sit between the saved global block and the pooled car features, so zero columns there
    keep the saved network's function unchanged."""
    missing = in_features - w.shape[1]
    if missing <= 0:
        return w
    at = w.shape[1] - 4 * enc                                     # end of the saved global block
    assert at > 0, (tuple(w.shape), in_features, enc)
    return torch.cat([w[:, :at], w.new_zeros(w.shape[0], missing), w[:, at:]], dim=1)


def pad_hidden(t: torch.Tensor, shape) -> torch.Tensor:
    """A trunk weight or bias saved for narrower hidden layers, placed in the top-left corner of the wider
    shape with zeros elsewhere: the added units have no incoming and no outgoing weight, so the saved
    network's function is unchanged. This is how a wider learner's league loads older, narrower snapshots
    (the learner itself is widened by scripts/widen_model.py, whose new units do get incoming weights)."""
    if tuple(t.shape) == tuple(shape) or t.dim() != len(shape) or any(a > b for a, b in zip(t.shape, shape)):
        return t
    out = t.new_zeros(tuple(shape))
    out[tuple(slice(0, n) for n in t.shape)] = t
    return out


_OPTIONAL = ("attn.", "mem_in.", "gru.", "mem_out.")


class ActorCritic(nn.Module):
    def __init__(self, enc: int = 192, actor: list[int] | None = None, critic: list[int] | None = None,
                 ball_pred: bool = False, critic_extra: bool = False, attention: int = 0, memory: int = 0,
                 attention_critic: bool = True, opp_inputs: bool = False, norm: bool = False,
                 fly_graph: str = "", fly_hops: int = 3, fly_fan: int = 16, fly_kernel: str = "torch", fly_batch: int = 2048):
        super().__init__()
        self.norm = norm
        self.fly = bool(fly_graph)
        global_dim = GLOBAL_DIM + (BALL_PRED_DIM if ball_pred else 0)
        car_feat_dim = CAR_FEAT_DIM + (OPP_INPUT_DIM if opp_inputs else 0)
        self.ball_pred, self.critic_extra, self.memory, self.opp_inputs = ball_pred, critic_extra, memory, opp_inputs
        if fly_graph:                                        # flyguy: the actor is the fly's wiring (flyguy/fly.py)
            from .fly import FlyActor
            cls = FlyActor
            if fly_kernel == "native":                           # hops, backward pass and Adam in our own library (flyguy/fly_native.py)
                from .fly_native import NativeFlyActor as cls
            self.actor = cls(fly_graph, global_dim + OTHER_SLOTS * car_feat_dim, NUM_ACTIONS, hops=fly_hops, fan=fly_fan,
                             **({"max_batch": fly_batch} if fly_kernel == "native" else {}))
            self.actor.kernel = "cuda" if fly_kernel == "native" else fly_kernel    # (what it falls back to without the library)
            # this actor's forward pass IS the collection's cost: without a captured graph the environment sends
            # only the cars it drives through it (the trainer's environment), not the league opponents' as well
            self.split_act = True
        else:
            self.actor = SetNet(enc, actor or [1024, 1024, 512], NUM_ACTIONS, 0.01, global_dim,
                                attention=attention, memory=memory, car_feat_dim=car_feat_dim, norm=norm)
        self.critic = SetNet(enc, critic or [1024, 1024, 1024, 512], 1, 1.0, global_dim,
                             extra_dim=CRITIC_EXTRA_DIM if critic_extra else 0,
                             attention=attention if attention_critic else 0, car_feat_dim=car_feat_dim, norm=norm)

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        """Also loads checkpoints of a plainer architecture: widened inputs get zero weights and
        optional modules missing from the checkpoint keep their (function-preserving) init."""
        sd = dict(state_dict)
        for name in ("actor", "critic"):
            if not hasattr(getattr(self, name), "trunk"):    # the fly actor: nothing to widen
                continue
            first = getattr(self, name).trunk[0]
            enc = getattr(self, name).encoder[0].out_features
            k = f"{name}.trunk.0.weight"
            if k in sd:
                sd[k] = widen_input(sd[k], first.in_features, enc)
            # per-car inputs appended at the end of each car row (opp_inputs): zero columns there
            for k, layer in ((f"{name}.encoder.0.weight", getattr(self, name).encoder[0]),
                             (f"{name}.attn.car_in.weight", getattr(getattr(self, name).attn, "car_in", None))):
                if layer is not None and k in sd:
                    sd[k] = widen_end(sd[k], layer.in_features)
        own = self.state_dict()
        for k in sd:                                         # narrower hidden layers: zero-padded, same function
            if k.split(".", 1)[-1].startswith("trunk.") and k in own:
                sd[k] = pad_hidden(sd[k], own[k].shape)
        missing = set(own) - set(sd)
        allowed = {k for k in missing if k.split(".", 1)[1].startswith(_OPTIONAL)}
        if strict and missing - allowed:
            raise RuntimeError(f"checkpoint lacks non-optional weights: {sorted(missing - allowed)[:5]}")
        result = super().load_state_dict(sd, strict=False, assign=assign)
        if strict and result.unexpected_keys:
            raise RuntimeError(f"checkpoint has weights this architecture lacks: {result.unexpected_keys[:5]}")
        return result

    def logits(self, obs: torch.Tensor) -> torch.Tensor:
        return self.actor(obs).float()

    def step(self, obs: torch.Tensor, h: torch.Tensor):
        """Memory actor: (logits, next hidden state)."""
        out, h = self.actor.step(obs, h)
        return out.float(), h

    def value(self, obs: torch.Tensor, extra: torch.Tensor | None = None) -> torch.Tensor:
        return self.critic(obs, extra).float().squeeze(-1)


def build_model(m: dict) -> ActorCritic:
    """ActorCritic from a (checkpoint) model-config dict; missing flags mean the plain architecture."""
    return ActorCritic(m["enc"], m["actor"], m["critic"], ball_pred=m.get("ball_pred", False),
                       critic_extra=m.get("critic_extra", False), attention=m.get("attention", 0),
                       memory=m.get("memory", 0), attention_critic=m.get("attention_critic", True),
                       opp_inputs=m.get("opp_inputs", False), norm=m.get("norm", False),
                       fly_graph=m.get("fly_graph", ""), fly_hops=m.get("fly_hops", 3), fly_fan=m.get("fly_fan", 16),
                       fly_kernel=m.get("fly_kernel", "torch"), fly_batch=m.get("fly_batch", 2048))
