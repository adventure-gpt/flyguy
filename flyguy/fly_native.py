"""flyguy: the fly actor with its hops, their backward pass and Adam in our own CUDA library (2026-10-10).

Reinforcement learning needs the fly network inside the ordinary trainer: it acts in the simulator, PPO computes its
loss on the logits, and the gradient has to go back through 3 million connections. Through PyTorch routines that
ran at 1,450 samples a second and through Python-hosted kernels at 4,800; the teaching trainer's own kernels do
10,000. `flyguy_core.dll` (flyguy/native/flyguy_train.cu built with -DFLYGUY_CORE, build_core.bat) is those kernels
as a library in the trainer's own process and GPU context: observations and logits are PyTorch's GPU memory, PyTorch
computes the loss, and only the gradient of the logits is handed back.

NativeFlyActor IS a FlyActor (same weights, same checkpoint keys: a checkpoint loads into either and the bot plays
it through the CPU path as before). On the GPU with the library present:
  - forward without gradients (acting in the simulator): the library, in pieces of at most `max_batch` cars;
  - forward with gradients (PPO's minibatch): the library again; backward() then gives it the gradient of the logits
    and it ADDS the weights' gradients to its own buffers - for a minibatch larger than max_batch the pieces are sent
    through a second time one by one, their states do not fit together in the card's memory;
  - `native_adam()` is what the PPO learner uses as the actor's optimiser: zero_grad / step (clipped, three rates:
    strengths, neurons at 0.3x, plugs) in the library, and after every step the weights are copied back into the
    PyTorch tensors (14 MB on the card), so everything that reads the model - snapshots, averages, checkpoints - is
    current without knowing about any of this.
Without the library, or on the CPU, it behaves exactly as FlyActor."""
from __future__ import annotations

import ctypes
import os
import tempfile
from pathlib import Path

import numpy as np
import torch

from . import dist
from .fly import ROOT, FlyActor

DLL = ROOT / "flyguy" / "native" / ("flyguy_core.dll" if os.name == "nt" else "libflyguy_core.so")     # build_core.bat / build_core.sh


class _NativeLogits(torch.autograd.Function):
    @staticmethod
    def forward(ctx, obs, hook, actor):
        ctx.actor, ctx.obs = actor, obs
        out = actor._run(obs)
        ctx.gen = actor._gen                                    # (the states of the last piece are still in the library)
        return out

    @staticmethod
    def backward(ctx, g):
        a = ctx.actor
        n, mb = ctx.obs.shape[0], a.max_batch
        for i in range(0, n, mb):
            if not (n <= mb and a._gen == ctx.gen):             # more than one piece, or something ran in between: its states again
                a._piece(ctx.obs[i:i + mb])
            gi = g[i:i + mb]
            pad = (-gi.shape[0]) % 32
            if pad:
                gi = torch.cat([gi, gi.new_zeros(pad, gi.shape[1])])
            gi = gi.contiguous().float()
            a._lib.fc_backward(a._core, gi.data_ptr(), gi.shape[0])
        return None, None, None


class NativeAdam:
    """The actor's optimiser as the PPO learner sees it; the work is done by the library."""

    def __init__(self, actor: "NativeFlyActor", lr: float, max_norm: float = 0.0, ema: float = 0.999):
        self.actor, self.max_norm, self.ema = actor, max_norm, ema
        self.param_groups = [{"lr": lr}]
        self.grad_norm = 0.0

    def zero_grad(self, set_to_none: bool = True) -> None:
        a = self.actor
        if a._ready():
            a._lib.fc_zero_grad(a._core)

    def step(self) -> None:
        a = self.actor
        if not a._ready():
            return
        if dist.on():                                           # several GPUs: the mean of every rank's gradient
            g = a._fetch(4)
            dist.mean_(g)
            a._store(4, g)
        lr = float(self.param_groups[0]["lr"])
        self.grad_norm = float(a._lib.fc_step(a._core, lr, lr * a.lr_neuron, lr * a.lr_plug, float(self.max_norm), float(self.ema)))
        a.pull()

    def state_dict(self) -> dict:
        a = self.actor
        if not a._ready():
            return {"native": True, "adam_t": 0}
        out = {"native": True, "adam_t": int(a._lib.fc_adam_t(a._core, -1))}
        for name, which in (("ema", 1), ("m", 2), ("v", 3)):
            out[name] = [t.cpu() for t in a._fetch(which)]
        return out

    def load_state_dict(self, d: dict) -> None:
        a = self.actor
        if not d.get("native") or "m" not in d or not a._ready():   # (a PyTorch Adam's state: the library starts its own)
            return
        for name, which in (("ema", 1), ("m", 2), ("v", 3)):
            a._store(which, [t.to(a.gain.device) for t in d[name]])
        a._lib.fc_adam_t(a._core, int(d.get("adam_t", 0)))
        a._avg_follows = False                                  # (its own running average came with the state)


class NativeFlyActor(FlyActor):
    lr_neuron, lr_plug = 0.3, 1.0                               # rates of gain / threshold and of the plugs, as shares of the strengths'

    def __init__(self, *args, max_batch: int = 2048, **kw):
        super().__init__(*args, **kw)
        self.max_batch = int(max_batch)
        self._lib = self._core = None
        self._gen, self._stale, self._hook_t = 0, True, None
        self._avg_follows = True                                # the library's running average starts as the loaded weights

    def __deepcopy__(self, memo):
        """A copy is a plain FlyActor with the same weights (the trainer copies the model for its frozen league
        opponents): two Python objects must never share one network in the library."""
        import copy
        new = FlyActor.__new__(FlyActor)
        memo[id(self)] = new
        for k, v in self.__dict__.items():
            if k not in ("_lib", "_core", "_hook_t", "_wmap", "_w_play", "_gen", "_stale", "_avg_follows", "max_batch"):
                new.__dict__[k] = copy.deepcopy(v, memo)
        return new

    # ---- the library
    def _ready(self) -> bool:
        if self._core is not None:
            return True
        if not (self.gain.is_cuda and DLL.exists()):
            return False
        from flyguy.native.export import write_net               # the index arrays, from this actor's own buffers
        if hasattr(os, "add_dll_directory"):
            os.add_dll_directory(str(Path(torch.__file__).parent / "lib"))      # the CUDA runtime PyTorch ships
        lib = ctypes.CDLL(str(DLL))
        vp, ci = ctypes.c_void_p, ctypes.c_int
        lib.fc_create.restype, lib.fc_create.argtypes = vp, [ctypes.c_char_p, ci]
        lib.fc_sizes.restype, lib.fc_sizes.argtypes = None, [vp, ctypes.POINTER(ctypes.c_longlong)]
        lib.fc_copy.restype, lib.fc_copy.argtypes = None, [vp, ci, ci, ctypes.POINTER(vp)]
        lib.fc_forward.restype, lib.fc_forward.argtypes = None, [vp, vp, ci, vp, ci]
        lib.fc_zero_grad.restype, lib.fc_zero_grad.argtypes = None, [vp]
        lib.fc_backward.restype, lib.fc_backward.argtypes = None, [vp, vp, ci]
        lib.fc_grad_norm.restype, lib.fc_grad_norm.argtypes = ctypes.c_float, [vp]
        lib.fc_step.restype, lib.fc_step.argtypes = ctypes.c_float, [vp] + [ctypes.c_float] * 5
        lib.fc_adam_t.restype, lib.fc_adam_t.argtypes = ctypes.c_longlong, [vp, ctypes.c_longlong]
        torch.cuda.synchronize()
        with tempfile.TemporaryDirectory() as tmp:
            s = write_net(self, Path(tmp) / "net.bin")
            core = lib.fc_create(str(Path(tmp) / "net.bin").encode(), self.max_batch)
        sizes = (ctypes.c_longlong * 10)()
        lib.fc_sizes(core, sizes)
        dev = self.gain.device
        self._wmap = torch.as_tensor(s["wmap"], device=dev)
        self._w_play = torch.empty(len(s["wmap"]), device=dev)
        assert [int(v) for v in sizes[:7]] == [t.numel() for t in self._tensors()], "the library's network is not this actor's"
        self._lib, self._core = lib, core
        self._hook_t = torch.zeros(1, device=dev, requires_grad=True)
        self.push(also_average=True)
        return True

    def _tensors(self) -> list[torch.Tensor]:
        """The seven weight arrays in the library's order, as the PyTorch memory they are copied to and from."""
        return [self._w_play, self.gain.data, self.theta.data, self.in_w.data, self.in_b.data, self.out.weight.data, self.out.bias.data]

    def _ptrs(self, tensors):
        assert all(t.is_cuda and t.is_contiguous() and t.dtype == torch.float32 for t in tensors)
        return (ctypes.c_void_p * 7)(*[t.data_ptr() for t in tensors])

    def push(self, also_average: bool = False) -> None:
        """PyTorch's weights -> the library (after a checkpoint was loaded)."""
        self._w_play.copy_(self.w.data[self._wmap])
        self._lib.fc_copy(self._core, 0, 1, self._ptrs(self._tensors()))
        if also_average:
            self._lib.fc_copy(self._core, 1, 1, self._ptrs(self._tensors()))
        self._stale = False

    def pull(self) -> None:
        """The library's weights -> PyTorch's (after every optimiser step)."""
        self._lib.fc_copy(self._core, 0, 0, self._ptrs(self._tensors()))
        self.w.data[self._wmap] = self._w_play

    def _fetch(self, which: int) -> list[torch.Tensor]:
        out = [torch.empty_like(t) for t in self._tensors()]
        self._lib.fc_copy(self._core, which, 0, self._ptrs(out))
        return out

    def _store(self, which: int, tensors: list[torch.Tensor]) -> None:
        self._lib.fc_copy(self._core, which, 1, self._ptrs([t.contiguous().float() for t in tensors]))

    def _load_from_state_dict(self, *a, **k):
        super()._load_from_state_dict(*a, **k)
        self._stale = True                                      # (pushed before the next forward)

    def native_adam(self, lr: float, max_norm: float = 0.0) -> NativeAdam | None:
        return NativeAdam(self, lr, max_norm) if self._ready() else None

    # ---- forward
    def _piece(self, obs: torch.Tensor) -> torch.Tensor:
        b = obs.shape[0]
        pad = (-b) % 32
        x = obs if not pad else torch.cat([obs, obs[-1:].expand(pad, -1)])
        x = x.contiguous()
        out = torch.empty(x.shape[0], self.out.out_features, device=x.device)
        self._lib.fc_forward(self._core, x.data_ptr(), x.shape[0], out.data_ptr(), 0)
        self._gen += 1
        return out[:b]

    def _run(self, obs: torch.Tensor) -> torch.Tensor:
        if self._stale:
            self.push(also_average=self._avg_follows)
        if obs.shape[0] <= self.max_batch:
            return self._piece(obs)
        return torch.cat([self._piece(obs[i:i + self.max_batch]) for i in range(0, obs.shape[0], self.max_batch)])

    def forward(self, obs: torch.Tensor, extra: torch.Tensor | None = None) -> torch.Tensor:
        if not (obs.is_cuda and self._ready()):
            return super().forward(obs, extra)
        obs = obs.float()
        if torch.is_grad_enabled():
            return _NativeLogits.apply(obs, self._hook_t, self)
        return self._run(obs)


def grad_arrays(actor: NativeFlyActor) -> list[np.ndarray]:
    """The gradient the library has gathered since zero_grad (for checks against PyTorch's own)."""
    return [t.cpu().numpy() for t in actor._fetch(4)]
