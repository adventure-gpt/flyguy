"""flyguy: hand-written CUDA kernels for the fly actor's hops (2026-10-09).

Why: a hop is `connections x batch` multiply-adds over scattered memory, and so is each half of its backward pass;
measured with the library routines the card was moving ~120 GB a second for 2,000 samples - the limit is memory
traffic. These kernels keep the batch along the fast axis (states are [neurons, batch], so the 32 threads of a
warp read 32 neighbouring samples of ONE source neuron: coalesced), can read the states as 16-bit numbers (half
the traffic; sums are still accumulated in 32 bits) and compute the per-connection gradient with one warp
reduction and one atomic add per connection per 32 samples.

Compiled at run time by CuPy (NVRTC; `pip install cupy-cuda13x[ctk]`), sharing memory with PyTorch tensors. The
batch must be a multiple of 32. flyguy/fly.py falls back to the PyTorch routines when this is not available.
"""
from __future__ import annotations

import torch

try:
    import cupy as cp
except Exception:                                   # no CuPy here: the PyTorch path is used
    cp = None

_SRC = r'''
#include <cuda_fp16.h>
#define IDX long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x; \
            int t = (int)(idx / B); int b = (int)(idx - (long long)t * B); if (t >= n_rows) return;

extern "C" __global__ void spmm_f(const long long* crow, const int* col, const float* w, const float* h,
                                  float* out, int n_rows, int B) {
    IDX
    float acc = 0.f;
    for (long long e = crow[t]; e < crow[t + 1]; ++e) acc += w[e] * h[(long long)col[e] * B + b];
    out[idx] = acc;
}
extern "C" __global__ void spmm_h(const long long* crow, const int* col, const float* w, const __half* h,
                                  float* out, int n_rows, int B) {
    IDX
    float acc = 0.f;
    for (long long e = crow[t]; e < crow[t + 1]; ++e) acc += w[e] * __half2float(h[(long long)col[e] * B + b]);
    out[idx] = acc;
}
#define REDUCE v += __shfl_down_sync(0xffffffff, v, 16); v += __shfl_down_sync(0xffffffff, v, 8); \
               v += __shfl_down_sync(0xffffffff, v, 4);  v += __shfl_down_sync(0xffffffff, v, 2); \
               v += __shfl_down_sync(0xffffffff, v, 1);  if ((threadIdx.x & 31) == 0) atomicAdd(&gw[e], v);

extern "C" __global__ void sddmm_f(const long long* crow, const int* col, const float* gz, const float* h,
                                   float* gw, int n_rows, int B) {
    IDX
    float g = gz[idx];
    for (long long e = crow[t]; e < crow[t + 1]; ++e) { float v = g * h[(long long)col[e] * B + b]; REDUCE }
}
extern "C" __global__ void sddmm_h(const long long* crow, const int* col, const __half* gz, const __half* h,
                                   float* gw, int n_rows, int B) {
    IDX
    float g = __half2float(gz[idx]);
    for (long long e = crow[t]; e < crow[t + 1]; ++e) { float v = g * __half2float(h[(long long)col[e] * B + b]); REDUCE }
}
'''
_mod = None


def available() -> bool:
    return cp is not None and torch.cuda.is_available()


def _k(name: str):
    global _mod
    if _mod is None:
        _mod = cp.RawModule(code=_SRC, options=("--std=c++14",))
    return _mod.get_function(name)


def _run(name: str, n_rows: int, batch: int, *args) -> None:
    total = n_rows * batch
    _k(name)(((total + 255) // 256,), (256,), tuple(cp.asarray(a.detach()) if isinstance(a, torch.Tensor) else a for a in args)
             + (cp.int32(n_rows), cp.int32(batch)))


class HopCuda(torch.autograd.Function):
    """z[target, b] = sum over connections (source -> target) of strength * h[source, b]; states are [neurons, batch].
    crow / col: the connections by target (CSR, int64 / int32); tcrow / tcol / tperm: the same by source."""

    @staticmethod
    def forward(ctx, w, h, crow, col, tcrow, tcol, tperm, half: bool):
        n_rows, batch = crow.numel() - 1, h.shape[1]
        assert batch % 32 == 0 and h.is_contiguous()
        hs = h.half() if half else h
        out = torch.empty(n_rows, batch, device=h.device)
        _run("spmm_h" if half else "spmm_f", n_rows, batch, crow, col, w.contiguous(), hs, out)
        ctx.save_for_backward(w, crow, col, tcrow, tcol, tperm)
        ctx.hs, ctx.half, ctx.n_src = hs, half, h.shape[0]
        return out

    @staticmethod
    def backward(ctx, gz):
        w, crow, col, tcrow, tcol, tperm = ctx.saved_tensors
        n_rows, batch = gz.shape
        gz = gz.contiguous()
        gw = gh = None
        scale = 1.0
        if ctx.half:                                        # 16-bit gradients: scaled into the range first
            scale = float(gz.abs().max().clamp_min(1e-30))
            gzs = (gz / scale).half()
        else:
            gzs = gz
        if ctx.needs_input_grad[0]:
            gw = torch.zeros_like(w)
            _run("sddmm_h" if ctx.half else "sddmm_f", n_rows, batch, crow, col, gzs, ctx.hs, gw)
            if ctx.half:
                gw = gw * scale
        if ctx.needs_input_grad[1]:
            gh = torch.empty(ctx.n_src, batch, device=gz.device)
            _run("spmm_h" if ctx.half else "spmm_f", ctx.n_src, batch, tcrow, tcol, w[tperm].contiguous(), gzs, gh)
            if ctx.half:
                gh = gh * scale
        return gw, gh, None, None, None, None, None, None
