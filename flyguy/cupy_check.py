"""Does a hand-written CUDA kernel compile and run here, and does it share memory with PyTorch?"""
import cupy as cp
import torch

k = cp.RawKernel(r'''
extern "C" __global__ void twice(float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = 2.0f * x[i];
}''', "twice")
t = torch.arange(8, dtype=torch.float32, device="cuda")
a = cp.asarray(t)                       # zero-copy view of the torch tensor
k((1,), (8,), (a, cp.int32(8)))
print("cupy", cp.__version__, "| torch sees the kernel's result:", t.tolist())
