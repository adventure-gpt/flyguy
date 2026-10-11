// flyguy native trainer (2026-10-09). The fly-wired actor of flyguy/fly.py, taught by a recorded teacher - everything on
// the GPU written by hand: batch gather from 16-bit data in GPU memory, fused hops, KL loss, fused backward (dead
// warps skipped), Adam with a running average of the weights. States are [neurons, batch], so a warp reads 32
// neighbouring samples of one source neuron. Only connections that can affect a decision are parameters.
//
//   build (desktop):  nvcc -O3 -std=c++17 -o flyguy_train.exe flyguy_train.cu
//   check:            flyguy_train --check NET.bin PARAMS.bin BATCH.bin          (compare with export.py batch)
//   train:            flyguy_train NET.bin PARAMS_IN.bin DATA_DIR OUT_PREFIX [--batch 512] [--lr 3e-4] [--lr-neuron 1e-4]
//                         [--lr-plug 3e-4] [--lr-min 3e-5] [--half-life 20000] [--passes 1] [--ema 0.999] [--log 100]
//                         [--save-sec 120] [--steps 0] [--clip 0] [--temp 1] [--eval-batches 16]
//                         [--revive-every 0] [--revive-batches 4] [--revive-target 0.4] [--revive-first 0]
//                         [--ring 3] [--max-reuse 4] [--holdout 0] [--eval-every 0]
// DATA_DIR holds chunk_*.bin ("FLYD", n int64, obs_dim, n_act, then obs and teacher logits as 16-bit floats, one
// sample after another). Chunks: the newest one never trained on, failing that the one trained on longest ago.
// Before a chunk is learned from, the running average of the weights (what gets installed) is scored on it:
//   eval step N | kl | agree | firing share per layer | silent neurons | chunk unseen / seen before
// --revive-every N: every N steps the neurons that fire for none of --revive-batches batches are brought back with the
// network's output unchanged (see revive()). --temp: temperature of the teacher's distribution in the loss.
// The same kernels as a library for reinforcement learning: -DFLYGUY_CORE (the end of this file, build_core.bat).
// Saves OUT.bin, OUT.ema.bin, OUT.opt.bin (Adam's state): started again from OUT.bin the run goes on exactly where it
// was. A file OUT_PREFIX.STOP ends the run after a save. Formats: flyguy/native/export.py.
#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#endif
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <climits>
#include <filesystem>
#include <map>
#include <memory>
#include <mutex>
#include <atomic>
#include <random>
#include <string>
#include <thread>
#include <vector>

#define CK(call) do { cudaError_t e_ = (call); if (e_ != cudaSuccess) { fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); fflush(stderr); fflush(stdout); std::_Exit(1); } } while (0)   // (no teardown: after an out-of-memory error it hung)
#define IDX(rows) long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x; int t = (int)(idx / B); int b = (int)(idx - (long long)t * B); if (t >= (rows)) return;
#define WSUM(v) v += __shfl_down_sync(0xffffffff, v, 16); v += __shfl_down_sync(0xffffffff, v, 8); v += __shfl_down_sync(0xffffffff, v, 4); v += __shfl_down_sync(0xffffffff, v, 2); v += __shfl_down_sync(0xffffffff, v, 1);

__device__ inline float softplus(float x) { return x > 20.f ? x : log1pf(expf(x)); }
__device__ inline float sigmoidf(float x) { return 1.f / (1.f + expf(-x)); }

// ---- batch from the chunk in GPU memory: x [obs_dim, B], tl [n_act, B]
__global__ void k_gather16(const __half* obs, const __half* lg, const long long* sel, float* x, float* tl, int obs_dim, int n_act, int B) {
    IDX(obs_dim + n_act)
    long long s = sel[b];
    if (t < obs_dim) x[idx] = __half2float(obs[s * obs_dim + t]);
    else tl[(long long)(t - obs_dim) * B + b] = __half2float(lg[s * n_act + (t - obs_dim)]);
}
// the batch the host drew: st [B, obs_dim + n_act] 16-bit -> x [obs_dim, B], tl [n_act, B]
__global__ void k_unpack16(const __half* st, float* x, float* tl, int obs_dim, int n_act, int B) {
    IDX(obs_dim + n_act)
    float v = __half2float(st[(long long)b * (obs_dim + n_act) + t]);
    if (t < obs_dim) x[idx] = v; else tl[(long long)(t - obs_dim) * B + b] = v;
}
__global__ void k_gather32(const float* obs, const float* lg, float* x, float* tl, int obs_dim, int n_act, int B) {
    IDX(obs_dim + n_act)
    if (t < obs_dim) x[idx] = obs[(long long)b * obs_dim + t];
    else tl[(long long)(t - obs_dim) * B + b] = lg[(long long)b * n_act + (t - obs_dim)];
}
// ---- forward
__global__ void k_plug(const int* crow, const int* col, const float* w, const float* x, const float* bias, float* drive, int rows, int B) {
    IDX(rows)
    float acc = bias[t];
    for (int e = crow[t]; e < crow[t + 1]; ++e) acc += w[e] * x[(long long)col[e] * B + b];
    drive[idx] = acc;
}
__global__ void k_act0(const int* n0, const float* gain, const float* theta, const float* drive, float* h, int rows, int B) {
    IDX(rows)
    int n = n0[t];
    h[idx] = fmaxf(gain[n] * drive[idx] - softplus(theta[n]), 0.f);
}
__global__ void k_hop(const int* crow, const int* col, const int* widx, const float* W, const float* hp, const int* nid, const int* drow,
                      const float* drive, const float* gain, const float* theta, float* z, float* h, int rows, int B) {
    IDX(rows)
    float acc = 0.f;
    for (int e = crow[t]; e < crow[t + 1]; ++e) acc += W[widx[e]] * hp[(long long)col[e] * B + b];
    int d = drow[t];
    if (d >= 0) acc += drive[(long long)d * B + b];
    z[idx] = acc;
    int n = nid[t];
    h[idx] = fmaxf(gain[n] * acc - softplus(theta[n]), 0.f);
}
__global__ void k_logits(const float* ow, const float* ob, const float* hK, const int* dn, float* lg, int n_dn, int rows, int B) {
    IDX(rows)
    float acc = ob[t];
    const float* w = ow + (long long)t * n_dn;
    for (int j = 0; j < n_dn; ++j) acc += w[j] * hK[(long long)dn[j] * B + b];
    lg[idx] = acc;
}
// ---- loss: KL(teacher || student), mean over the batch; dl = d loss / d logits
// (it = 1 / temperature of the teacher's distribution: above 1 the target leans harder on the teacher's first choice)
__global__ void k_loss(const float* lg, const float* tl, float* dl, float* stats, int n_act, int B, float it) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B) return;
    float ms = -1e30f, mt = -1e30f; int as = 0, at = 0;
    for (int a = 0; a < n_act; ++a) {
        float s = lg[(long long)a * B + b], t = tl[(long long)a * B + b] * it;
        if (s > ms) { ms = s; as = a; }
        if (t > mt) { mt = t; at = a; }
    }
    float zs = 0.f, zt = 0.f;
    for (int a = 0; a < n_act; ++a) { zs += expf(lg[(long long)a * B + b] - ms); zt += expf(tl[(long long)a * B + b] * it - mt); }
    float ls = ms + logf(zs), lt = mt + logf(zt), kl = 0.f;
    for (int a = 0; a < n_act; ++a) {
        float lps = lg[(long long)a * B + b] - ls, lpt = tl[(long long)a * B + b] * it - lt, pt = expf(lpt);
        kl += pt * (lpt - lps);
        dl[(long long)a * B + b] = (expf(lps) - pt) / (float)B;
    }
    atomicAdd(&stats[0], kl);
    atomicAdd(&stats[1], as == at ? 1.f : 0.f);
}
// ---- backward
__global__ void k_out_bwd_w(const float* dl, const float* hK, const int* dn, float* gow, float* gob, int n_dn, int n_act, int B) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    int a = (int)(idx / (n_dn + 1)), j = (int)(idx - (long long)a * (n_dn + 1));
    if (a >= n_act) return;
    const float* d = dl + (long long)a * B;
    float acc = 0.f;
    if (j == n_dn) { for (int b = 0; b < B; ++b) acc += d[b]; gob[a] += acc; return; }
    const float* h = hK + (long long)dn[j] * B;
    for (int b = 0; b < B; ++b) acc += d[b] * h[b];
    gow[(long long)a * n_dn + j] += acc;
}
__global__ void k_out_bwd_h(const float* dl, const float* ow, const int* dn, float* gh, int n_dn, int n_act, int B) {
    IDX(n_dn)
    float acc = 0.f;
    for (int a = 0; a < n_act; ++a) acc += ow[(long long)a * n_dn + t] * dl[(long long)a * B + b];
    gh[(long long)dn[t] * B + b] = acc;
}
// through the neuron: gz = gh * (h > 0) * gain; gradients of gain and threshold; what flows back into the drive
__global__ void k_hop_bwd(const float* gh, const float* h, const float* z, const int* nid, const int* drow, const float* gain,
                          const float* theta, float* gz, float* ggain, float* gtheta, float* gdrive, int rows, int B) {
    IDX(rows)
    int n = nid[t];
    float g = h[idx] > 0.f ? gh[idx] : 0.f;
    if (__ballot_sync(0xffffffff, g != 0.f) == 0) { if (gz) gz[idx] = 0.f; return; }       // the whole warp is silent
    float gzv = g * gain[n];
    if (gz) gz[idx] = gzv;
    int d = drow ? drow[t] : t;
    if (d >= 0) gdrive[(long long)d * B + b] += gzv;
    float a = g * z[idx], c = -g * sigmoidf(theta[n]);
    WSUM(a) WSUM(c)
    if ((threadIdx.x & 31) == 0) { atomicAdd(&ggain[n], a); atomicAdd(&gtheta[n], c); }
}
// per-connection gradient: gW[widx[e]] += sum_b gz[target, b] * hprev[source, b]
__global__ void k_sddmm(const int* crow, const int* col, const int* widx, const float* gz, const float* hp, float* gW, int rows, int B) {
    IDX(rows)
    float g = gz[idx];
    if (__ballot_sync(0xffffffff, g != 0.f) == 0) return;
    for (int e = crow[t]; e < crow[t + 1]; ++e) {
        float v = g * hp[(long long)col[e] * B + b];
        WSUM(v)
        if ((threadIdx.x & 31) == 0) atomicAdd(&gW[widx ? widx[e] : e], v);
    }
}
__global__ void k_spmmT(const int* tcrow, const int* tcol, const int* twidx, const float* W, const float* gz, float* gh, int rows, int B) {
    IDX(rows)
    float acc = 0.f;
    for (int e = tcrow[t]; e < tcrow[t + 1]; ++e) acc += W[twidx[e]] * gz[(long long)tcol[e] * B + b];
    gh[idx] = acc;
}
__global__ void k_rowsum(const float* a, float* out, int rows, int B) {
    IDX(rows)
    float v = a[idx];
    WSUM(v)
    if ((threadIdx.x & 31) == 0) atomicAdd(&out[t], v);
}
// for the surveys: for how many of the batch's samples each row fires, and the sum of its squared inputs
__global__ void k_fired(const float* h, const float* z, int* cnt, float* sq, int rows, int B) {
    IDX(rows)
    int c = h[idx] > 0.f ? 1 : 0;
    float v = z ? z[idx] * z[idx] : 0.f;
    WSUM(c) WSUM(v)
    if ((threadIdx.x & 31) == 0) { atomicAdd(&cnt[t], c); atomicAdd(&sq[t], v); }
}
__global__ void k_sumsq(const float* g, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float v = i < n ? g[i] * g[i] : 0.f;
    WSUM(v)
    if ((threadIdx.x & 31) == 0) atomicAdd(out, v);
}
__global__ void k_adam(float* p, const float* g, float* m, float* v, float* ema, int n, float lr, float c1, float c2, float gs, float decay) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float gr = g[i] * gs;
    float mi = 0.9f * m[i] + 0.1f * gr, vi = 0.999f * v[i] + 0.001f * gr * gr;
    m[i] = mi; v[i] = vi;
    float pi = p[i] - lr * (mi / c1) / (sqrtf(vi / c2) + 1e-8f);
    p[i] = pi;
    ema[i] = decay * ema[i] + (1.f - decay) * pi;
}

// ---------------------------------------------------------------- host
template <class T> static T* dev(size_t n) { T* p; CK(cudaMalloc(&p, std::max<size_t>(n, 1) * sizeof(T))); CK(cudaMemset(p, 0, std::max<size_t>(n, 1) * sizeof(T))); return p; }
template <class T> static T* up(const std::vector<T>& v) { T* p = dev<T>(v.size()); if (!v.empty()) CK(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice)); return p; }
static std::vector<int> rd(FILE* f, size_t n) { std::vector<int> v(n); if (n && fread(v.data(), 4, n, f) != n) { fprintf(stderr, "short read\n"); exit(1); } return v; }
static std::vector<float> get(const float* d, size_t n) { std::vector<float> v(n); CK(cudaMemcpy(v.data(), d, n * 4, cudaMemcpyDeviceToHost)); return v; }
static void put(float* d, const std::vector<float>& v) { CK(cudaMemcpy(d, v.data(), v.size() * 4, cudaMemcpyHostToDevice)); }
static inline dim3 grid(long long n) { return dim3((unsigned)((n + 255) / 256)); }
#define L(n) grid(n), dim3(256)

struct Hop {
    int rows, E; int *nid, *crow, *col, *widx, *tcrow, *tcol, *twidx, *drow; float *z, *h, *gh;
    std::vector<int> h_nid, h_crow, h_col, h_widx;                  // host copies, for the surgery on silent neurons
};
struct Net {
    int obs_dim, n_act, hops, n_neurons, n_w, n_s, fan, n_dn, E_in;
    std::vector<int> sizes, h_n0, h_dn;
    int *in_crow, *in_col, *n0, *dn;
    std::vector<Hop> hop;
    size_t psize[7], rows_total;
    float *P[7], *G[7], *M[7], *V[7], *EMA[7];                      // W, gain, theta, in_w, in_b, out_w, out_b
    float *x, *tl, *drive, *gdrive, *h0, *gh0, *gz, *lg, *dl, *stats, *estats, *sq;
    int* cnt;
    int B = 0;
};

static void load_net(Net& n, const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
    char magic[4]; fread(magic, 1, 4, f);
    auto hd = rd(f, 8);
    n.obs_dim = hd[0]; n.n_act = hd[1]; n.hops = hd[2]; n.n_neurons = hd[3]; n.n_w = hd[4]; n.n_s = hd[5]; n.fan = hd[6]; n.n_dn = hd[7];
    n.sizes = rd(f, n.hops + 1);
    auto E = rd(f, n.hops);
    n.E_in = n.n_s * n.fan;
    n.in_crow = up(rd(f, n.n_s + 1)); n.in_col = up(rd(f, n.E_in));
    rd(f, n.obs_dim + 1); rd(f, n.E_in); rd(f, n.E_in);                // the plug's transposed form: not needed here
    n.h_n0 = rd(f, n.n_s); n.n0 = up(n.h_n0);
    n.rows_total = n.n_s;
    for (int k = 0; k < n.hops; ++k) {
        Hop h; h.rows = n.sizes[k + 1]; h.E = E[k];
        h.h_nid = rd(f, h.rows); h.h_crow = rd(f, h.rows + 1); h.h_col = rd(f, h.E); h.h_widx = rd(f, h.E);
        h.nid = up(h.h_nid); h.crow = up(h.h_crow); h.col = up(h.h_col); h.widx = up(h.h_widx);
        h.tcrow = up(rd(f, n.sizes[k] + 1)); h.tcol = up(rd(f, h.E)); h.twidx = up(rd(f, h.E)); h.drow = up(rd(f, h.rows));
        n.rows_total += h.rows;
        n.hop.push_back(h);
    }
    n.h_dn = rd(f, n.n_dn); n.dn = up(n.h_dn);
    fclose(f);
    size_t ps[7] = {(size_t)n.n_w, (size_t)n.n_neurons, (size_t)n.n_neurons, (size_t)n.E_in, (size_t)n.n_s, (size_t)n.n_act * n.n_dn, (size_t)n.n_act};
    for (int i = 0; i < 7; ++i) { n.psize[i] = ps[i]; n.P[i] = dev<float>(ps[i]); n.G[i] = dev<float>(ps[i]); n.M[i] = dev<float>(ps[i]); n.V[i] = dev<float>(ps[i]); n.EMA[i] = dev<float>(ps[i]); }
    n.cnt = dev<int>(n.rows_total); n.sq = dev<float>(n.rows_total);
}
// PARAMS.bin; with ema_path the running average is read from there (a restart keeps it), else it starts as a copy
static long long load_params(Net& n, const char* path, const char* ema_path = nullptr) {
    long long step = 0;
    for (int which = 0; which < 2; ++which) {
        FILE* f = fopen(which ? (ema_path ? ema_path : path) : path, "rb");
        if (!f && which) f = fopen(path, "rb");
        if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
        char magic[4]; long long st; fread(magic, 1, 4, f); fread(&st, 8, 1, f);
        if (!which) step = st;
        for (int i = 0; i < 7; ++i) {
            std::vector<float> v(n.psize[i]);
            if (fread(v.data(), 4, v.size(), f) != v.size()) { fprintf(stderr, "short params file\n"); exit(1); }
            CK(cudaMemcpy(which ? n.EMA[i] : n.P[i], v.data(), v.size() * 4, cudaMemcpyHostToDevice));
        }
        fclose(f);
    }
    return step;
}
static void finish_file(FILE* f, const std::string& tmp, const std::string& path) {
    fclose(f);
    std::error_code ec; std::filesystem::rename(tmp, path, ec);
    if (ec) { std::filesystem::remove(path, ec); std::filesystem::rename(tmp, path, ec); }
}
static void save_params(const Net& n, const std::string& path, long long step, bool ema) {
    std::string tmp = path + ".tmp";
    FILE* f = fopen(tmp.c_str(), "wb");
    fwrite("FLYP", 1, 4, f); fwrite(&step, 8, 1, f);
    for (int i = 0; i < 7; ++i) { auto v = get(ema ? n.EMA[i] : n.P[i], n.psize[i]); fwrite(v.data(), 4, v.size(), f); }
    finish_file(f, tmp, path);
}
// Adam's two running averages and its step count: a restart at the same step goes on exactly where the run stopped
static void save_opt(const Net& n, const std::string& path, long long step, long long adam_t) {
    std::string tmp = path + ".tmp";
    FILE* f = fopen(tmp.c_str(), "wb");
    fwrite("FLYO", 1, 4, f); fwrite(&step, 8, 1, f); fwrite(&adam_t, 8, 1, f);
    for (int i = 0; i < 7; ++i) { auto m = get(n.M[i], n.psize[i]), v = get(n.V[i], n.psize[i]); fwrite(m.data(), 4, m.size(), f); fwrite(v.data(), 4, v.size(), f); }
    finish_file(f, tmp, path);
}
static bool load_opt(Net& n, const std::string& path, long long step, long long& adam_t) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return false;
    char magic[4]; long long st = -1, at = 0;
    bool ok = fread(magic, 1, 4, f) == 4 && !memcmp(magic, "FLYO", 4) && fread(&st, 8, 1, f) == 1 && fread(&at, 8, 1, f) == 1 && st == step;
    std::vector<std::vector<float>> m(7), v(7);
    for (int i = 0; ok && i < 7; ++i) { m[i].resize(n.psize[i]); v[i].resize(n.psize[i]); ok = fread(m[i].data(), 4, m[i].size(), f) == m[i].size() && fread(v[i].data(), 4, v[i].size(), f) == v[i].size(); }
    fclose(f);
    if (!ok) return false;
    for (int i = 0; i < 7; ++i) { put(n.M[i], m[i]); put(n.V[i], v[i]); }
    adam_t = at;
    return true;
}
static void alloc_batch(Net& n, int B) {
    n.B = B;
    n.x = dev<float>((size_t)n.obs_dim * B); n.tl = dev<float>((size_t)n.n_act * B);
    n.drive = dev<float>((size_t)n.n_s * B); n.gdrive = dev<float>((size_t)n.n_s * B);
    n.h0 = dev<float>((size_t)n.n_s * B); n.gh0 = dev<float>((size_t)n.n_s * B);
    size_t mx = 0;
    for (auto& h : n.hop) { h.z = dev<float>((size_t)h.rows * B); h.h = dev<float>((size_t)h.rows * B); h.gh = dev<float>((size_t)h.rows * B); mx = std::max(mx, (size_t)h.rows * B); }
    n.gz = dev<float>(mx);
    n.lg = dev<float>((size_t)n.n_act * B); n.dl = dev<float>((size_t)n.n_act * B); n.stats = dev<float>(4); n.estats = dev<float>(4);
}
// the network on the batch in n.x / n.tl with the weights P (the trained ones or their running average); the loss
// sums go to `stats` (whoever reads them clears them); it = 1 / temperature of the teacher's distribution
static void forward_logits(Net& n, float* const* P) {
    const int B = n.B, K = n.hops;
    const float *W = P[0], *gain = P[1], *theta = P[2];
    k_plug<<<L((long long)n.n_s * B)>>>(n.in_crow, n.in_col, P[3], n.x, P[4], n.drive, n.n_s, B);
    k_act0<<<L((long long)n.n_s * B)>>>(n.n0, gain, theta, n.drive, n.h0, n.n_s, B);
    const float* hp = n.h0;
    for (int k = 0; k < K; ++k) {
        Hop& h = n.hop[k];
        k_hop<<<L((long long)h.rows * B)>>>(h.crow, h.col, h.widx, W, hp, h.nid, h.drow, n.drive, gain, theta, h.z, h.h, h.rows, B);
        hp = h.h;
    }
    k_logits<<<L((long long)n.n_act * B)>>>(P[5], P[6], n.hop[K - 1].h, n.dn, n.lg, n.n_dn, n.n_act, B);
}
static void forward(Net& n, float* const* P, float* stats, float it) {
    forward_logits(n, P);
    k_loss<<<grid(n.B), dim3(256)>>>(n.lg, n.tl, n.dl, stats, n.n_act, n.B, it);
}
// from the gradient of the logits (n.dl) of the batch that forward_logits(n, n.P) just went through, back to every
// weight: ADDED to n.G (whoever starts a step clears it)
static void backward(Net& n) {
    const int B = n.B, K = n.hops;
    float *W = n.P[0], *gain = n.P[1], *theta = n.P[2];
    CK(cudaMemsetAsync(n.gdrive, 0, (size_t)n.n_s * B * 4));
    Hop& top = n.hop[K - 1];
    k_out_bwd_w<<<L((long long)n.n_act * (n.n_dn + 1))>>>(n.dl, top.h, n.dn, n.G[5], n.G[6], n.n_dn, n.n_act, B);
    CK(cudaMemsetAsync(top.gh, 0, (size_t)top.rows * B * 4));
    k_out_bwd_h<<<L((long long)n.n_dn * B)>>>(n.dl, n.P[5], n.dn, top.gh, n.n_dn, n.n_act, B);
    for (int k = K - 1; k >= 0; --k) {
        Hop& h = n.hop[k];
        const float* below = k ? n.hop[k - 1].h : n.h0;
        float* ghb = k ? n.hop[k - 1].gh : n.gh0;
        int rows_below = k ? n.hop[k - 1].rows : n.n_s;
        k_hop_bwd<<<L((long long)h.rows * B)>>>(h.gh, h.h, h.z, h.nid, h.drow, gain, theta, n.gz, n.G[1], n.G[2], n.gdrive, h.rows, B);
        k_sddmm<<<L((long long)h.rows * B)>>>(h.crow, h.col, h.widx, n.gz, below, n.G[0], h.rows, B);
        k_spmmT<<<L((long long)rows_below * B)>>>(h.tcrow, h.tcol, h.twidx, W, n.gz, ghb, rows_below, B);
    }
    k_hop_bwd<<<L((long long)n.n_s * B)>>>(n.gh0, n.h0, n.drive, n.n0, nullptr, gain, theta, nullptr, n.G[1], n.G[2], n.gdrive, n.n_s, B);
    k_rowsum<<<L((long long)n.n_s * B)>>>(n.gdrive, n.G[4], n.n_s, B);
    k_sddmm<<<L((long long)n.n_s * B)>>>(n.in_crow, n.in_col, nullptr, n.gdrive, n.x, n.G[3], n.n_s, B);
}
// one forward + backward on the batch in n.x / n.tl; gradients land in n.G
static void fwd_bwd(Net& n, float it = 1.f) {
    for (int i = 0; i < 7; ++i) CK(cudaMemsetAsync(n.G[i], 0, n.psize[i] * 4));
    forward(n, n.P, n.stats, it);
    backward(n);
}
static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

// ---- data. Chunk files are read whole into host memory by a loader thread and every batch is drawn from the newest
// `ring` chunks at once: the frames of one chunk are consecutive moments of the same cars, near copies of each other,
// and one chunk at a time taught the same moments over and over (10-09: the loss fell 0.06 inside every chunk and
// jumped back at the next). The GPU keeps its memory for the network.
struct Buf { long long n = 0; std::vector<uint16_t> obs, lg; std::string name; std::filesystem::file_time_type mtime; };
typedef std::shared_ptr<Buf> BufP;
static BufP read_chunk(const std::string& path, const Net& net) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return nullptr;
    char magic[4]; long long n; int od, na;
    if (fread(magic, 1, 4, f) != 4 || memcmp(magic, "FLYD", 4) || fread(&n, 8, 1, f) != 1 || fread(&od, 4, 1, f) != 1 || fread(&na, 4, 1, f) != 1 || od != net.obs_dim || na != net.n_act || n <= 0) { fclose(f); return nullptr; }
    auto b = std::make_shared<Buf>();
    b->obs.resize((size_t)n * od); b->lg.resize((size_t)n * na);
    bool ok = fread(b->obs.data(), 2, b->obs.size(), f) == b->obs.size() && fread(b->lg.data(), 2, b->lg.size(), f) == b->lg.size();
    fclose(f);
    if (!ok) return nullptr;
    b->n = n; b->name = std::filesystem::path(path).filename().string();
    return b;
}
struct Loader {
    std::mutex mu; std::deque<BufP> pending; std::atomic<bool> quit{false};
    std::map<std::string, std::filesystem::file_time_type> known;      // (the loader's own)
    std::string dir; const Net* net = nullptr; size_t keep = 3;
    // reads every chunk file that is new or was rewritten since it was read, oldest first - of several only the
    // newest `keep`; `wait`: not more than one chunk ahead of the trainer
    int scan(bool wait) {
        std::vector<std::pair<std::filesystem::file_time_type, std::string>> files;
        std::error_code ec;
        for (auto& e : std::filesystem::directory_iterator(dir, ec)) {
            auto p = e.path().filename().string();
            std::error_code e2; auto mt = std::filesystem::last_write_time(e.path(), e2);
            if (e2 || p.rfind("chunk_", 0) != 0 || e.path().extension() != ".bin") continue;
            auto k = known.find(p);
            if (k == known.end() || k->second != mt) files.push_back({mt, e.path().string()});
        }
        std::sort(files.begin(), files.end());
        while (files.size() > keep) { known[std::filesystem::path(files.front().second).filename().string()] = files.front().first; files.erase(files.begin()); }
        int got = 0;
        for (auto& fm : files) {
            while (wait && !quit) { { std::lock_guard<std::mutex> g(mu); if (pending.empty()) break; } std::this_thread::sleep_for(std::chrono::milliseconds(200)); }
            if (quit) break;
            BufP b = read_chunk(fm.second, *net);
            if (!b) continue;                                          // (being replaced right now: the next round has it)
            b->mtime = fm.first; known[b->name] = fm.first;
            { std::lock_guard<std::mutex> g(mu); pending.push_back(b); }
            ++got;
        }
        return got;
    }
};
struct Stage { std::vector<uint16_t> buf; __half* d = nullptr; };
// B samples drawn from all the chunks in `src` into n.x / n.tl
static void draw_batch(Net& n, const std::vector<BufP>& src, std::mt19937_64& r, Stage& st) {
    const int B = n.B, od = n.obs_dim, na = n.n_act, w = od + na;
    long long total = 0; for (auto& b : src) total += b->n;
    std::uniform_int_distribution<long long> u(0, total - 1);
    for (int i = 0; i < B; ++i) {
        long long s = u(r); size_t c = 0;
        while (s >= src[c]->n) { s -= src[c]->n; ++c; }
        memcpy(&st.buf[(size_t)i * w], &src[c]->obs[(size_t)s * od], (size_t)od * 2);
        memcpy(&st.buf[(size_t)i * w + od], &src[c]->lg[(size_t)s * na], (size_t)na * 2);
    }
    CK(cudaMemcpy(st.d, st.buf.data(), (size_t)B * w * 2, cudaMemcpyHostToDevice));
    k_unpack16<<<L((long long)w * B)>>>(st.d, n.x, n.tl, od, na, B);
}

// ---- a look at the network on `batches` batches from `src` (the same samples for the same seed), no learning: loss
// and agreement against the teacher at temperature 1, how often every row of every layer fires, its input size
struct Survey { std::vector<int> cnt; std::vector<float> sq; double kl = 0, agree = 0, samples = 0; };
static Survey survey(Net& n, float* const* P, const std::vector<BufP>& src, Stage& st, int batches, uint64_t seed) {
    const int B = n.B;
    std::mt19937_64 r(seed);
    CK(cudaMemset(n.cnt, 0, n.rows_total * 4)); CK(cudaMemset(n.sq, 0, n.rows_total * 4)); CK(cudaMemset(n.estats, 0, 16));
    for (int i = 0; i < batches; ++i) {
        draw_batch(n, src, r, st);
        forward(n, P, n.estats, 1.f);
        k_fired<<<L((long long)n.n_s * B)>>>(n.h0, nullptr, n.cnt, n.sq, n.n_s, B);
        size_t off = n.n_s;
        for (auto& h : n.hop) { k_fired<<<L((long long)h.rows * B)>>>(h.h, h.z, n.cnt + off, n.sq + off, h.rows, B); off += h.rows; }
    }
    CK(cudaDeviceSynchronize());
    Survey s; s.cnt.resize(n.rows_total); s.sq = get(n.sq, n.rows_total);
    CK(cudaMemcpy(s.cnt.data(), n.cnt, n.rows_total * 4, cudaMemcpyDeviceToHost));
    float stt[4]; CK(cudaMemcpy(stt, n.estats, 16, cudaMemcpyDeviceToHost));
    s.samples = (double)B * batches; s.kl = stt[0] / s.samples; s.agree = stt[1] / s.samples;
    return s;
}
// per neuron (it can sit in several hops): how many times it fired; play = in a hop past the sensory layer
static void per_neuron(const Net& n, const Survey& s, std::vector<long long>& fired, std::vector<char>& play, std::vector<char>& sens) {
    fired.assign(n.n_neurons, 0); play.assign(n.n_neurons, 0); sens.assign(n.n_neurons, 0);
    for (int g : n.h_n0) sens[g] = 1;
    size_t off = n.n_s;
    for (auto& h : n.hop) { for (int t = 0; t < h.rows; ++t) { int g = h.h_nid[t]; play[g] = 1; fired[g] += s.cnt[off + t]; } off += h.rows; }
}
static void firing_line(const Net& n, const Survey& s, char* buf, size_t cap, int& silent, int& in_play) {
    std::vector<long long> fired; std::vector<char> play, sens; per_neuron(n, s, fired, play, sens);
    silent = in_play = 0;
    for (int g = 0; g < n.n_neurons; ++g) if (play[g] && !sens[g]) { ++in_play; if (!fired[g]) ++silent; }
    size_t off = 0, at = 0;
    for (int k = 0; k <= n.hops; ++k) {
        int rows = k ? n.hop[k - 1].rows : n.n_s; double c = 0;
        for (int t = 0; t < rows; ++t) c += s.cnt[off + t];
        at += snprintf(buf + at, cap - at, " %.3f", c / (rows * s.samples)); off += rows;
    }
}
static float median_positive(const std::vector<float>& v) {
    std::vector<float> p; p.reserve(v.size());
    for (float x : v) if (x > 0.f) p.push_back(x);
    if (p.empty()) return 0.f;
    std::nth_element(p.begin(), p.begin() + p.size() / 2, p.end());
    return p[p.size() / 2];
}
// ---- silent neurons back to work, the network's output unchanged. A rectifying neuron whose input is never positive
// passes no gradient: it and its connections never learn. For every neuron past the sensory layer that fired for none
// of the surveyed samples in any hop: incoming strengths keep their relative sizes, get a random sign and one common
// factor that brings the neuron's input to `target` (rms); gain and threshold return to their starting values; its
// strengths onto live neurons and onto the actions become 0 - it sent nothing before and sends nothing now, but it
// fires, so those can learn. The running average takes the new values; Adam restarts for them at a typical scale.
static void revive(Net& n, const std::vector<BufP>& src, Stage& st, int batches, float target, uint64_t seed, long long step) {
    Survey s0 = survey(n, n.P, src, st, batches, seed);
    const int N = n.n_neurons, K = n.hops;
    std::vector<long long> fired; std::vector<char> play, sens, dead(N, 0);
    per_neuron(n, s0, fired, play, sens);
    int n_dead = 0, n_play = 0;
    for (int g = 0; g < N; ++g) if (play[g] && !sens[g]) { ++n_play; if (!fired[g]) { dead[g] = 1; ++n_dead; } }
    if (!n_dead) { printf("revive step %lld | no silent neuron among %d over %.0f samples\n", step, n_play, s0.samples); fflush(stdout); return; }
    auto W = get(n.P[0], n.psize[0]), gain = get(n.P[1], N), theta = get(n.P[2], N), ow = get(n.P[5], n.psize[5]);
    std::vector<char> mark(n.n_w, 0);                    // 1: into a silent neuron (new sign), 2: from a silent one into a live one (zero)
    std::mt19937_64 r(seed * 7919ULL + 17);
    long long n_in = 0, n_out = 0;
    for (int k = 0; k < K; ++k) {
        Hop& h = n.hop[k];
        const std::vector<int>& below = k ? n.hop[k - 1].h_nid : n.h_n0;
        for (int t = 0; t < h.rows; ++t) {
            bool dpost = dead[h.h_nid[t]];
            for (int e = h.h_crow[t]; e < h.h_crow[t + 1]; ++e) {
                int wi = h.h_widx[e];
                if (mark[wi]) continue;
                if (dpost) { W[wi] = std::max(fabsf(W[wi]), 1e-4f) * ((r() & 1) ? 1.f : -1.f); mark[wi] = 1; ++n_in; }
                else if (dead[below[h.h_col[e]]]) { W[wi] = 0.f; mark[wi] = 2; ++n_out; }
            }
        }
    }
    for (int g = 0; g < N; ++g) if (dead[g]) { gain[g] = 1.f; theta[g] = -4.f; }
    Hop& top = n.hop[K - 1];
    std::vector<char> dcol(n.n_dn, 0); int n_dn_dead = 0;
    for (int j = 0; j < n.n_dn; ++j) if (dead[top.h_nid[n.h_dn[j]]]) { dcol[j] = 1; ++n_dn_dead; for (int a = 0; a < n.n_act; ++a) ow[(size_t)a * n.n_dn + j] = 0.f; }
    put(n.P[0], W); put(n.P[1], gain); put(n.P[2], theta); put(n.P[5], ow);
    for (int pass = 0; pass < 3; ++pass) {               // hop after hop the revived neurons' inputs come to the target size
        Survey s = survey(n, n.P, src, st, batches, seed);
        std::vector<float> rms(N, 0.f);
        size_t off = n.n_s;
        for (auto& h : n.hop) { for (int t = 0; t < h.rows; ++t) { int g = h.h_nid[t]; rms[g] = std::max(rms[g], (float)sqrt(s.sq[off + t] / s.samples)); } off += h.rows; }
        std::vector<char> done(n.n_w, 0);
        for (auto& h : n.hop) for (int t = 0; t < h.rows; ++t) {
            int g = h.h_nid[t];
            if (!dead[g] || rms[g] <= 0.f) continue;
            float sc = std::min(200.f, std::max(0.02f, target / rms[g]));
            for (int e = h.h_crow[t]; e < h.h_crow[t + 1]; ++e) { int wi = h.h_widx[e]; if (!done[wi]) { W[wi] *= sc; done[wi] = 1; } }
        }
        put(n.P[0], W);
    }
    Survey s1 = survey(n, n.P, src, st, batches, seed);
    per_neuron(n, s1, fired, play, sens);
    int still = 0; for (int g = 0; g < N; ++g) if (dead[g] && !fired[g]) ++still;
    {   // the running average and Adam follow
        auto e = get(n.EMA[0], n.psize[0]), m = get(n.M[0], n.psize[0]), v = get(n.V[0], n.psize[0]);
        float fill = median_positive(v);
        for (int i = 0; i < n.n_w; ++i) if (mark[i]) { e[i] = W[i]; m[i] = 0.f; v[i] = fill; }
        put(n.EMA[0], e); put(n.M[0], m); put(n.V[0], v);
        for (int p = 1; p <= 2; ++p) {
            auto ee = get(n.EMA[p], N), mm = get(n.M[p], N), vv = get(n.V[p], N);
            float f2 = median_positive(vv);
            for (int g = 0; g < N; ++g) if (dead[g]) { ee[g] = p == 1 ? 1.f : -4.f; mm[g] = 0.f; vv[g] = f2; }
            put(n.EMA[p], ee); put(n.M[p], mm); put(n.V[p], vv);
        }
        auto eo = get(n.EMA[5], n.psize[5]), mo = get(n.M[5], n.psize[5]);
        for (int j = 0; j < n.n_dn; ++j) if (dcol[j]) for (int a = 0; a < n.n_act; ++a) { eo[(size_t)a * n.n_dn + j] = 0.f; mo[(size_t)a * n.n_dn + j] = 0.f; }
        put(n.EMA[5], eo); put(n.M[5], mo);
    }
    printf("revive step %lld | %d of %d neurons past the sensory layer silent over %.0f samples | new sign for %lld incoming strengths, zero for %lld outgoing and for %d descending neurons' action weights | %d still silent | kl %.5f -> %.5f (must be equal)\n",
           step, n_dead, n_play, s0.samples, n_in, n_out, n_dn_dead, still, s0.kl, s1.kl);
    fflush(stdout);
}

#ifdef FLYGUY_CORE
// ---------------------------------------------------------------- the same network as a library: flyguy_core.dll
// For reinforcement learning inside the Python trainer (flyguy/fly_native.py). PyTorch and this library live in one
// process and one GPU context: observations come in and logits go out as GPU memory PyTorch owns, PyTorch computes
// its loss and hands back the gradient of the logits, and the hops, their backward pass and Adam run here - the same
// kernels as the teaching trainer above. A batch is a multiple of 32 and at most the size given at creation.
//   build (desktop):  flyguy\native\build_core.bat
__global__ void k_rows_in(const float* in, float* x, int dim, int B) { IDX(dim) x[idx] = in[(long long)b * dim + t]; }
__global__ void k_rows_out(const float* x, float* out, int dim, int B) { IDX(dim) out[(long long)b * dim + t] = x[idx]; }
struct Core { Net n; int maxB = 0; long long adam_t = 0; float* nrm = nullptr; };
#ifdef _WIN32
#define API extern "C" __declspec(dllexport)
#else
#define API extern "C" __attribute__((visibility("default")))
#endif

API void* fc_create(const char* net_path, int max_batch) {
    Core* c = new Core;
    load_net(c->n, net_path);
    alloc_batch(c->n, max_batch);
    c->maxB = max_batch; c->nrm = dev<float>(1);
    return c;
}
// out[0..6]: the lengths of the seven weight arrays (strengths in play, gain, threshold, plug in, its bias, plug out, its bias)
API void fc_sizes(void* h, long long* out) {
    Core* c = (Core*)h;
    for (int i = 0; i < 7; ++i) out[i] = (long long)c->n.psize[i];
    out[7] = c->n.obs_dim; out[8] = c->n.n_act; out[9] = c->maxB;
}
// which: 0 the weights being trained, 1 their running average, 2 / 3 Adam's two averages, 4 the gradient.
// ptrs: seven GPU arrays of PyTorch's (null = skip); to_core 1 = read from them, 0 = write to them
API void fc_copy(void* h, int which, int to_core, float** ptrs) {
    Net& n = ((Core*)h)->n;
    float** A = which == 0 ? n.P : which == 1 ? n.EMA : which == 2 ? n.M : which == 3 ? n.V : n.G;
    for (int i = 0; i < 7; ++i)
        if (ptrs[i]) CK(cudaMemcpy(to_core ? A[i] : ptrs[i], to_core ? ptrs[i] : A[i], n.psize[i] * 4, cudaMemcpyDeviceToDevice));
}
// obs [B, obs_dim] -> logits [B, n_act]; ema 1 = with the running average (playing), 0 = with the trained weights
// (the states stay in place for fc_backward)
API void fc_forward(void* h, const float* obs, int B, float* logits, int ema) {
    Net& n = ((Core*)h)->n;
    n.B = B;
    k_rows_in<<<L((long long)n.obs_dim * B)>>>(obs, n.x, n.obs_dim, B);
    forward_logits(n, ema ? n.EMA : n.P);
    k_rows_out<<<L((long long)n.n_act * B)>>>(n.lg, logits, n.n_act, B);
}
API void fc_zero_grad(void* h) {
    Net& n = ((Core*)h)->n;
    for (int i = 0; i < 7; ++i) CK(cudaMemsetAsync(n.G[i], 0, n.psize[i] * 4));
}
// dlogits [B, n_act]: the gradient of the loss with respect to the logits of the batch fc_forward(.., ema 0) just did
API void fc_backward(void* h, const float* dlogits, int B) {
    Net& n = ((Core*)h)->n;
    n.B = B;
    k_rows_in<<<L((long long)n.n_act * B)>>>(dlogits, n.dl, n.n_act, B);
    backward(n);
}
API float fc_grad_norm(void* h) {
    Core* c = (Core*)h; Net& n = c->n;
    CK(cudaMemset(c->nrm, 0, 4));
    for (int i = 0; i < 7; ++i) k_sumsq<<<L((long long)((n.psize[i] + 31) / 32 * 32))>>>(n.G[i], c->nrm, (int)n.psize[i]);
    float ss; CK(cudaMemcpy(&ss, c->nrm, 4, cudaMemcpyDeviceToHost));
    return sqrtf(ss);
}
// one Adam step from the gradient gathered since fc_zero_grad; the gradient is scaled down to max_norm first if it
// is longer (0 = no limit). Returns the gradient's length before that.
API float fc_step(void* h, float lr, float lr_neuron, float lr_plug, float max_norm, float ema) {
    Core* c = (Core*)h; Net& n = c->n;
    float g = max_norm > 0 ? fc_grad_norm(h) : 0.f, gs = (max_norm > 0 && g > max_norm) ? max_norm / g : 1.f;
    ++c->adam_t;
    float c1 = 1.f - (float)pow(0.9, (double)std::min<long long>(c->adam_t, 100000)), c2 = 1.f - (float)pow(0.999, (double)std::min<long long>(c->adam_t, 100000));
    for (int i = 0; i < 7; ++i)
        k_adam<<<L((long long)n.psize[i])>>>(n.P[i], n.G[i], n.M[i], n.V[i], n.EMA[i], (int)n.psize[i], i == 0 ? lr : (i <= 2 ? lr_neuron : lr_plug), c1, c2, gs, ema);
    return g;
}
API long long fc_adam_t(void* h, long long set) { Core* c = (Core*)h; if (set >= 0) c->adam_t = set; return c->adam_t; }
API void fc_sync(void*) { CK(cudaDeviceSynchronize()); }
#else
int main(int argc, char** argv) {
    if (argc >= 5 && !strcmp(argv[1], "--check")) {
        Net n; load_net(n, argv[2]); load_params(n, argv[3]);
        FILE* f = fopen(argv[4], "rb"); char magic[4]; int B; fread(magic, 1, 4, f); fread(&B, 4, 1, f);
        std::vector<float> x((size_t)B * n.obs_dim), t((size_t)B * n.n_act);
        fread(x.data(), 4, x.size(), f); fread(t.data(), 4, t.size(), f); fclose(f);
        alloc_batch(n, B);
        float *dx = up(x), *dt = up(t);
        k_gather32<<<L((long long)(n.obs_dim + n.n_act) * B)>>>(dx, dt, n.x, n.tl, n.obs_dim, n.n_act, B);
        fwd_bwd(n);
        CK(cudaDeviceSynchronize());
        float st[4]; CK(cudaMemcpy(st, n.stats, 16, cudaMemcpyDeviceToHost));
        std::vector<float> lg((size_t)n.n_act * B); CK(cudaMemcpy(lg.data(), n.lg, lg.size() * 4, cudaMemcpyDeviceToHost));
        printf("native: loss %.6f agree %.4f\n   logits[0][:5]", st[0] / B, st[1] / B);
        for (int a = 0; a < 5; ++a) printf(" %.6f", lg[(size_t)a * B]);
        printf("\n");
        const char* nm[7] = {"W", "gain", "theta", "in_w", "in_b", "out_w", "out_b"};
        for (int i = 0; i < 7; ++i) {
            std::vector<float> g(n.psize[i]); CK(cudaMemcpy(g.data(), n.G[i], g.size() * 4, cudaMemcpyDeviceToHost));
            double ss = 0, s = 0; for (float v : g) { ss += (double)v * v; s += v; }
            printf("   grad %-6s norm %.6e sum %+.6e\n", nm[i], sqrt(ss), s);
        }
        return 0;
    }
    if (argc < 5) { fprintf(stderr, "usage: flyguy_train NET PARAMS DATA_DIR OUT_PREFIX [options] | --check NET PARAMS BATCH\n"); return 2; }
    int B = 512, log_every = 100, eval_batches = 16, revive_batches = 4, revive_first = 0, ring_k = 3, holdout = 0;
    long long max_steps = 0, revive_every = 0, eval_every = 0; double save_sec = 120, half_life = 20000;
    float lr = 3e-4f, lr_neuron = 1e-4f, lr_plug = 3e-4f, lr_min = 3e-5f, ema = 0.999f, clip = 0.f, temp = 1.f, revive_target = 0.4f; double max_reuse = -1;
    for (int i = 5; i + 1 < argc; i += 2) {
        std::string a = argv[i]; double v = atof(argv[i + 1]);
        if (a == "--batch") B = (int)v; else if (a == "--lr") lr = (float)v; else if (a == "--lr-neuron") lr_neuron = (float)v;
        else if (a == "--lr-plug") lr_plug = (float)v; else if (a == "--lr-min") lr_min = (float)v; else if (a == "--half-life") half_life = v;
        else if (a == "--ema") ema = (float)v; else if (a == "--log") log_every = (int)v;
        else if (a == "--save-sec") save_sec = v; else if (a == "--steps") max_steps = (long long)v; else if (a == "--clip") clip = (float)v;
        else if (a == "--temp") temp = (float)v; else if (a == "--eval-batches") eval_batches = (int)v; else if (a == "--eval-every") eval_every = (long long)v;
        else if (a == "--revive-every") revive_every = (long long)v; else if (a == "--revive-batches") revive_batches = (int)v;
        else if (a == "--revive-target") revive_target = (float)v; else if (a == "--revive-first") revive_first = (int)v;
        else if (a == "--max-reuse") max_reuse = v; else if (a == "--ring") ring_k = std::max(1, (int)v); else if (a == "--holdout") holdout = (int)v;
        else if (a == "--passes") { }                                  // (of the one-chunk-at-a-time trainer: ignored)
        else { fprintf(stderr, "unknown option %s\n", a.c_str()); return 2; }
    }
    if (B % 32) { fprintf(stderr, "--batch must be a multiple of 32\n"); return 2; }
#ifdef _WIN32
    SetPriorityClass(GetCurrentProcess(), ABOVE_NORMAL_PRIORITY_CLASS);   // the thread that feeds the GPU must not wait behind the recording's simulation threads
#endif
    Net n; load_net(n, argv[1]);
    std::string data = argv[3], out = argv[4];
    // a restart from this run's own save keeps the running average and the optimiser state
    bool own = std::filesystem::path(argv[2]).lexically_normal() == std::filesystem::path(out + ".bin").lexically_normal();
    std::string ema_file = out + ".ema.bin";
    long long step = load_params(n, argv[2], own && std::filesystem::exists(ema_file) ? ema_file.c_str() : nullptr);
    long long adam_t = 0, ran = 0;
    bool opt = own && load_opt(n, out + ".opt.bin", step, adam_t);
    alloc_batch(n, B);
    Stage st; st.buf.resize((size_t)B * (n.obs_dim + n.n_act)); CK(cudaMalloc(&st.d, st.buf.size() * 2));
    float* nrm = dev<float>(1);
    std::mt19937_64 rng(12345 + step);
    const float it = 1.f / temp;
    printf("flyguy native trainer: %d neurons, %d strengths in play, %d hops, batch %d, lr %g (neurons %g, plugs %g), teacher temperature %g, ring of %d chunks, start step %lld%s\n",
           n.n_neurons, n.n_w, n.hops, B, lr, lr_neuron, lr_plug, temp, ring_k, step, opt ? ", optimiser state restored" : ""); fflush(stdout);

    Loader ld; ld.dir = data; ld.net = &n; ld.keep = (size_t)ring_k + (holdout ? 1 : 0);
    while (!ld.scan(false)) { printf("waiting for data in %s\n", data.c_str()); fflush(stdout); std::this_thread::sleep_for(std::chrono::seconds(20)); }
    std::vector<BufP> ring; BufP held;                    // held (--holdout 1): the newest chunk at the start stays out of the training
    auto eval_on = [&](const BufP& b, uint64_t seed, const char* what) {   // what the bot that would be installed (the running average) does
        if (eval_batches <= 0) return;
        std::vector<BufP> one{b};
        Survey ev = survey(n, n.EMA, one, st, eval_batches, seed), raw = survey(n, n.P, one, st, eval_batches, seed);   // (raw: the weights being trained)
        char fb[128]; int silent, in_play; firing_line(n, ev, fb, sizeof fb, silent, in_play);
        printf("eval step %lld | kl %.4f agree %.4f | raw kl %.4f agree %.4f | firing%s | silent %d of %d | %s %s\n", step, ev.kl, ev.agree, raw.kl, raw.agree, fb, silent, in_play, b->name.c_str(), what);
        fflush(stdout);
    };
    auto into_ring = [&](const BufP& b) {
        ring.erase(std::remove_if(ring.begin(), ring.end(), [&](const BufP& o) { return o->name == b->name; }), ring.end());
        ring.push_back(b);
        std::sort(ring.begin(), ring.end(), [](const BufP& x, const BufP& y) { return x->mtime < y->mtime; });
        while (ring.size() > (size_t)ring_k) ring.erase(ring.begin());
    };
    {
        std::lock_guard<std::mutex> g(ld.mu);
        if (holdout && ld.pending.size() > 1) { held = ld.pending.back(); ld.pending.pop_back(); }
        for (auto& b : ld.pending) into_ring(b);
        ld.pending.clear();
    }
    if (held) eval_on(held, 4242, "held out"); else eval_on(ring.back(), 4242, "at the start");
    std::thread loader([&]() { while (!ld.quit) { ld.scan(true); for (int i = 0; i < 20 && !ld.quit; ++i) std::this_thread::sleep_for(std::chrono::milliseconds(100)); } });
    long long last_revive = revive_first ? step - revive_every : step;
    // no more than max_reuse draws per recorded sample: if the recording stops, the trainer waits instead of learning
    // the same chunks by heart (default 4 for an open-ended run, off for --steps)
    if (max_reuse < 0) max_reuse = max_steps ? 0 : 4;
    auto ring_total = [&]() { double s = 0; for (auto& b : ring) s += (double)b->n; return s; };
    double allow = max_reuse * ring_total(); bool waiting = false;
    double t_log = now(), t_save = now(), t_new = now(); long long since = 0;
    auto save_all = [&]() {
        CK(cudaDeviceSynchronize());
        save_params(n, out + ".bin", step, false); save_params(n, out + ".ema.bin", step, true); save_opt(n, out + ".opt.bin", step, adam_t);
        printf("saved %s.bin, .ema.bin and .opt.bin at step %lld\n", out.c_str(), step); fflush(stdout); t_save = now();
    };
    int rc = 0;
    while (!max_steps || ran < max_steps) {
        for (;;) {      // a new chunk: scored before anything is learned from it, then into the ring (the oldest leaves)
            BufP nb;
            { std::lock_guard<std::mutex> g(ld.mu); if (!ld.pending.empty()) { nb = ld.pending.front(); ld.pending.pop_front(); } }
            if (!nb) break;
            eval_on(nb, 1000003ULL * (uint64_t)step + 7, "unseen");
            into_ring(nb); t_new = now();
            allow = std::min(allow + max_reuse * (double)nb->n, max_reuse * ring_total());
        }
        if (max_reuse > 0 && allow < B) {
            if (!waiting) { printf("step %lld | waiting for new data: every sample of the ring has been drawn %.0f times on average\n", step, max_reuse); fflush(stdout); waiting = true; t_log = now(); since = 0; CK(cudaMemset(n.stats, 0, 16)); }
            if (std::filesystem::exists(out + ".STOP")) { save_all(); std::error_code ec; std::filesystem::rename(out + ".STOP", out + ".STOP.done", ec); printf("stopped on request at step %lld\n", step); fflush(stdout); goto finish; }
            std::this_thread::sleep_for(std::chrono::seconds(1));
            t_log = now();
            continue;
        }
        waiting = false; allow -= B;
        if (revive_every > 0 && step - last_revive >= revive_every) { revive(n, ring, st, revive_batches, revive_target, 99991ULL * (uint64_t)step + 3, step); last_revive = step; }
        draw_batch(n, ring, rng, st);
        fwd_bwd(n, it);
        ++step; ++ran; ++since; ++adam_t;
        float gs = 1.f;
        if (clip > 0) {
            CK(cudaMemsetAsync(nrm, 0, 4));
            for (int i = 0; i < 7; ++i) k_sumsq<<<L((long long)((n.psize[i] + 31) / 32 * 32))>>>(n.G[i], nrm, (int)n.psize[i]);
            float ss; CK(cudaMemcpy(&ss, nrm, 4, cudaMemcpyDeviceToHost));
            float g = sqrtf(ss); if (g > clip) gs = clip / g;
        }
        float sched = (float)pow(0.5, (double)step / half_life);
        float c1 = 1.f - (float)pow(0.9, (double)std::min<long long>(adam_t, 100000)), c2 = 1.f - (float)pow(0.999, (double)std::min<long long>(adam_t, 100000));
        for (int i = 0; i < 7; ++i) {
            float base = i == 0 ? lr : (i <= 2 ? lr_neuron : lr_plug);
            float l = std::max(lr_min * base / lr, base * sched);
            k_adam<<<L((long long)n.psize[i])>>>(n.P[i], n.G[i], n.M[i], n.V[i], n.EMA[i], (int)n.psize[i], l, c1, c2, gs, ema);
        }
        if (since >= log_every) {
            float stt[4]; CK(cudaMemcpy(stt, n.stats, 16, cudaMemcpyDeviceToHost));          // sums over the steps since the last line
            CK(cudaMemset(n.stats, 0, 16));
            double t = now();
            char late[64] = "";
            if (!max_steps && t - t_new > 900) snprintf(late, sizeof late, " | NO NEW DATA for %.0f min", (t - t_new) / 60);
            printf("step %lld | %.0f samples/s | kl %.4f agree %.4f | lr %.2e | ring %d, newest %s%s\n", step, since * (double)B / (t - t_log), stt[0] / ((double)B * since),
                   stt[1] / ((double)B * since), std::max(lr_min, lr * sched), (int)ring.size(), ring.back()->name.c_str(), late);
            fflush(stdout); t_log = t; since = 0;
        }
        if (eval_every > 0 && step % eval_every == 0) { if (held) eval_on(held, 4242, "held out"); else eval_on(ring.back(), 4242, "in the ring"); }
        bool stop = std::filesystem::exists(out + ".STOP");
        if (stop || now() - t_save > save_sec) {
            save_all();
            if (stop) { std::error_code ec; std::filesystem::rename(out + ".STOP", out + ".STOP.done", ec); printf("stopped on request at step %lld\n", step); fflush(stdout); goto finish; }
        }
    }
    save_all();
    printf("done at step %lld\n", step); fflush(stdout);
finish:
    ld.quit = true; loader.join();
    std::_Exit(rc);                                       // (no GPU teardown: it has hung before)
}
#endif
