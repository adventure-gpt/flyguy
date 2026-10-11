// flyguy: the PLAYING network on the CPU, for the RLBot bot (2026-10-09). flyguy/fly.py loads this DLL when it is there
// and hands it the same arrays its own forward pass uses; the answer is the same network's (sums in another order).
//
// Why: RLBot's performance figure is the share of game ticks the bot answers in time, a tick is 8.3 ms, and a decision
// through PyTorch's sparse products took 5 ms plus the rest of the bot's work - Flyguy showed 80-90% (50% before
// that). Here a hop only walks the connections of the neurons that FIRE (about a third of them): one pass over
// roughly 1.3 of the 3.7 million connection steps, no allocation, no interpreter.
//
//   build (Visual Studio Build Tools):  flyguy\native\build_infer.bat   ->  flyguy\native\fly_infer.dll
#include <cmath>
#include <cstring>
#include <vector>
#ifdef _WIN32
#define API extern "C" __declspec(dllexport)
#else
#define API extern "C"
#endif

struct Hop {
    int rows = 0, src = 0;
    std::vector<int> tcrow, tcol, drow;      // by SOURCE neuron: its connections (target row); the sensory drive row of a target or -1
    std::vector<float> w, g, t;              // strengths in that order; gain and threshold per target row
};
struct Fly {
    int obs_dim = 0, n_act = 0, hops = 0, n_s = 0, n_dn = 0;
    std::vector<int> in_crow, in_col, dn;
    std::vector<float> in_w, in_b, g0, t0, out_w, out_b;
    std::vector<Hop> hop;
    std::vector<float> drive, a, b;
};
static inline unsigned char shade(float v) { return v > 0.f ? (unsigned char)(255.f * tanhf(1.5f * v)) : 0; }

API void* fly_create(int obs_dim, int n_act, int hops) {
    Fly* f = new Fly; f->obs_dim = obs_dim; f->n_act = n_act; f->hops = hops; f->hop.resize(hops);
    return f;
}
API void fly_set_plug(void* p, int n_s, int E, const int* crow, const int* col, const float* w, const float* b, const float* g0, const float* t0) {
    Fly* f = (Fly*)p; f->n_s = n_s;
    f->in_crow.assign(crow, crow + n_s + 1); f->in_col.assign(col, col + E); f->in_w.assign(w, w + E);
    f->in_b.assign(b, b + n_s); f->g0.assign(g0, g0 + n_s); f->t0.assign(t0, t0 + n_s);
    f->drive.resize(n_s);
}
API void fly_set_hop(void* p, int k, int rows, int src, int E, const int* tcrow, const int* tcol, const float* w, const float* g, const float* t, const int* drow) {
    Hop& h = ((Fly*)p)->hop[k]; h.rows = rows; h.src = src;
    h.tcrow.assign(tcrow, tcrow + src + 1); h.tcol.assign(tcol, tcol + E); h.w.assign(w, w + E);
    h.g.assign(g, g + rows); h.t.assign(t, t + rows); h.drow.assign(drow, drow + rows);
}
API void fly_set_out(void* p, int n_dn, const int* dn, const float* w, const float* b) {
    Fly* f = (Fly*)p; f->n_dn = n_dn;
    f->dn.assign(dn, dn + n_dn); f->out_w.assign(w, w + (size_t)f->n_act * n_dn); f->out_b.assign(b, b + f->n_act);
    size_t mx = f->n_s;
    for (auto& h : f->hop) mx = mx > (size_t)h.rows ? mx : (size_t)h.rows;
    f->a.resize(mx); f->b.resize(mx);
}
// obs [batch, obs_dim] -> logits [batch, n_act]; tap (or null): one byte per neuron state of the FIRST car, layer after
// layer (sensory first), 255 * tanh(1.5 * firing) - what the visualizer draws
API void fly_forward(void* p, const float* obs, int batch, float* logits, unsigned char* tap) {
    Fly& f = *(Fly*)p;
    for (int c = 0; c < batch; ++c) {
        const float* x = obs + (size_t)c * f.obs_dim;
        unsigned char* tp = c == 0 ? tap : nullptr;
        float *hp = f.a.data(), *hn = f.b.data();
        for (int r = 0; r < f.n_s; ++r) {
            float acc = f.in_b[r];
            for (int e = f.in_crow[r]; e < f.in_crow[r + 1]; ++e) acc += f.in_w[e] * x[f.in_col[e]];
            f.drive[r] = acc;
            float v = f.g0[r] * acc - f.t0[r];
            hp[r] = v > 0.f ? v : 0.f;
        }
        if (tp) { for (int r = 0; r < f.n_s; ++r) *tp++ = shade(hp[r]); }
        for (auto& h : f.hop) {
            memset(hn, 0, (size_t)h.rows * sizeof(float));
            const int* tcrow = h.tcrow.data(); const int* tcol = h.tcol.data(); const float* w = h.w.data();
            for (int j = 0; j < h.src; ++j) {
                const float v = hp[j];
                if (v <= 0.f) continue;                                  // a silent neuron sends nothing
                const int e1 = tcrow[j + 1];
                for (int e = tcrow[j]; e < e1; ++e) hn[tcol[e]] += w[e] * v;
            }
            for (int r = 0; r < h.rows; ++r) {
                float acc = hn[r];
                const int d = h.drow[r];
                if (d >= 0) acc += f.drive[d];                           // sensory neurons stay driven by the game state
                const float v = h.g[r] * acc - h.t[r];
                hn[r] = v > 0.f ? v : 0.f;
            }
            if (tp) { for (int r = 0; r < h.rows; ++r) *tp++ = shade(hn[r]); }
            float* s = hp; hp = hn; hn = s;
        }
        float* lg = logits + (size_t)c * f.n_act;
        for (int a = 0; a < f.n_act; ++a) {
            float acc = f.out_b[a];
            const float* w = f.out_w.data() + (size_t)a * f.n_dn;
            for (int j = 0; j < f.n_dn; ++j) acc += w[j] * hp[f.dn[j]];
            lg[a] = acc;
        }
    }
}
API void fly_free(void* p) { delete (Fly*)p; }
