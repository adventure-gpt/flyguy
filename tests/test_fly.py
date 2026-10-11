"""flyguy: an actor wired like a (here: made-up, small) connectome - flyguy/fly.py."""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flyguy.fly import FlyActor  # noqa: E402
from flyguy.obs import OBS_DIM  # noqa: E402


def _graph(tmp_path, n=80, n_sens=14, n_dn=7, edges=420, seed=0):
    rng = np.random.default_rng(seed)
    pairs = set()
    while len(pairs) < edges:
        a, b = int(rng.integers(n)), int(rng.integers(n))
        if a != b:
            pairs.add((a, b))
    pre, post = (np.array(x, np.int32) for x in zip(*sorted(pairs)))
    ids = rng.permutation(n)
    f = tmp_path / "toy.npz"
    np.savez(f, body=np.arange(n) + 1000, cls=np.array([""] * n), nt=rng.choice(["acetylcholine", "gaba", "glutamate"], n),
             pre=pre, post=post, syn=rng.integers(2, 40, len(pre)).astype(np.int32),
             sensory=np.sort(ids[:n_sens]).astype(np.int32), descending=np.sort(ids[n_sens:n_sens + n_dn]).astype(np.int32))
    return f, n, pre, post


def _blind(f: FlyActor, obs, n, pre, post):
    """The same network computed without skipping anything: every neuron, every connection, every hop."""
    W = torch.zeros(n, n)
    W[torch.as_tensor(post).long(), torch.as_tensor(pre).long()] = f.w.detach()
    drive_in = torch.zeros(obs.shape[0], f.sizes[0]).index_add_(1, f.in_r, obs[:, f.in_c] * f.in_w.detach()) + f.in_b.detach()
    drive = torch.zeros(obs.shape[0], n)
    drive[:, f.n0] = drive_in
    thr = torch.nn.functional.softplus(f.theta.detach())
    h = torch.relu(f.gain.detach() * drive - thr)
    for _ in range(f.hops):
        h = torch.relu(f.gain.detach() * (h @ W.t() + drive) - thr)
    dn_ids = getattr(f, f"n{f.hops}")[f.dn_rows]
    return h[:, dn_ids] @ f.out.weight.detach().t() + f.out.bias.detach()


def test_skipping_idle_connections_changes_nothing_and_both_paths_agree(tmp_path):
    """The fly actor computes only connections that can carry the signal to an output in the hops left; the result
    equals the blind computation over every neuron and connection. The sparse-product path with its own backward
    gives the outputs and gradients of the plain gather path. One strength per connection, shared by the hops."""
    g, n, pre, post = _graph(tmp_path)
    torch.manual_seed(0)
    for hops in (2, 3, 4):
        f = FlyActor(g, OBS_DIM, 90, hops=hops, fan=5)
        with torch.no_grad():
            f.w.mul_(8.0); f.out.weight.normal_(0, 0.3); f.theta.fill_(-2.0)      # lively, so the comparison means something
        obs = torch.randn(6, OBS_DIM)
        out = f(obs)
        assert out.shape == (6, 90) and torch.isfinite(out).all()
        ref = _blind(f, obs, n, pre, post)
        assert torch.allclose(out, ref, atol=1e-4, rtol=1e-4), float((out - ref).abs().max())
        assert f.used <= hops * f.n_conn and f.in_play <= f.n_conn and f.w.numel() == f.n_conn
        (out ** 2).mean().backward()
        g_gather = {k: p.grad.clone() for k, p in f.named_parameters() if p.grad is not None}
        assert int((g_gather["w"] != 0).sum()) <= f.in_play
        try:
            f.gather_limit = 0
            f.zero_grad()
            out2 = f(obs)
            (out2 ** 2).mean().backward()
        except RuntimeError as e:                               # a build without the sampled product on this device
            if "sampled_addmm" in str(e) or "not implemented" in str(e).lower():
                continue
            raise
        assert torch.allclose(out, out2, atol=1e-5)
        for k, v in g_gather.items():
            assert torch.allclose(v, dict(f.named_parameters())[k].grad, atol=1e-6, rtol=1e-4), k


def test_compiled_forward_matches(tmp_path, monkeypatch):
    """The bot's compiled forward pass (flyguy/native/fly_infer.dll) answers like the PyTorch path and like the
    network computed blind, and hands the visualizer the same bytes."""
    import pytest
    from flyguy.fly import ROOT
    if not (ROOT / "flyguy" / "native" / "fly_infer.dll").exists():
        pytest.skip("fly_infer.dll is not built")
    path, n, pre, post = _graph(tmp_path)
    f = FlyActor(str(path), OBS_DIM, 90, hops=3, fan=6)
    torch.manual_seed(1)
    with torch.no_grad():
        f.w.mul_(3.0)
        f.theta.add_(torch.randn_like(f.theta))
        f.gain.mul_(1 + 0.3 * torch.randn_like(f.gain))
    obs = torch.randn(5, OBS_DIM)
    taps = []

    def tap(a):
        taps.append(a.copy() if isinstance(a, np.ndarray) else np.concatenate([(torch.tanh(x * 1.5) * 255.0).to(torch.uint8).numpy() for x in a]))
    f.tap = tap
    with torch.no_grad():
        got = f(obs)
        assert f._native() is not None
        monkeypatch.setenv("FLYGUY_NATIVE", "0")
        f._nat = None
        ref = f(obs)
        assert f._native() is None
    assert torch.allclose(got, ref, atol=1e-4)
    assert torch.allclose(got, _blind(f, obs, n, pre, post), atol=1e-4)
    assert len(taps) == 2 and taps[0].shape == taps[1].shape
    assert np.abs(taps[0].astype(int) - taps[1].astype(int)).max() <= 1


def test_trimmed_graph_is_the_same_network(tmp_path):
    """flyguy/trim_graph.py: a graph file with only the connections a network of N hops uses builds the same network
    (same start, same neurons per hop), a checkpoint cut down to it answers the same, and a network of more hops
    than the file was cut for is refused."""
    import pytest
    from flyguy.trim_graph import in_play, trim
    g, n, pre, post = _graph(tmp_path)
    keep = trim(g, tmp_path / "toy3.npz", 3)
    assert 0 < keep.sum() < len(keep) and np.array_equal(keep, in_play(g, 3))
    torch.manual_seed(2)
    a = FlyActor(g, OBS_DIM, 90, hops=3, fan=5)
    with torch.no_grad():
        a.w.mul_(1 + torch.rand_like(a.w)); a.theta.fill_(-2.0); a.out.weight.normal_(0, 0.3)
    b = FlyActor(tmp_path / "toy3.npz", OBS_DIM, 90, hops=3, fan=5)
    sd = {k: (v[torch.as_tensor(keep)] if k == "w" else v) for k, v in a.state_dict().items()}
    b.load_state_dict(sd)
    obs = torch.randn(6, OBS_DIM)
    assert a.whole_graph and not b.whole_graph and b.sizes == a.sizes
    assert torch.allclose(a(obs), b(obs), atol=1e-6)
    two = FlyActor(tmp_path / "toy3.npz", OBS_DIM, 90, hops=2, fan=5)        # fewer hops: still the whole of that network
    assert two.sizes == FlyActor(g, OBS_DIM, 90, hops=2, fan=5).sizes
    with pytest.raises(AssertionError):
        FlyActor(tmp_path / "toy3.npz", OBS_DIM, 90, hops=4, fan=5)
