"""Critical tests 5-6: interpolation, velocity target, time encodings, Euler integration."""
import math

import pytest
import torch

from monet_action.config import ActionHeadConfig
from monet_action.dit_action import DiTActionHead, FlowMatchingDiT, diffusers_timestep_embedding, sinusoidal_encoding
from monet_action.flow_matching import (
    discretize_time,
    euler_integrate,
    interpolate,
    masked_mse,
    sample_time,
    velocity_target,
)


def test_interpolation_endpoints_and_velocity():
    a, eps = torch.randn(4, 8, 7), torch.randn(4, 8, 7)
    torch.testing.assert_close(interpolate(a, eps, torch.zeros(4)), eps)
    torch.testing.assert_close(interpolate(a, eps, torch.ones(4)), a)
    t = torch.tensor([0.25, 0.5, 0.75, 0.1])
    x = interpolate(a, eps, t)
    torch.testing.assert_close(x, (1 - t)[:, None, None] * eps + t[:, None, None] * a)
    v = velocity_target(a, eps)
    torch.testing.assert_close(v, a - eps)
    # d x_t / dt == v_target
    h = 1e-3
    torch.testing.assert_close((interpolate(a, eps, t + h) - x) / h, v, atol=1e-3, rtol=1e-3)


def test_time_sampling_schedule():
    g = torch.Generator().manual_seed(0)
    t = sample_time(200000, schedule="beta", alpha=1.5, beta=1.0, s=0.999, generator=g)
    assert t.min() >= 0 and t.max() <= 0.999
    # GR00T: t = (1 - Beta(1.5, 1)) * 0.999, E[Beta] = 0.6 -> E[t] = 0.3996
    assert abs(float(t.mean()) - 0.4 * 0.999) < 3e-3
    u = sample_time(100000, schedule="uniform", generator=g)
    assert abs(float(u.mean()) - 0.5) < 5e-3
    g1, g2 = torch.Generator().manual_seed(7), torch.Generator().manual_seed(7)
    torch.testing.assert_close(sample_time(16, generator=g1), sample_time(16, generator=g2))


def test_discretisation():
    t = torch.tensor([0.0, 0.0004, 0.5, 0.999, 1.0])
    assert discretize_time(t, 1000).tolist() == [0, 0, 500, 999, 999]


def test_timestep_encodings():
    ts = torch.tensor([0, 1, 500, 999])
    e = diffusers_timestep_embedding(ts, 256)
    assert e.shape == (4, 256)
    half = 128
    freqs = torch.exp(-math.log(10000) * torch.arange(half) / (half - 1))
    ref = torch.cat([torch.cos(ts[:, None] * freqs), torch.sin(ts[:, None] * freqs)], -1)
    torch.testing.assert_close(e, ref, atol=1e-5, rtol=1e-5)
    s = sinusoidal_encoding(ts.view(4, 1).expand(4, 8), 64)
    assert s.shape == (4, 8, 64) and torch.allclose(s[:, 0], s[:, 7])
    try:  # cross-check with diffusers when installed
        from diffusers.models.embeddings import get_timestep_embedding

        torch.testing.assert_close(e, get_timestep_embedding(ts, 256, flip_sin_to_cos=True, downscale_freq_shift=1))
    except ImportError:
        pass


def test_masked_mse():
    p, t = torch.zeros(2, 4, 3), torch.ones(2, 4, 3)
    m = torch.tensor([[1, 1, 0, 0], [1, 0, 0, 0]], dtype=torch.float32)
    assert abs(float(masked_mse(p, t, m)) - 1.0) < 1e-5
    t2 = t.clone()
    t2[0, 2:] = 100.0  # masked entries must not contribute
    assert abs(float(masked_mse(p, t2, m)) - 1.0) < 1e-5
    assert abs(float(masked_mse(p, t, torch.zeros(2, 4)))) < 1e-5


@pytest.mark.parametrize("steps", [1, 2, 4, 8])
def test_euler_exact_on_conditional_flow(steps):
    """For the exact conditional field v=(a-x)/(1-t), Euler reproduces a for any step count."""
    a = torch.randn(3, 8, 7)

    def vfn(x, t_disc):
        t = (t_disc.float() / 1000.0).view(-1, 1, 1)
        return (a - x) / (1 - t)

    out = euler_integrate(vfn, a.shape, steps, generator=torch.Generator().manual_seed(0))
    torch.testing.assert_close(out, a, atol=1e-4, rtol=1e-4)


def test_euler_deterministic_noise_and_step_counts():
    cfg = ActionHeadConfig(hidden_size=32, num_layers=2, num_heads=4, dropout=0.0)
    head = DiTActionHead(cfg, state_dim=8, cond_dim=16, use_state=True).eval()
    cond, mask, st = torch.randn(2, 5, 16), torch.ones(2, 5, dtype=torch.bool), torch.randn(2, 8)
    outs = {}
    for n in (2, 4, 8):
        a = head.sample(cond, mask, st, num_steps=n, generator=torch.Generator().manual_seed(3))
        b = head.sample(cond, mask, st, num_steps=n, generator=torch.Generator().manual_seed(3))
        c = head.sample(cond, mask, st, num_steps=n, generator=torch.Generator().manual_seed(4))
        assert a.shape == (2, 8, 7) and torch.isfinite(a).all()
        torch.testing.assert_close(a, b)
        assert (a - c).abs().max() > 1e-4
        outs[n] = a
    assert (outs[2] - outs[8]).abs().max() > 1e-5  # step count actually changes the integration


def test_dit_shapes_and_random_decoding():
    m = FlowMatchingDiT(action_dim=7, state_dim=8, horizon=8, dim=32, num_layers=4, heads=4, cond_dim=24, dropout=0.0)
    v = m(torch.randn(3, 8, 7), torch.tensor([0, 10, 999]), torch.randn(3, 11, 24), torch.ones(3, 11, dtype=torch.bool),
          state=torch.randn(3, 8))
    assert v.shape == (3, 8, 7) and torch.isfinite(v).all()
    blocks = [b.is_cross for b in m.blocks]
    assert blocks == [True, False, True, False]  # GR00T interleaving: cross on even, self on odd


def test_loss_decreases_on_fixed_target():
    torch.manual_seed(0)
    cfg = ActionHeadConfig(hidden_size=64, num_layers=2, num_heads=4, dropout=0.0)
    head = DiTActionHead(cfg, state_dim=8, cond_dim=16, use_state=True)
    cond, mask, st = torch.randn(16, 4, 16), torch.ones(16, 4, dtype=torch.bool), torch.randn(16, 8)
    acts = torch.tanh(cond[:, :1, :7].expand(-1, 8, -1) + 0.1 * torch.arange(8).view(1, 8, 1))
    opt = torch.optim.Adam(head.parameters(), lr=2e-3)
    losses = []
    for i in range(300):
        loss, _ = head.loss(acts, torch.ones(16, 8), cond, mask, st, generator=torch.Generator().manual_seed(i))
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss))
    assert sum(losses[-20:]) / 20 < 0.5 * sum(losses[:20]) / 20
    pred = head.sample(cond, mask, st, num_steps=8, generator=torch.Generator().manual_seed(0))
    assert float(((pred - acts) ** 2).mean()) < float((acts ** 2).mean())
