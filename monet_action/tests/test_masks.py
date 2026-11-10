"""Critical tests 7-8 (part) and 10 (part): masks, horizon padding, conditioning groups, interventions."""
import numpy as np
import pytest
import torch

from monet_action.conditioning import (
    TOKEN_LATENT,
    TOKEN_NULL,
    TOKEN_VL,
    LatentBank,
    LatentToActionConditioner,
    apply_latent_intervention,
)
from monet_action.config import ConditionerConfig
from monet_action.dit_action import FlowMatchingDiT


def _dit():
    torch.manual_seed(0)
    return FlowMatchingDiT(action_dim=7, state_dim=8, horizon=8, dim=32, num_layers=4, heads=4, cond_dim=16,
                           dropout=0.0).eval()


def test_masked_condition_keys_do_not_affect_output():
    m = _dit()
    x, t, st = torch.randn(2, 8, 7), torch.tensor([3, 700]), torch.randn(2, 8)
    cond = torch.randn(2, 6, 16)
    mask = torch.tensor([[1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 1, 0]], dtype=torch.bool)
    v1 = m(x, t, cond, mask, state=st)
    cond2 = cond.clone()
    cond2[~mask] = 1e3 * torch.randn(int((~mask).sum()), 16)
    v2 = m(x, t, cond2, mask, state=st)
    torch.testing.assert_close(v1, v2, atol=1e-5, rtol=1e-5)
    mask3 = mask.clone()
    mask3[0, 3] = True
    assert (m(x, t, cond2, mask3, state=st) - v1).abs().max() > 1e-3


def test_action_token_mask_isolates_padded_horizon():
    m = _dit()
    x, t, st = torch.randn(1, 8, 7), torch.tensor([100]), torch.randn(1, 8)
    cond, cm = torch.randn(1, 4, 16), torch.ones(1, 4, dtype=torch.bool)
    am = torch.tensor([[1, 1, 1, 1, 1, 0, 0, 0]], dtype=torch.bool)
    v1 = m(x, t, cond, cm, state=st, action_token_mask=am)
    x2 = x.clone()
    x2[0, 5:] = 50.0
    v2 = m(x2, t, cond, cm, state=st, action_token_mask=am)
    torch.testing.assert_close(v1[:, :5], v2[:, :5], atol=1e-5, rtol=1e-5)
    assert v1[:, 5:].abs().max() == 0


def _cond(source, K=4):
    cfg = ConditionerConfig(feature_source=source, cond_dim=16, max_latents=8)
    c = LatentToActionConditioner(32, cfg).eval()
    vl, vm = torch.randn(2, 5, 32), torch.tensor([[0, 0, 1, 1, 1], [1, 1, 1, 1, 1]], dtype=torch.bool)
    lat = torch.randn(2, K, 32)
    return c, c(vl, vm, lat), vl, vm, lat


@pytest.mark.parametrize("source,n_vl,n_lat", [("vl_plus_latents", 5, 4), ("vl_only", 5, 0), ("latents_only", 0, 4)])
def test_feature_source_switch(source, n_vl, n_lat):
    c, out, *_ = _cond(source)
    types = out.token_types.tolist()
    assert types[0] == TOKEN_NULL
    assert types.count(TOKEN_VL) == n_vl and types.count(TOKEN_LATENT) == n_lat
    assert out.tokens.shape == (2, 1 + n_vl + n_lat, 16) and out.mask.shape == (2, 1 + n_vl + n_lat)
    assert out.mask[:, 0].all()
    lat_order = out.latent_order[out.token_types == TOKEN_LATENT].tolist()
    assert lat_order == list(range(n_lat))  # latent ordering preserved


def test_null_token_prevents_empty_rows():
    c, _, vl, vm, lat = _cond("latents_only")
    out = c(vl, vm, lat, torch.zeros(2, 4, dtype=torch.bool))
    assert out.mask.sum(1).min() >= 1
    m = _dit()
    v = m(torch.randn(2, 8, 7), torch.tensor([1, 2]), torch.randn(2, out.tokens.shape[1], 16), out.mask,
          state=torch.randn(2, 8))
    assert torch.isfinite(v).all()


def test_latent_order_embedding_makes_shuffle_visible():
    c, out, vl, vm, lat = _cond("vl_plus_latents")
    out2 = c(vl, vm, lat[:, [3, 2, 1, 0]])
    assert (out.tokens - out2.tokens).abs().max() > 1e-4


def test_interventions():
    lat = torch.randn(3, 6, 8)
    g = torch.Generator().manual_seed(0)
    z, _ = apply_latent_intervention(lat, None, "zero")
    assert z.abs().sum() == 0
    _, m = apply_latent_intervention(lat, None, "mask")
    assert m.sum() == 0
    s, _ = apply_latent_intervention(lat, None, "shuffle_order", generator=g)
    assert not torch.equal(s, lat) and torch.allclose(s.sort(1).values, lat.sort(1).values)
    _, m = apply_latent_intervention(lat, None, "truncate:2")
    assert m[:, :2].all() and not m[:, 2:].any()
    gauss, _ = apply_latent_intervention(lat, None, "gaussian", generator=g)
    assert abs(float(gauss.mean() - lat.mean())) < 0.5 and not torch.equal(gauss, lat)
    bank = LatentBank()
    bank.add(torch.ones(2, 6, 8), keys=[1, 1])
    bank.add(2 * torch.ones(1, 6, 8), keys=[2])
    sw, _ = apply_latent_intervention(lat[:1], None, "swap", bank=bank, bank_keys=[1], generator=g)
    assert torch.equal(sw, 2 * torch.ones(1, 6, 8))  # never draws from the same task
    with pytest.raises(ValueError):
        apply_latent_intervention(lat, None, "nonsense")


def test_no_cross_episode_leakage_and_horizon_padding():
    from monet_action.libero_dataset import EpisodeInfo, LiberoWindowDataset

    class FakeFrames:
        camera_keys = ["image"]
        state = np.arange(20 * 8, dtype=np.float32).reshape(20, 8)
        action = np.repeat(np.arange(20, dtype=np.float32)[:, None], 7, 1)

        def image(self, k, row):
            return np.full((4, 4, 3), row, np.uint8)

    eps = [EpisodeInfo(0, 0, "a", 0, 12), EpisodeInfo(1, 0, "a", 12, 8)]
    ds = LiberoWindowDataset(FakeFrames(), eps, horizon=8)
    assert len(ds) == 20
    last0 = ds[11]  # final frame of episode 0
    assert last0["actions"][:, 0].tolist() == [11.0] * 8  # repeats a[T-1], never episode 1
    assert last0["loss_mask"].tolist() == [1, 0, 0, 0, 0, 0, 0, 0]
    mid = ds[8]
    assert mid["actions"][:, 0].tolist() == [8, 9, 10, 11, 11, 11, 11, 11]
    assert mid["loss_mask"].tolist() == [1, 1, 1, 1, 0, 0, 0, 0]
    first1 = ds[12]
    assert first1["actions"][:, 0].tolist() == list(range(12, 20)) and first1["loss_mask"].all()
    assert int(first1["images"][0][0, 0, 0]) == 12
