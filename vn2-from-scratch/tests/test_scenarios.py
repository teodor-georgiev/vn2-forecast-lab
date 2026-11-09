"""SIP/SLURP scenario packs from quantile forecasts.

Covers the Task 5 brief: ``pack_scenarios(quantiles, n_scenarios, seed)``
returns shape (n_scenarios, n_series, 3), finite, non-negative, seed-stable,
via piecewise CDF inversion with one uniform per series-scenario so
cross-horizon (cross-week) dependence is preserved.
"""

import numpy as np

from vn2_from_scratch.scenarios import pack_scenarios


def _q(n_series: int = 4, fill: float = 10.0):
    return np.ones((n_series, 3, 2)) * fill


def test_pack_seed_stable_finite():
    a = pack_scenarios(np.ones((4, 3, 2)) * 10, 16, seed=7)
    b = pack_scenarios(np.ones((4, 3, 2)) * 10, 16, seed=7)
    assert np.array_equal(a, b) and np.all(np.isfinite(a)) and np.all(a >= 0)


def test_pack_shape():
    out = pack_scenarios(_q(4), 16, seed=7)
    assert out.shape == (16, 4, 3)
    assert np.all(np.isfinite(out))
    assert np.all(out >= 0)


def test_pack_different_seeds_differ():
    q = np.zeros((8, 3, 4))
    spreads = np.array([0.0, 10.0, 20.0, 40.0])
    for h in range(3):
        q[:, h, :] = spreads[None, :]
    a = pack_scenarios(q, 32, seed=7)
    b = pack_scenarios(q, 32, seed=8)
    assert not np.array_equal(a, b)


def test_pack_bounded_by_quantile_range():
    """Every draw lies within its series/horizon [q_min, q_max] range."""
    rng = np.random.default_rng(0)
    q = rng.uniform(0.0, 50.0, size=(6, 3, 4))
    q = np.maximum.accumulate(np.clip(q, 0.0, None), axis=2)
    out = pack_scenarios(q, 64, seed=3)
    for h in range(3):
        lo = q[:, h, 0]
        hi = q[:, h, -1]
        assert np.all(out[:, :, h] >= lo[None, :] - 1e-9)
        assert np.all(out[:, :, h] <= hi[None, :] + 1e-9)


def test_pack_cross_week_dependence_one_uniform():
    """Identical CDFs across horizons plus one uniform => identical draws.

    Quantiles are the same for every horizon; with a single uniform per
    series-scenario (the dependence-preserving SIP/SLURP choice) the three
    horizon draws must agree. Independent per-horizon uniforms would diverge.
    """
    q = np.zeros((5, 3, 2))
    q[:, :, 0] = 0.0
    q[:, :, 1] = 100.0
    out = pack_scenarios(q, 64, seed=11)
    assert np.array_equal(out[:, :, 0], out[:, :, 1])
    assert np.array_equal(out[:, :, 0], out[:, :, 2])


def test_pack_constant_in_tau_is_constant():
    out = pack_scenarios(_q(4, 12.0), 8, seed=1)
    assert np.all(out == 12.0)


def test_pack_single_tau():
    q = np.full((3, 3, 1), 5.0)
    out = pack_scenarios(q, 4, seed=2)
    assert out.shape == (4, 3, 3)
    assert np.all(out == 5.0)


def test_pack_zero_scenarios_and_empty_series():
    assert pack_scenarios(_q(4), 0, seed=1).shape == (0, 4, 3)
    assert pack_scenarios(np.ones((0, 3, 2)), 5, seed=1).shape == (5, 0, 3)


def test_pack_does_not_mutate_inputs():
    q = _q(4)
    q0 = q.copy()
    pack_scenarios(q, 8, seed=7)
    np.testing.assert_array_equal(q, q0)


def test_pack_invalid_input_raises():
    try:
        pack_scenarios(np.ones((4, 3)), 4, seed=1)
    except ValueError:
        pass
    else:
        raise AssertionError("2-D quantiles must raise ValueError")
    try:
        pack_scenarios(_q(4), -1, seed=1)
    except ValueError:
        pass
    else:
        raise AssertionError("negative n_scenarios must raise ValueError")


def test_pack_piecewise_constant_step_draws():
    """Draws are grid quantiles only, never interpolated (brief)."""
    q = np.zeros((2, 3, 4))
    spreads = np.array([10.0, 20.0, 30.0, 40.0])
    for h in range(3):
        q[:, h, :] = spreads[None, :]
    out = pack_scenarios(q, 256, seed=13)
    assert np.all(np.isin(out, spreads)), sorted(set(out.ravel().tolist()))
    # Two-point grid: only endpoints, no blended middle values.
    q2 = np.zeros((1, 3, 2))
    q2[:, :, 0] = 0.0
    q2[:, :, 1] = 100.0
    out2 = pack_scenarios(q2, 256, seed=14)
    assert np.all(np.isin(out2, [0.0, 100.0]))
    assert np.any(out2 == 0.0) and np.any(out2 == 100.0)


def test_pack_n_taus_zero_raises():
    try:
        pack_scenarios(np.ones((2, 3, 0)), 4, seed=1)
    except ValueError:
        pass
    else:
        raise AssertionError("n_taus == 0 must raise ValueError")


def test_pack_nonfinite_rejected():
    for bad in (np.inf, -np.inf, np.nan):
        q = _q(2)
        q[0, 0, 0] = bad
        try:
            pack_scenarios(q, 4, seed=1)
        except ValueError:
            pass
        else:
            raise AssertionError(f"non-finite quantile {bad!r} must raise ValueError")


def test_pack_integration_quantile_forecast():
    from vn2_from_scratch.quantiles import quantile_forecast

    rng = np.random.default_rng(7)
    level = rng.uniform(5.0, 60.0, size=(8, 1))
    season = 1.0 + 0.35 * np.sin(2 * np.pi * np.arange(32) / 52.0)
    y = level * season[None, :] * rng.gamma(2.0, 0.5, size=(8, 32))
    in_stock = rng.random((8, 32)) > 0.12
    q = quantile_forecast(y, in_stock, 16)
    assert q.shape == (8, 3, 4)
    out = pack_scenarios(q, 24, seed=5)
    assert out.shape == (24, 8, 3)
    assert np.isfinite(out).all()
    assert (out >= 0).all()
