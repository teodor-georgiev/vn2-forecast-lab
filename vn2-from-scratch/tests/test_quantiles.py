"""Causal quantile forecasts built on the seasonal reference.

Covers the Task 4 brief: ``quantile_forecast(y, in_stock, t)`` returns
shape (n, 3, len(taus)), monotone in tau via cummax, non-negative via
clip at 0, and causal — byte-identical under post-t mutation of sales
and stock masks. Stockouts are censored (exclude invalid weeks from
residuals), never observed zeros.
"""

import numpy as np

from vn2_from_scratch.quantiles import quantile_forecast


def _panel(n_series: int = 8, weeks: int = 32, seed: int = 7):
    rng = np.random.default_rng(seed)
    level = rng.uniform(5.0, 60.0, size=(n_series, 1))
    season = 1.0 + 0.35 * np.sin(2 * np.pi * np.arange(weeks) / 52.0)
    y = level * season[None, :] * rng.gamma(2.0, 0.5, size=(n_series, weeks))
    in_stock = rng.random((n_series, weeks)) > 0.12
    return y, in_stock


def _y():
    return _panel()[0]


def _s():
    return _panel()[1]


def test_quantiles_monotone_nonneg():
    q = quantile_forecast(_y(), _s(), 12)
    assert q.shape[1] == 3 and np.all(q >= 0)
    assert np.all(q[:, :, 1:] >= q[:, :, :-1])


def test_quantiles_shape_and_custom_taus():
    y, s = _panel()
    q = quantile_forecast(y, s, 16, taus=(0.5, 0.9))
    assert q.shape == (8, 3, 2)
    assert np.isfinite(q).all()
    assert np.all(q >= 0)
    assert np.all(q[:, :, 1:] >= q[:, :, :-1])


def test_quantiles_default_shape():
    y, s = _panel()
    q = quantile_forecast(y, s, 12)
    assert q.shape == (8, 3, 4)


def test_quantiles_causal_post_t_mutation():
    y, s = _panel()
    keep_y = y[:, 12:].copy()
    keep_s = s[:, 12:].copy()
    a = quantile_forecast(y, s, 12)
    y[:, 12:] += 1e6
    s[:, 12:] = False
    b = quantile_forecast(y, s, 12)
    assert np.array_equal(a, b), (
        f"quantiles leaked past t (max abs diff {float(np.max(np.abs(a - b))):.3e})"
    )
    y[:, 12:] = keep_y
    s[:, 12:] = keep_s


def test_quantiles_causal_const_shift():
    y, s = _panel()
    keep_y = y[:, 16:].copy()
    a = quantile_forecast(y, s, 16)
    y[:, 16:] = 1.0
    b = quantile_forecast(y, s, 16)
    y[:, 16:] = keep_y
    assert np.array_equal(a, b)


def test_quantiles_past_perturbation_changes_output():
    """Harness self-check: outputs depend on [0:t), not constant."""
    y, s = _panel()
    a = quantile_forecast(y, s, 16)
    y0 = y.copy()
    y[:, 12:16] += 37.0
    b = quantile_forecast(y, s, 16)
    y[:, :] = y0
    assert not np.array_equal(a, b)


def test_quantiles_censored_values_ignored():
    """Mutating sales where stock is False must not move the forecast."""
    y, s = _panel()
    a = quantile_forecast(y, s, 16)
    y_censored = y.copy()
    y_censored[~s] = 9999.0
    b = quantile_forecast(y_censored, s, 16)
    assert np.array_equal(a, b)


def test_quantiles_does_not_mutate_inputs():
    y, s = _panel()
    y0, s0 = y.copy(), s.copy()
    quantile_forecast(y, s, 16)
    np.testing.assert_array_equal(y, y0)
    np.testing.assert_array_equal(s, s0)


def test_quantiles_empty_history_edge():
    y, s = _panel()
    q = quantile_forecast(y, s, 0)
    assert q.shape == (8, 3, 4)
    assert np.all(q == 0.0)


def test_quantiles_real_panel():
    from vn2_from_scratch.data import load_dataset
    from vn2_from_scratch.organizer import full_panel

    ds = load_dataset()
    _, woy, _ = full_panel(ds)
    q = quantile_forecast(ds.history, ds.in_stock, 60, woy=woy)
    assert q.shape == (599, 3, 4)
    assert np.isfinite(q).all()
    assert np.all(q >= 0)
    assert np.all(q[:, :, 1:] >= q[:, :, :-1])
