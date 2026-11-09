"""Stock-out imputation tests.

Stockouts are CENSORED demand (unobserved), never zero demand: censored
weeks must be filled from observed in-stock weeks only, causally from
columns [0:upto), with a cross-series median fallback when a series has
no observed week in the window.
"""

import numpy as np

from vn2_from_scratch.impute import impute_panel


def test_impute_all_censored_fallback():
    y = np.ones((3, 10))
    s = np.zeros((3, 10), dtype=bool)
    out = impute_panel(np.nan_to_num(y), s, 10)
    assert np.all(np.isfinite(out))


def test_impute_forward_fill():
    hist = np.array([[1.0, 99.0, 3.0, 99.0, 5.0]])
    stock = np.array([[True, False, True, False, True]])
    out = impute_panel(hist, stock, 5)
    np.testing.assert_array_equal(out, [[1.0, 1.0, 3.0, 3.0, 5.0]])


def test_impute_leading_censored_uses_cross_series_median():
    # Series 0 is all-censored; series 1 starts censored. Censored means are
    # [NaN, 10, 20], so the cross-series median fallback is 15.
    hist = np.array([[5.0, 5.0, 5.0], [10.0, 10.0, 10.0], [20.0, 20.0, 20.0]])
    stock = np.array([[False, False, False], [False, True, True], [True, True, True]])
    out = impute_panel(hist, stock, 3)
    np.testing.assert_array_equal(out[0], [15.0, 15.0, 15.0])
    assert out[1, 0] == 15.0
    np.testing.assert_array_equal(out[1, 1:], [10.0, 10.0])


def test_impute_is_causal_beyond_upto():
    rng = np.random.default_rng(0)
    hist = rng.uniform(1.0, 50.0, size=(4, 12))
    stock = rng.random((4, 12)) > 0.2
    a = impute_panel(hist, stock, 6)
    hist_mut = hist.copy()
    stock_mut = stock.copy()
    hist_mut[:, 6:] = 999.0
    stock_mut[:, 6:] = False
    b = impute_panel(hist_mut, stock_mut, 6)
    np.testing.assert_array_equal(a[:, :6], b[:, :6])
    # Columns at/after upto are passed through unchanged.
    np.testing.assert_array_equal(a[:, 6:], hist[:, 6:])


def test_impute_same_shape_finite_non_negative():
    rng = np.random.default_rng(1)
    hist = rng.uniform(0.0, 80.0, size=(599, 157))
    stock = rng.random((599, 157)) > 0.106
    out = impute_panel(hist, stock, 157)
    assert out.shape == hist.shape
    assert np.isfinite(out).all()
    assert (out >= 0).all()
    # Observed weeks keep their observed value.
    assert np.array_equal(out[stock], hist[stock])


def test_impute_does_not_mutate_inputs():
    hist = np.array([[1.0, 99.0, 3.0]])
    stock = np.array([[True, False, True]])
    hist0, stock0 = hist.copy(), stock.copy()
    impute_panel(hist, stock, 3)
    np.testing.assert_array_equal(hist, hist0)
    np.testing.assert_array_equal(stock, stock0)


def test_impute_non_finite_and_negative_observed_are_sanitised():
    hist = np.array([[1.0, np.nan, -2.0, 4.0]])
    stock = np.array([[True, True, True, True]])
    out = impute_panel(hist, stock, 4)
    assert np.isfinite(out).all()
    assert (out >= 0).all()


def test_impute_negative_history_is_unobserved_not_zero():
    # A negative history value must be treated as unobserved, never as an
    # observed 0.0 that LOCF then propagates.
    # Negative between observed values: LOCF from the prior observed week.
    hist = np.array([[5.0, -1.0, 2.0]])
    stock = np.array([[True, True, True]])
    out = impute_panel(hist, stock, 3)
    np.testing.assert_array_equal(out, [[5.0, 5.0, 2.0]])

    # Negative followed by a censored week: the censored week must forward
    # from the last genuine observed value, not from a clipped 0.0.
    hist = np.array([[4.0, -3.0, 99.0]])
    stock = np.array([[True, True, False]])
    out = impute_panel(hist, stock, 3)
    np.testing.assert_array_equal(out, [[4.0, 4.0, 4.0]])
    assert not (out == 0.0).any(), "negative must not become an observed zero"

    # Leading negative with no prior observed: fallback (cross-series
    # median of observed means; here the single series' mean 3.0), not 0.0.
    hist = np.array([[-5.0, 3.0]])
    stock = np.array([[True, True]])
    out = impute_panel(hist, stock, 2)
    np.testing.assert_array_equal(out, [[3.0, 3.0]])


def test_impute_upto_zero_edge():
    # Empty window: nothing to fill from; input is returned unchanged.
    hist = np.array([[1.0, 2.0], [3.0, 4.0]])
    stock = np.array([[True, False], [False, True]])
    out = impute_panel(hist, stock, 0)
    np.testing.assert_array_equal(out, hist)


def test_impute_real_panel():
    from vn2_from_scratch.data import load_dataset

    ds = load_dataset()
    out = impute_panel(ds.history, ds.in_stock, ds.history.shape[1])
    assert out.shape == ds.history.shape
    assert np.isfinite(out).all()
    assert (out >= 0).all()
