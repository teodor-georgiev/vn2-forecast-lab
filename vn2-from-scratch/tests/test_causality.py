"""Anti-leakage gate: decision-time features must not see the future.

The mutation is performed IN PLACE on a single allocation. That matters: two
arrays holding equal values but living in different buffers can disagree in the
last few ULPs, because numpy's SIMD reductions choose their unrolling order
from the base pointer's alignment. Comparing two separately-allocated panels
would therefore report a spurious ~1e-14 "leak" even for a perfectly causal
function. Mutating and restoring one buffer removes that artefact entirely, so
byte-identity becomes both achievable and meaningful.

Each case perturbs the future two different ways, because a leak can be
shape-dependent: additive noise, and a coherent constant shift.
"""

import numpy as np
import pytest

from vn2_from_scratch import forecast
from vn2_from_scratch.data import load_dataset
from vn2_from_scratch.forecast import (
    fit_lgbm,
    lgbm_forecast,
    make_features,
    seasonal_forecast,
)
from vn2_from_scratch.impute import impute_panel
from vn2_from_scratch.organizer import full_panel, global_season

CUTOFFS = (40, 90, 140, 157)


@pytest.fixture(scope="module")
def panel():
    ds = load_dataset()
    y, woy, _ = full_panel(ds)
    return ds, y, woy


def test_features_ignore_columns_at_or_after_cutoff(panel):
    ds, y, woy = panel
    obs = np.isfinite(y)

    for cutoff in CUTOFFS:
        keep = y[:, cutoff:].copy()
        keep_obs = obs[:, cutoff:].copy()
        # The decision time must equal the mutation cutoff. Columns before it
        # are legitimately in the past and are allowed to move the result.
        a = make_features(y, obs, cutoff)
        for i in range(2):
            future = np.random.default_rng(cutoff + i).uniform(
                0.0, 90.0, size=(y.shape[0], y.shape[1] - cutoff)
            )
            if i == 1:
                future = np.full_like(future, 1.0)
            y[:, cutoff:] = future
            obs[:, cutoff:] = np.isfinite(future)
            b = make_features(y, obs, cutoff)
            y[:, cutoff:] = keep
            obs[:, cutoff:] = keep_obs
            assert np.array_equal(a, b), (
                f"features leaked past cutoff {cutoff} "
                f"(max abs diff {float(np.max(np.abs(a - b))):.3e})"
            )

        # Harness self-check: perturbing the PAST must change the output, so
        # that a pass above evidences causality rather than a no-op function.
        lo = max(0, cutoff - 13)
        past = y[:, lo:cutoff].copy()
        y[:, lo:cutoff] += 37.0
        c = make_features(y, obs, cutoff)
        y[:, lo:cutoff] = past
        assert not np.array_equal(a, c), (
            f"harness is vacuous at cutoff {cutoff}: a past perturbation changed nothing"
        )


def test_seasonal_forecast_ignores_future(panel):
    ds, y, woy = panel

    for cutoff in (60, 100, 150, 157):
        keep = y[:, cutoff:].copy()
        a = seasonal_forecast(y, woy, cutoff)
        for i in range(2):
            future = np.random.default_rng(cutoff + i).uniform(
                0.0, 90.0, size=(y.shape[0], y.shape[1] - cutoff)
            )
            if i == 1:
                future = np.full_like(future, 1.0)
            y[:, cutoff:] = future
            b = seasonal_forecast(y, woy, cutoff)
            y[:, cutoff:] = keep
            assert np.array_equal(a, b), (
                f"seasonal forecast leaked past cutoff {cutoff} "
                f"(max abs diff {float(np.max(np.abs(a - b))):.3e})"
            )


def _synthetic_panel(n_series: int = 8, weeks: int = 160, seed: int = 7):
    """Small deterministic panel with seasonality, scale and stockouts.

    Kept deliberately tiny so one model fit plus a handful of feature passes
    stays well under a second; the real panel's feature causality is covered by
    ``test_features_ignore_columns_at_or_after_cutoff`` above.
    """
    rng = np.random.default_rng(seed)
    level = rng.uniform(5.0, 60.0, size=(n_series, 1))
    season = 1.0 + 0.35 * np.sin(2 * np.pi * np.arange(weeks) / 52.0)
    y = level * season[None, :] * rng.gamma(2.0, 0.5, size=(n_series, weeks))
    in_stock = rng.random((n_series, weeks)) > 0.12
    return np.where(in_stock, y, np.nan), in_stock


def test_lgbm_forecast_ignores_future():
    y, in_stock = _synthetic_panel()
    # One fit serves every cutoff below; all cutoffs are at or after the fit
    # origin, so the model itself never trains past the decision week.
    origin = 140
    models = fit_lgbm(y, in_stock, origin)

    for cutoff in (origin, 145, 150):
        keep = y[:, cutoff:].copy()
        keep_obs = in_stock[:, cutoff:].copy()
        # The mutation cutoff IS the decision time the forecast is called with.
        a = lgbm_forecast(y, in_stock, cutoff, models)
        for i in range(2):
            future = np.random.default_rng(cutoff + i).uniform(
                0.0, 90.0, size=(y.shape[0], y.shape[1] - cutoff)
            )
            if i == 1:
                future = np.full_like(future, 1.0)
            y[:, cutoff:] = future
            in_stock[:, cutoff:] = np.isfinite(future)
            b = lgbm_forecast(y, in_stock, cutoff, models)
            y[:, cutoff:] = keep
            in_stock[:, cutoff:] = keep_obs
            assert np.array_equal(a, b), (
                f"LightGBM forecast leaked past cutoff {cutoff} "
                f"(max abs diff {float(np.max(np.abs(a - b))):.3e})"
            )

        # Harness self-check: perturbing the PAST must change the output, so a
        # pass above evidences causality rather than a constant model.
        lo = max(0, cutoff - 13)
        past = y[:, lo:cutoff].copy()
        y[:, lo:cutoff] += 37.0
        c = lgbm_forecast(y, in_stock, cutoff, models)
        y[:, lo:cutoff] = past
        assert not np.array_equal(a, c), (
            f"harness is vacuous at cutoff {cutoff}: a past perturbation changed nothing"
        )


def test_forecast_shapes(panel):
    ds, y, woy = panel
    fc = seasonal_forecast(y, woy, 150)
    assert fc.shape == (599, 3)
    assert np.isfinite(fc).all()
    assert (fc >= 0).all()


def _woy(weeks: int) -> np.ndarray:
    """Deterministic week-of-year calendar for the small panel."""
    return np.arange(weeks) % 52 + 1


def _small_panel(n_series: int = 8, weeks: int = 32, seed: int = 7):
    """Small raw panel: (finite sales, in-stock mask) for mutation gates.

    Returns raw sales so callers can build the censored panel themselves with
    ``np.where(s, y, np.nan)`` (as the brief's quantile test does). The
    generator mirrors ``_synthetic_panel`` exactly except it does not
    pre-censor, which makes the in-place ``+= 1e6`` future perturbation
    visible in every future column instead of leaving NaNs as NaNs.
    """
    rng = np.random.default_rng(seed)
    level = rng.uniform(5.0, 60.0, size=(n_series, 1))
    season = 1.0 + 0.35 * np.sin(2 * np.pi * np.arange(weeks) / 52.0)
    y = level * season[None, :] * rng.gamma(2.0, 0.5, size=(n_series, weeks))
    in_stock = rng.random((n_series, weeks)) > 0.12
    return y, in_stock


def test_quantile_path_ignore_future():
    y, s = _small_panel()
    a = forecast.seasonal_forecast(np.where(s, y, np.nan), _woy(20), 12)
    y[:, 12:] += 1e6
    b = forecast.seasonal_forecast(np.where(s, y, np.nan), _woy(20), 12)
    assert np.array_equal(a, b)


def _perturb_future(y: np.ndarray, cutoff: int, kind: int) -> None:
    """Mutate columns [cutoff:] in place (same allocation), two ways."""
    if kind == 0:
        y[:, cutoff:] += 1e6
    else:
        rng = np.random.default_rng(cutoff + kind)
        y[:, cutoff:] = rng.uniform(0.0, 90.0, size=(y.shape[0], y.shape[1] - cutoff))


@pytest.mark.parametrize("cutoff", (12, 16, 20))
@pytest.mark.parametrize("kind", (0, 1))
def test_make_features_ignore_future_small_panel(cutoff: int, kind: int):
    y, s = _small_panel()
    keep = y[:, cutoff:].copy()
    keep_obs = s[:, cutoff:].copy()
    a = make_features(y, s, cutoff)
    _perturb_future(y, cutoff, kind)
    if kind == 1:
        s[:, cutoff:] = np.isfinite(y[:, cutoff:])
    b = make_features(y, s, cutoff)
    y[:, cutoff:] = keep
    s[:, cutoff:] = keep_obs
    assert np.array_equal(a, b), (
        f"features leaked past cutoff {cutoff} (kind {kind}) "
        f"(max abs diff {float(np.max(np.abs(a - b))):.3e})"
    )


@pytest.mark.parametrize("cutoff", (12, 16, 20))
@pytest.mark.parametrize("kind", (0, 1))
def test_seasonal_forecast_ignore_future_small_panel(cutoff: int, kind: int):
    y, s = _small_panel()
    yc = np.where(s, y, np.nan)
    keep = yc[:, cutoff:].copy()
    a = seasonal_forecast(yc, _woy(yc.shape[1]), cutoff)
    _perturb_future(yc, cutoff, kind)
    b = seasonal_forecast(yc, _woy(yc.shape[1]), cutoff)
    yc[:, cutoff:] = keep
    assert np.array_equal(a, b), (
        f"seasonal leaked past cutoff {cutoff} (kind {kind}) "
        f"(max abs diff {float(np.max(np.abs(a - b))):.3e})"
    )


@pytest.mark.parametrize("upto", (12, 16, 20))
@pytest.mark.parametrize("kind", (0, 1))
def test_impute_panel_ignore_future(upto: int, kind: int):
    y, s = _small_panel()
    keep = y[:, upto:].copy()
    keep_obs = s[:, upto:].copy()
    a = impute_panel(y, s, upto)
    # In-place mutation of the same allocations: future history and mask.
    if kind == 0:
        y[:, upto:] += 1e6
    else:
        rng = np.random.default_rng(upto + kind)
        y[:, upto:] = rng.uniform(0.0, 90.0, size=(y.shape[0], y.shape[1] - upto))
    s[:, upto:] = False
    b = impute_panel(y, s, upto)
    # Only the observable window is comparable: columns [upto:] pass through.
    assert np.array_equal(a[:, :upto], b[:, :upto]), (
        f"impute leaked past upto {upto} (kind {kind})"
    )
    np.testing.assert_array_equal(b[:, upto:], y[:, upto:])
    y[:, upto:] = keep
    s[:, upto:] = keep_obs


@pytest.mark.parametrize("cutoff", (12, 16, 20))
@pytest.mark.parametrize("kind", (0, 1))
def test_global_season_ignore_future(cutoff: int, kind: int):
    y, s = _small_panel()
    yc = np.where(s, y, np.nan)
    woy = _woy(yc.shape[1])
    keep = yc[:, cutoff:].copy()
    a = global_season(yc, woy, cutoff)
    _perturb_future(yc, cutoff, kind)
    b = global_season(yc, woy, cutoff)
    yc[:, cutoff:] = keep
    assert np.array_equal(a, b), (
        f"global_season leaked past cutoff {cutoff} (kind {kind}) "
        f"(max abs diff {float(np.max(np.abs(a - b))):.3e})"
    )


def test_small_panel_harness_not_vacuous():
    """Harness self-check: perturbing the PAST must change every output."""
    cutoff = 16
    y, s = _small_panel()
    yc = np.where(s, y, np.nan)
    woy = _woy(yc.shape[1])

    feats = make_features(y, s, cutoff)
    seasonal = seasonal_forecast(yc, woy, cutoff)
    season = global_season(yc, woy, cutoff)
    imputed = impute_panel(y, s, cutoff)

    lo = max(0, cutoff - 4)
    y_past = y[:, lo:cutoff].copy()
    yc_past = yc[:, lo:cutoff].copy()
    y[:, lo:cutoff] += 37.0
    yc[:, lo:cutoff] += 37.0
    assert not np.array_equal(feats, make_features(y, s, cutoff))
    assert not np.array_equal(seasonal, seasonal_forecast(yc, woy, cutoff))
    assert not np.array_equal(season, global_season(yc, woy, cutoff))
    assert not np.array_equal(
        imputed[:, :cutoff], impute_panel(y, s, cutoff)[:, :cutoff]
    )
    y[:, lo:cutoff] = y_past
    yc[:, lo:cutoff] = yc_past
