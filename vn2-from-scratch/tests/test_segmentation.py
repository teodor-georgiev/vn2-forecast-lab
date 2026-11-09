"""ABC/XYZ segmentation: causal per-series coverage/buffer params.

``segment_params(master, history, in_stock, upto)`` classifies each series
by (a) ABC class from cumulative observed demand over ``[0:upto)``
(A = the smallest set of series covering the top 80% of total demand,
B = the next set up to 95%, C = the rest) and (b) XYZ class from the
coefficient of variation of observed weekly sales (terciles across
series: X = stable, Y = moderate, Z = volatile). Each of the nine
combinations maps to a fixed ``(coverage, buffer_coef)`` pair with
coverage bounded in ``[0.5, 4.0]``. Only columns ``[0:upto)`` are read;
the ``master`` hierarchy is read-only.
"""

import numpy as np
import pandas as pd

from vn2_from_scratch.segmentation import abc_classes, segment_params, xyz_classes


def _panel(n_series: int = 4, weeks: int = 32, seed: int = 7):
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


def _master(n_series: int = 4):
    return pd.DataFrame(
        {
            "ProductGroup": ["G"] * n_series,
            "Division": ["D"] * n_series,
            "Department": ["Dep"] * n_series,
            "DepartmentGroup": ["DG"] * n_series,
            "StoreFormat": ["F"] * n_series,
            "Format": ["Fmt"] * n_series,
        }
    )


def _params():
    return segment_params(_master(), _y(), _s(), 12)


def test_segment_params_causal_bounded():
    p = segment_params(_master(), _y(), _s(), 12)
    assert len(p) == 4 and all(0.5 <= v["coverage"] <= 4.0 for v in p.values())


def test_segment_params_schema_and_bounds():
    p = _params()
    assert sorted(p) == [0, 1, 2, 3]
    for v in p.values():
        assert set(v) == {"coverage", "buffer_coef"}
        assert 0.5 <= v["coverage"] <= 4.0
        assert 0.0 <= v["buffer_coef"] <= 2.0
        assert np.isfinite(v["coverage"]) and np.isfinite(v["buffer_coef"])


def test_segment_params_causal_post_upto_mutation():
    y, s = _panel()
    a = segment_params(_master(), y, s, 12)
    keep_y, keep_s = y[:, 12:].copy(), s[:, 12:].copy()
    y[:, 12:] += 1e6
    s[:, 12:] = False
    b = segment_params(_master(), y, s, 12)
    y[:, 12:], s[:, 12:] = keep_y, keep_s
    assert a == b, "segment_params leaked past upto"


def test_segment_params_past_perturbation_changes_output():
    """Harness self-check: outputs depend on [0:upto), not constant."""
    y, s = _panel()
    a = segment_params(_master(), y, s, 12)
    y[:, 8:12] += 37.0
    b = segment_params(_master(), y, s, 12)
    assert a != b


def test_segment_params_does_not_mutate_inputs():
    y, s = _y(), _s()
    m = _master()
    y0, s0, m0 = y.copy(), s.copy(), m.copy()
    segment_params(m, y, s, 12)
    np.testing.assert_array_equal(y, y0)
    np.testing.assert_array_equal(s, s0)
    pd.testing.assert_frame_equal(m, m0)


def test_abc_classes_top80_and_disjoint():
    """A must be the smallest set covering >=80% of cumulative demand."""
    y = np.full((5, 16), 10.0)
    y[0, :] = 100.0  # dominant series
    s = np.ones(y.shape, dtype=bool)
    cls = abc_classes(y, s, 16)
    assert list(cls) == ["A", "A", "A", "B", "B"], "A = smallest set reaching 80%"
    assert all(c in {"A", "B", "C"} for c in cls)


def test_abc_classes_respect_demand_order():
    """Higher cumulative demand never falls in a lower class."""
    rng = np.random.default_rng(3)
    y = rng.uniform(5.0, 200.0, size=(30, 20))
    s = np.ones(y.shape, dtype=bool)
    cls = list(abc_classes(y, s, 20))
    order = {"A": 0, "B": 1, "C": 2}
    cum = y.sum(axis=1)
    ranked = np.argsort(-cum)
    ranks = [order[cls[i]] for i in ranked]
    assert ranks == sorted(ranks), "ABC classes must follow demand order"


def test_xyz_classes_stable_vs_volatile():
    """Near-constant series -> X; wildly varying series -> Z."""
    weeks = 24
    y = np.vstack(
        [
            np.full(weeks, 20.0),  # stable
            np.full(weeks, 20.0) * (1.0 + 0.05 * np.sin(np.arange(weeks))),  # mild
            (np.arange(weeks) % 2) * 40.0 + 1.0,  # alternating -> volatile
            (np.arange(weeks) % 2) * 40.0 + 1.0,
        ]
    )
    s = np.ones(y.shape, dtype=bool)
    cls = xyz_classes(y, s, weeks)
    assert cls[0] == "X"
    assert cls[-1] == "Z"
    assert all(c in {"X", "Y", "Z"} for c in cls)


def test_xyz_classes_terciles_split():
    """With spread CVs, terciles split series across X/Y/Z."""
    rng = np.random.default_rng(11)
    base = rng.uniform(10.0, 50.0, size=(30, 1))
    scale = np.linspace(0.02, 0.9, 30)[:, None]
    y = base * (1.0 + scale * rng.normal(0.0, 1.0, size=(30, 24)))
    y = np.maximum(y, 0.5)
    s = np.ones(y.shape, dtype=bool)
    cls = set(xyz_classes(y, s, 24))
    assert cls == {"X", "Y", "Z"}


def test_segment_params_edge_cases():
    y, s = _panel()
    empty = segment_params(_master(), y, s, 0)
    assert len(empty) == 4 and all(0.5 <= v["coverage"] <= 4.0 for v in empty.values())
    single = segment_params(_master(1), y[:1], s[:1], 8)
    assert len(single) == 1 and 0.5 <= single[0]["coverage"] <= 4.0
    all_censored = segment_params(_master(), y, np.zeros_like(s), 12)
    assert len(all_censored) == 4 and all(
        0.5 <= v["coverage"] <= 4.0 for v in all_censored.values()
    )


def test_segment_params_real_panel():
    from vn2_from_scratch.data import load_dataset

    ds = load_dataset()
    p = segment_params(ds.master, ds.history, ds.in_stock, 140)
    assert len(p) == 599
    assert all(0.5 <= v["coverage"] <= 4.0 for v in p.values())
    assert all(np.isfinite(v["buffer_coef"]) for v in p.values())
