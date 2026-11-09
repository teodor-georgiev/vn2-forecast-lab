"""Monte Carlo expected-cost scoring over scenario packs.

Covers the Task 6 brief: ``mc_score(orders, scenarios, on_hand0, pipe_w1,
pipe_w2)`` is the mean ``scored_cost`` over scenarios, each scored with
the existing ``simulator.simulate()`` (weeks 3-8, holding EUR 0.20 on ending
inventory, shortage EUR 1.00 per lost unit). ``best_of`` iterates
candidates and returns the minimum. Scenarios are full 8-week demand
panels of shape (n_scenarios, n_series, 8) -- the horizon expansion of
SIP-style ``scenarios.pack_scenarios`` packs, which emit 3-horizon draws.
"""

import numpy as np
import pytest

from vn2_from_scratch.mc_policy import best_of, mc_score
from vn2_from_scratch.simulator import simulate


def test_mc_zero_vs_huge_orders():
    sc = np.ones((5, 4, 8)) * 5
    lo = mc_score(np.zeros((6, 4)), sc, np.zeros(4), np.zeros(4), np.zeros(4))
    hi = mc_score(np.full((6, 4), 1e4), sc, np.zeros(4), np.zeros(4), np.zeros(4))
    assert np.isfinite(lo) and np.isfinite(hi) and hi > lo


def test_mc_score_is_mean_scored_cost():
    rng = np.random.default_rng(0)
    orders = rng.uniform(0.0, 20.0, size=(6, 5))
    sc = rng.uniform(0.0, 10.0, size=(7, 5, 8))
    on_hand0 = rng.uniform(0.0, 5.0, size=5)
    pipe_w1 = rng.uniform(0.0, 2.0, size=5)
    pipe_w2 = rng.uniform(0.0, 2.0, size=5)

    got = mc_score(orders, sc, on_hand0, pipe_w1, pipe_w2)

    expected = float(
        np.mean(
            [
                simulate(orders, sc[s].T, on_hand0, pipe_w1, pipe_w2).scored_cost
                for s in range(sc.shape[0])
            ]
        )
    )
    assert got == pytest.approx(expected, rel=1e-12)
    assert isinstance(got, float) and np.isfinite(got)


def test_mc_score_does_not_mutate_inputs():
    orders = np.ones((6, 3))
    sc = np.ones((4, 3, 8)) * 2.0
    on_hand0 = np.ones(3)
    pipe_w1 = np.ones(3)
    pipe_w2 = np.ones(3)
    snapshot = [a.copy() for a in (orders, sc, on_hand0, pipe_w1, pipe_w2)]

    mc_score(orders, sc, on_hand0, pipe_w1, pipe_w2)

    for a, before in zip((orders, sc, on_hand0, pipe_w1, pipe_w2), snapshot):
        np.testing.assert_array_equal(a, before)


def test_mc_score_invalid_scenarios_raise():
    orders = np.zeros((6, 4))
    zeros = np.zeros(4)

    with pytest.raises(ValueError):  # no scenarios: mean undefined
        mc_score(orders, np.zeros((0, 4, 8)), zeros, zeros, zeros)
    with pytest.raises(ValueError):  # 2-D demand panel, not 3-D scenarios
        mc_score(orders, np.ones((4, 8)), zeros, zeros, zeros)
    with pytest.raises(ValueError):  # wrong horizon count
        mc_score(orders, np.ones((3, 4, 3)), zeros, zeros, zeros)
    with pytest.raises(ValueError):  # series count disagrees with orders
        mc_score(orders, np.ones((3, 7, 8)), zeros, zeros, zeros)


def test_best_of_returns_min_candidate():
    sc = np.ones((5, 4, 8)) * 6.0
    zeros = np.zeros(4)
    zero = np.zeros((6, 4))
    moderate = np.full((6, 4), 6.0)
    huge = np.full((6, 4), 1e4)
    candidates = [huge, zero, moderate]

    best_orders, best_score = best_of(candidates, sc, zeros, zeros, zeros)

    # Matched orders (6/week against 6 demand) cover every scored week: ~0.
    assert best_score == pytest.approx(mc_score(moderate, sc, zeros, zeros, zeros))
    np.testing.assert_array_equal(best_orders, moderate)
    # Zero orders is all shortage, huge orders is all holding: both worse.
    assert mc_score(zero, sc, zeros, zeros, zeros) > best_score
    assert mc_score(huge, sc, zeros, zeros, zeros) > best_score
    for cand in candidates:
        assert best_score <= mc_score(cand, sc, zeros, zeros, zeros)


def test_best_of_empty_raises():
    with pytest.raises(ValueError):
        best_of([], np.ones((2, 4, 8)), np.zeros(4), np.zeros(4), np.zeros(4))


def test_mc_score_pack_scenarios_integration():
    """8-week panels assembled from pack_scenarios horizon-1 draws score."""
    from vn2_from_scratch.scenarios import pack_scenarios

    # Rolling assembly: one pack per horizon-1 step, stacked into 8 weeks.
    weeks = [
        pack_scenarios(np.ones((4, 3, 2)) * (w + 1.0), 5, seed=100 + w)[:, :, 0]
        for w in range(8)
    ]
    sc = np.stack(weeks, axis=2)  # (5, 4, 8)
    assert sc.shape == (5, 4, 8)
    assert np.all(np.isfinite(sc)) and np.all(sc >= 0)

    score = mc_score(np.zeros((6, 4)), sc, np.zeros(4), np.zeros(4), np.zeros(4))
    assert np.isfinite(score)
    # Every scenario row is constant demand; mean must match simulate().
    one = simulate(np.zeros((6, 4)), sc[0].T, np.zeros(4), np.zeros(4), np.zeros(4))
    assert score == pytest.approx(one.scored_cost, rel=1e-12)
