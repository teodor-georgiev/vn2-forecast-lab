"""Simulator correctness: independent reconciliation and the game invariants."""

import numpy as np
import pytest

from vn2_from_scratch.config import (
    ANCHOR_ORGANIZER_BASELINE_WEEKS_3_8,
    ANCHOR_POLICY_INDEPENDENT_WEEK1,
    ANCHOR_POLICY_INDEPENDENT_WEEK2,
)
from vn2_from_scratch.data import load_dataset
from vn2_from_scratch.organizer import run_organizer_benchmark
from vn2_from_scratch.simulator import simulate
from vn2_from_scratch.verify import (
    closed_form_no_shortage,
    independent_simulate,
    randomized_crosscheck,
)


def test_reconciles_with_independent_implementation():
    assert randomized_crosscheck(simulate, n=25, seed=0) < 1e-9


def test_closed_form_when_no_stockouts():
    rng = np.random.default_rng(7)
    n = 40
    on_hand = rng.integers(200, 400, size=n).astype(float)
    p1 = rng.integers(0, 50, size=n).astype(float)
    p2 = rng.integers(0, 50, size=n).astype(float)
    orders = rng.integers(50, 200, size=(6, n)).astype(float)
    demand = rng.integers(0, 20, size=(8, n)).astype(float)
    res = simulate(orders, demand, on_hand, p1, p2)
    assert res.shortage_cost == 0.0
    landed = np.zeros((8, n))
    landed[0], landed[1] = p1, p2
    for r in range(6):
        landed[r + 2] += orders[r]
    expected = on_hand + np.cumsum(landed - demand, axis=0)
    measured_holding = res.holding_cost / 0.20
    assert measured_holding == pytest.approx(expected.sum(), rel=1e-9)


def test_scored_weeks_are_three_to_eight():
    rng = np.random.default_rng(3)
    res = simulate(
        rng.integers(0, 10, size=(6, 20)).astype(float),
        rng.integers(0, 10, size=(8, 20)).astype(float),
        rng.integers(0, 10, size=20).astype(float),
        rng.integers(0, 10, size=20).astype(float),
        rng.integers(0, 10, size=20).astype(float),
    )
    assert res.scored_cost == pytest.approx(res.weekly_total[2:].sum(), rel=1e-12)


def test_weeks_one_and_two_ignore_orders():
    ds = load_dataset()
    rng = np.random.default_rng(11)
    base = simulate(
        np.zeros((6, ds.n)), ds.game_sales.T, ds.on_hand0, ds.pipe_w1, ds.pipe_w2
    )
    assert base.weekly_total[0] == pytest.approx(
        ANCHOR_POLICY_INDEPENDENT_WEEK1, abs=1e-9
    )
    assert base.weekly_total[1] == pytest.approx(
        ANCHOR_POLICY_INDEPENDENT_WEEK2, abs=1e-9
    )
    for _ in range(4):
        other = simulate(
            rng.integers(0, 300, size=(6, ds.n)).astype(float),
            ds.game_sales.T,
            ds.on_hand0,
            ds.pipe_w1,
            ds.pipe_w2,
        )
        assert other.weekly_total[0] == pytest.approx(base.weekly_total[0], abs=1e-9)
        assert other.weekly_total[1] == pytest.approx(base.weekly_total[1], abs=1e-9)


def test_organizer_baseline_reproduces_published_number():
    ds = load_dataset()
    _, res = run_organizer_benchmark(ds)
    assert res.scored_cost == pytest.approx(
        ANCHOR_ORGANIZER_BASELINE_WEEKS_3_8, abs=1e-6
    )


def test_gates_all_pass():
    import subprocess
    import sys
    from pathlib import Path

    from vn2_from_scratch.config import verify_data_hashes

    assert all(verify_data_hashes().values())
    gate = Path(__file__).resolve().parents[1] / "scripts" / "gate_all.py"
    assert gate.is_file(), "gate_all.py not found"
    proc = subprocess.run(
        [sys.executable, str(gate)], capture_output=True, text=True, timeout=600
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_no_negative_stock_or_orders():
    rng = np.random.default_rng(5)
    res = simulate(
        rng.integers(0, 5, size=(6, 30)).astype(float),
        rng.integers(0, 100, size=(8, 30)).astype(float),
        np.zeros(30),
        np.zeros(30),
        np.zeros(30),
    )
    assert res.holding_cost >= 0 and res.shortage_cost >= 0
    assert 0.0 <= res.fill_rate_scored <= 1.0
