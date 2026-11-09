"""Reporting: CV summary dict must separate forecast quality from inventory cost."""

import numpy as np
import pytest

from vn2_from_scratch.config import ROBUST_WEIGHT_STD, ROBUST_WEIGHT_WORST
from vn2_from_scratch.cv import CVReport
from vn2_from_scratch.reporting import report_cv
from vn2_from_scratch.simulator import simulate


def test_report_cv_keys():
    r = report_cv(
        CVReport(robust=1.0, mean=1.0, std=0.1, worst=2.0, wape=10.0, bias=-5.0)
    )
    assert r["robust"] == pytest.approx(1.0 + 0.45 * 0.1 + 0.15 * (2.0 - 1.0))
    assert r["robust"] == pytest.approx(
        1.0 + ROBUST_WEIGHT_STD * 0.1 + ROBUST_WEIGHT_WORST * (2.0 - 1.0)
    )
    assert {"robust", "wape", "bias", "scored_holding", "scored_shortage"} <= set(r)
    assert {"holding", "shortage"} <= set(r) | {"mean", "std", "worst", "per_fold"}


def test_report_cv_keeps_wape_bias_separate():
    r = report_cv(
        CVReport(robust=0.0, mean=10.0, std=2.0, worst=12.0, wape=7.5, bias=-3.0)
    )
    assert r["wape"] == pytest.approx(7.5)
    assert r["bias"] == pytest.approx(-3.0)
    assert r["mean"] == pytest.approx(10.0)


def test_report_cv_recomputes_robust_from_mean_std_worst():
    # Stored robust is ignored: the formula mean+0.45*std+0.15*(worst-mean) governs.
    r = report_cv(
        CVReport(robust=999.0, mean=100.0, std=10.0, worst=120.0, wape=1.0, bias=0.0)
    )
    assert r["robust"] == pytest.approx(100.0 + 0.45 * 10.0 + 0.15 * (120.0 - 100.0))


def test_report_cv_splits_holding_shortage_from_sim_result():
    rng = np.random.default_rng(3)
    res = simulate(
        rng.integers(0, 10, size=(6, 20)).astype(float),
        rng.integers(0, 10, size=(8, 20)).astype(float),
        rng.integers(0, 10, size=20).astype(float),
        rng.integers(0, 10, size=20).astype(float),
        rng.integers(0, 10, size=20).astype(float),
    )
    r = report_cv(
        CVReport(robust=0.0, mean=100.0, std=5.0, worst=110.0, wape=4.0, bias=1.0),
        res,
    )
    assert r["scored_holding"] == pytest.approx(res.scored_holding)
    assert r["scored_shortage"] == pytest.approx(res.scored_shortage)
    assert r["holding"] == pytest.approx(res.scored_holding)
    assert r["shortage"] == pytest.approx(res.scored_shortage)
    assert r["scored_holding"] >= 0 and r["scored_shortage"] >= 0


def test_report_cv_retrospective_replay_wording():
    r = report_cv(CVReport(robust=0.0, mean=1.0, std=0.0, worst=1.0))
    text = " ".join(str(v) for v in r.values()).lower()
    assert "retrospective replay" in text
    assert "never blind" in text
    assert "never official" in text
    assert r["validity"] == "retrospective competition replay"


def test_report_cv_uses_scored_weeks_three_to_eight():
    r = report_cv(CVReport(robust=0.0, mean=1.0, std=0.0, worst=1.0))
    assert r["scored_weeks"] == [3, 4, 5, 6, 7, 8]


def test_report_cv_rejects_negative_costs():
    with pytest.raises(ValueError, match="negative"):
        report_cv(CVReport(robust=0.0, mean=-1.0, std=0.0, worst=1.0))
    with pytest.raises(ValueError, match="negative"):
        report_cv(CVReport(robust=0.0, mean=1.0, std=-0.1, worst=1.0))


def test_gate_all_runs_pytest():
    from pathlib import Path

    gate = Path(__file__).resolve().parents[1] / "scripts" / "gate_all.py"
    text = gate.read_text(encoding="utf-8")
    assert "pytest" in text and "tests/" in text
