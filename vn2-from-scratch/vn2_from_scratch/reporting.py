"""Reporting: CV summary with forecast quality kept separate from inventory cost.

``report_cv`` consumes a ``cv.CVReport`` shape (mean, std, worst, wape,
bias, per_fold) and optionally a ``simulator.SimResult``. It returns a
plain dict with the robust CV cost (verbatim formula
``mean + 0.45*std + 0.15*(worst - mean)``), forecast quality (WAPE,
bias) reported separately from inventory cost (scored holding vs
shortage split, game weeks 3-8 only).

Replay wording: any game-week result is a **retrospective competition
replay** — the game weeks 1-8 labels were published before this project
existed, so numbers are never blind and never official.
"""

from __future__ import annotations

import math

from .config import (
    ROBUST_WEIGHT_STD,
    ROBUST_WEIGHT_WORST,
    SCORE_WEEKS_END,
    SCORE_WEEKS_START,
)

VALIDITY = "retrospective competition replay"

NOTE = (
    "Retrospective replay: game weeks 1-8 labels were published before "
    "this project existed; never blind, never official. Scored weeks "
    "3-8 only. Forecast quality (WAPE, bias) is reported separately "
    "from inventory cost (holding vs shortage), and a better WAPE "
    "does not imply a better cost."
)


def _get(obj, name, default=None):
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _cost(name: str, value):
    """Coerce a cost field to float; costs must be finite and non-negative."""
    if value is None:
        return None
    v = float(value)
    if not math.isfinite(v):
        raise ValueError(f"{name} must be finite, got {v!r}")
    if v < 0:
        raise ValueError(f"{name} must not be negative, got {v!r}")
    return v


def _forecast_metric(value):
    """Forecast metrics (WAPE, bias) keep their sign; NaN is preserved."""
    if value is None:
        return float("nan")
    return float(value)


def _first_present(*sources_and_names):
    for obj, names in sources_and_names:
        for name in names:
            v = _get(obj, name)
            if v is not None:
                return v
    return None


def _per_fold_split(per_fold):
    """Average a holding/shortage split across per_fold entries, if carried."""
    if not per_fold:
        return None, None
    holding, shortage = [], []
    for fold in per_fold:
        h = _get(fold, "scored_holding", _get(fold, "holding"))
        s = _get(fold, "scored_shortage", _get(fold, "shortage"))
        if h is not None:
            holding.append(float(h))
        if s is not None:
            shortage.append(float(s))
    h_avg = sum(holding) / len(holding) if holding else None
    s_avg = sum(shortage) / len(shortage) if shortage else None
    return h_avg, s_avg


def report_cv(cv_report, sim_result=None) -> dict:
    """Summarise a CV report (and optional SimResult) as a plain dict.

    The robust figure is recomputed verbatim from mean/std/worst
    (``mean + 0.45*std + 0.15*(worst - mean)``) rather than trusting a
    stored ``robust`` field, so stale or hand-edited values cannot leak
    through. WAPE and bias are carried through as forecast quality and
    kept separate from inventory cost. ``scored_holding`` /
    ``scored_shortage`` come from the CV report when present, else from
    ``sim_result``, else from ``per_fold`` entries that carry a split;
    when no source carries them the keys are ``None`` and the
    ``holding`` / ``shortage`` aliases mirror them.
    """
    mean = _cost("mean", _get(cv_report, "mean"))
    std = _cost("std", _get(cv_report, "std"))
    worst = _cost("worst", _get(cv_report, "worst"))
    if mean is None or std is None or worst is None:
        raise ValueError("mean, std and worst are required for report_cv")

    # Verbatim robust formula: mean + 0.45*std + 0.15*(worst - mean).
    robust = mean + ROBUST_WEIGHT_STD * std + ROBUST_WEIGHT_WORST * (worst - mean)
    if robust < 0:
        raise ValueError(f"robust must not be negative, got {robust!r}")

    per_fold = _get(cv_report, "per_fold", [])
    fold_holding, fold_shortage = _per_fold_split(per_fold)

    raw_holding = _first_present(
        (cv_report, ("scored_holding", "holding")),
        (sim_result, ("scored_holding", "holding")),
    )
    raw_shortage = _first_present(
        (cv_report, ("scored_shortage", "shortage")),
        (sim_result, ("scored_shortage", "shortage")),
    )
    if raw_holding is None and fold_holding is not None:
        raw_holding = fold_holding
    if raw_shortage is None and fold_shortage is not None:
        raw_shortage = fold_shortage
    scored_holding = _cost("scored_holding", raw_holding)
    scored_shortage = _cost("scored_shortage", raw_shortage)

    scored_cost = _cost(
        "scored_cost",
        _first_present(
            (cv_report, ("scored_cost",)),
            (sim_result, ("scored_cost",)),
        ),
    )
    if (
        scored_cost is None
        and scored_holding is not None
        and scored_shortage is not None
    ):
        scored_cost = scored_holding + scored_shortage

    scored_weeks = list(range(SCORE_WEEKS_START, SCORE_WEEKS_END + 1))

    return {
        "robust": float(robust),
        "mean": float(mean),
        "std": float(std),
        "worst": float(worst),
        "wape": _forecast_metric(_get(cv_report, "wape")),
        "bias": _forecast_metric(_get(cv_report, "bias")),
        "scored_cost": scored_cost,
        "scored_holding": scored_holding,
        "scored_shortage": scored_shortage,
        "holding": scored_holding,
        "shortage": scored_shortage,
        "scored_weeks": scored_weeks,
        "validity": VALIDITY,
        "note": NOTE,
        "warning": NOTE,
        "per_fold": per_fold,
    }


def render_summary(report: dict) -> str:
    """One-paragraph human summary with the retrospective-replay caveat."""
    fmt = lambda v: f"{v:,.2f}" if v is not None else "n/a"
    lines = [
        f"Validity: {report.get('validity', VALIDITY)} — {NOTE}",
        f"Robust CV cost (weeks 3-8): {fmt(report.get('robust'))} "
        f"(mean {fmt(report.get('mean'))}, worst {fmt(report.get('worst'))}).",
        f"Forecast quality (separate from cost): WAPE {fmt(report.get('wape'))}, "
        f"bias {fmt(report.get('bias'))}.",
        f"Inventory cost split (weeks 3-8): holding {fmt(report.get('scored_holding'))}, "
        f"shortage {fmt(report.get('scored_shortage'))}.",
    ]
    return "\n".join(lines)
