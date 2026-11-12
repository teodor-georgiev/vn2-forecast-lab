"""Chronological, non-overlapping cross-validation.

Eight disjoint 8-week windows start at `config.CV_ORIGINS`. Each is evaluated
from three synthetic opening states under three demand levels, and only weeks
3..8 of the window are scored — mirroring the game exactly.

Two panels are kept deliberately separate:

* ``y_raw``   observed sales, zeros included. This is what the simulator is fed,
  because that is how the game behaves: an out-of-stock week still records the
  sales that were actually made.
* ``y_cens``  the same panel with out-of-stock weeks set to missing. This is
  what the forecasters are fed, because a zero recorded during a stockout is
  censored demand, not evidence of no demand.

Feeding the censored panel to the simulator would inject NaN into the stock
flow; feeding the raw panel to the forecasters would train them to under-predict.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import (
    CV_INITIAL_STATE_SEEDS,
    CV_ORIGINS,
    CV_WINDOW,
    DEMAND_STRESS,
    ROBUST_WEIGHT_STD,
    ROBUST_WEIGHT_WORST,
)
from .forecast import blend, fit_lgbm, lgbm_forecast, seasonal_forecast
from .organizer import nanmean_rows
from .policy import deficit_order
from .simulator import simulate

N_DECISIONS = 6


def synthetic_init(y_cens: np.ndarray, origin: int):
    """Plausible opening state derived from history only: ~1 week in each slot."""
    q = np.round(
        np.nan_to_num(nanmean_rows(y_cens[:, max(0, origin - 13) : origin]), nan=0.0)
    )
    return q.copy(), q.copy(), q.copy()


def rollout(window_demand, on_hand0, p1, p2, forecast_fn, coverage, buffer_coef):
    """Six decisions rolled forward with the real information set."""
    n = window_demand.shape[1]
    orders = np.zeros((N_DECISIONS, n), dtype=np.float64)
    oh = np.asarray(on_hand0, dtype=np.float64).copy()
    for d in range(1, N_DECISIONS + 1):
        t = d - 1
        fc = forecast_fn(t)
        if d == 1:
            a_now, a_next = p1, p2
        elif d == 2:
            a_now, a_next = p2, orders[0]
        else:
            a_now, a_next = orders[d - 3], orders[d - 2]
        orders[t] = deficit_order(fc, oh, a_now, a_next, coverage, buffer_coef)
        arr = p1 if t == 0 else (p2 if t == 1 else orders[t - 2])
        oh = np.maximum(0.0, oh + arr - window_demand[t])
    return orders


def _cum3(per_week: np.ndarray) -> np.ndarray:
    return np.cumsum(np.maximum(per_week, 0.0), axis=1)


@dataclass
class CVReport:
    robust: float
    mean: float
    std: float
    worst: float
    per_fold: list = field(default_factory=list)
    wape: float = float("nan")
    bias: float = float("nan")

    def as_dict(self) -> dict:
        return {
            "robust": self.robust,
            "mean": self.mean,
            "std": self.std,
            "worst": self.worst,
            "wape": self.wape,
            "bias": self.bias,
            "per_fold": self.per_fold,
        }


def run_cv(
    y_raw: np.ndarray,
    y_cens: np.ndarray,
    woy: np.ndarray,
    seasonal_weight: float = 0.6,
    coverage: float = 1.0,
    buffer_coef: float = 0.0,
    use_lgbm: bool = True,
) -> CVReport:
    """Evaluate one (forecast blend, policy) configuration on the eight folds."""
    observed = np.isfinite(y_cens)
    fold_costs: list[float] = []
    per_fold: list[dict] = []
    wape_num = wape_den = bias_num = 0.0

    for origin in CV_ORIGINS:
        models = fit_lgbm(y_cens, observed, origin) if use_lgbm else None

        def forecast_fn(t, origin=origin, models=models):
            cutoff = origin + t
            sfc = _cum3(seasonal_forecast(y_cens, woy, cutoff))
            if models is None:
                return sfc
            lfc = lgbm_forecast(y_cens, observed, cutoff, models)
            return blend(sfc, lfc, seasonal_weight)

        fc0 = forecast_fn(0)
        scored_actual = np.nansum(
            np.maximum(y_cens[:, origin + 2 : origin + CV_WINDOW], 0.0), axis=1
        )
        pred = np.maximum(fc0[:, 2], 0.0)
        wape_num += float(np.abs(pred - scored_actual).sum())
        wape_den += float(np.abs(scored_actual).sum())
        bias_num += float((pred - scored_actual).sum())

        costs = []
        for seed in CV_INITIAL_STATE_SEEDS:
            rng = np.random.default_rng(seed * 1000 + origin)
            on_hand, p1, p2 = synthetic_init(y_cens, origin)
            if seed > 0:
                on_hand = on_hand * rng.uniform(0.6, 1.4, size=on_hand.shape)
                p1 = p1 * rng.uniform(0.6, 1.4, size=p1.shape)
                p2 = p2 * rng.uniform(0.6, 1.4, size=p2.shape)
            for stress in DEMAND_STRESS:
                # Panels are (series, weeks); the simulator and rollout want
                # (weeks, series), so transpose at the boundary.
                demand = (
                    np.maximum(y_raw[:, origin : origin + CV_WINDOW], 0.0)
                    * (1.0 + stress)
                ).T
                orders = rollout(
                    demand, on_hand, p1, p2, forecast_fn, coverage, buffer_coef
                )
                costs.append(simulate(orders, demand, on_hand, p1, p2).scored_cost)

        mean_cost = float(np.mean(costs))
        fold_costs.append(mean_cost)
        per_fold.append(
            {
                "origin": int(origin),
                "cost": round(mean_cost, 2),
                "min": round(float(np.min(costs)), 2),
                "max": round(float(np.max(costs)), 2),
            }
        )

    arr = np.asarray(fold_costs, dtype=np.float64)
    mean, std, worst = float(arr.mean()), float(arr.std()), float(arr.max())
    robust = mean + ROBUST_WEIGHT_STD * std + ROBUST_WEIGHT_WORST * (worst - mean)
    return CVReport(
        robust=robust,
        mean=mean,
        std=std,
        worst=worst,
        per_fold=per_fold,
        wape=100.0 * wape_num / wape_den if wape_den else float("nan"),
        bias=100.0 * bias_num / wape_den if wape_den else float("nan"),
    )
