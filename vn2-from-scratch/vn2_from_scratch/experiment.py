"""End-to-end experiment runner: CV sweep, frozen choice, retrospective replay.

Order of operations matters and is fixed here:

1. Fit each fold's forecast model ONCE per origin, then cache every decision's
   forecast. The policy sweep reuses that cache, so tuning `coverage` or the
   buffer can never silently refit a model and contaminate the comparison.
2. Select by CV only, using the robust statistic in config.
3. Freeze the selection into a receipt.
4. Fit on the full history, produce the six real rounds, and replay the game.

The replay scores the published game weeks 1-8. Those labels were public
before this project existed, so every number it produces is a
**retrospective replay**, never a blind result and never an official score.
"""

from __future__ import annotations

import itertools
import json
import time
from pathlib import Path

import numpy as np

from .config import (
    ARTIFACTS_DIR,
    CV_INITIAL_STATE_SEEDS,
    CV_ORIGINS,
    CV_WINDOW,
    DEMAND_STRESS,
    HISTORY_WEEKS,
    N_DECISIONS,
    REPORTS_DIR,
    ROBUST_WEIGHT_STD,
    ROBUST_WEIGHT_WORST,
    SEED,
    sha256_file,
    DATA_DIR,
)
from .cv import _cum3, rollout
from .data import load_dataset
from .forecast import blend, fit_lgbm, lgbm_forecast, seasonal_forecast
from .organizer import full_panel, nanmean_rows, run_organizer_benchmark
from .policy import deficit_order
from .simulator import simulate
from .submit import write_rounds

VALIDITY = "retrospective competition replay"


def cache_forecasts(y_cens, woy, seasonal_weight, origins, use_lgbm=True):
    """Fit once per origin, then cache the forecast seen at each decision."""
    observed = np.isfinite(y_cens)
    cache: dict[int, dict[int, np.ndarray]] = {}
    for origin in origins:
        models = fit_lgbm(y_cens, observed, origin) if use_lgbm else None
        per_t = {}
        for t in range(N_DECISIONS):
            cutoff = origin + t
            sfc = _cum3(seasonal_forecast(y_cens, woy, cutoff))
            per_t[t] = (
                sfc
                if models is None
                else blend(
                    sfc,
                    lgbm_forecast(y_cens, observed, cutoff, models),
                    seasonal_weight,
                )
            )
        cache[origin] = per_t
    return cache


def eval_policy(raw, cens, cache, coverage, buffer_coef, stress_grid=DEMAND_STRESS):
    """Score one policy configuration over the cached folds."""
    fold_costs = []
    fold_detail = []
    for origin in CV_ORIGINS:
        fns = cache[origin]
        costs = []
        for seed in CV_INITIAL_STATE_SEEDS:
            rng = np.random.default_rng(seed * 1000 + origin)
            base = np.round(
                np.nan_to_num(nanmean_rows(cens[:, max(0, origin - 13) : origin]))
            )
            on_hand, p1, p2 = base.copy(), base.copy(), base.copy()
            if seed > 0:
                on_hand = on_hand * rng.uniform(0.6, 1.4, size=on_hand.shape)
                p1 = p1 * rng.uniform(0.6, 1.4, size=p1.shape)
                p2 = p2 * rng.uniform(0.6, 1.4, size=p2.shape)
            for stress in stress_grid:
                # Panels are (series, weeks); rollout/simulate want (weeks, series).
                demand = (
                    np.maximum(raw[:, origin : origin + CV_WINDOW], 0.0)
                    * (1.0 + stress)
                ).T
                orders = rollout(
                    demand, on_hand, p1, p2, lambda t: fns[t], coverage, buffer_coef
                )
                costs.append(simulate(orders, demand, on_hand, p1, p2).scored_cost)
        fold_costs.append(float(np.mean(costs)))
        fold_detail.append(
            {
                "origin": int(origin),
                "cost": round(float(np.mean(costs)), 2),
                "min": round(float(np.min(costs)), 2),
                "max": round(float(np.max(costs)), 2),
            }
        )
    arr = np.asarray(fold_costs)
    mean, std, worst = float(arr.mean()), float(arr.std()), float(arr.max())
    return {
        "coverage": coverage,
        "buffer_coef": buffer_coef,
        "mean": round(mean, 2),
        "std": round(std, 2),
        "worst": round(worst, 2),
        "robust": round(
            mean + ROBUST_WEIGHT_STD * std + ROBUST_WEIGHT_WORST * (worst - mean),
            4,
        ),
        # Same shape as cv.run_cv's per_fold, so the two CV paths agree.
        "per_fold": fold_detail,
    }


def game_orders(y_cens, woy, seasonal_weight, coverage, buffer_coef):
    """Fit on all 157 history weeks and decide the six real rounds."""
    observed = np.isfinite(y_cens)
    models = fit_lgbm(y_cens, observed, HISTORY_WEEKS)
    ds_cache = {}
    for t in range(N_DECISIONS):
        cutoff = HISTORY_WEEKS + t
        sfc = _cum3(seasonal_forecast(y_cens, woy, cutoff))
        ds_cache[t] = blend(
            sfc, lgbm_forecast(y_cens, observed, cutoff, models), seasonal_weight
        )

    ds = load_dataset()
    orders = np.zeros((N_DECISIONS, ds.n))
    oh = ds.on_hand0.copy()
    for d in range(1, N_DECISIONS + 1):
        t = d - 1
        if d == 1:
            a_now, a_next = ds.pipe_w1, ds.pipe_w2
        elif d == 2:
            a_now, a_next = ds.pipe_w2, orders[0]
        else:
            a_now, a_next = orders[d - 3], orders[d - 2]
        orders[t] = deficit_order(ds_cache[t], oh, a_now, a_next, coverage, buffer_coef)
        arr = ds.pipe_w1 if t == 0 else (ds.pipe_w2 if t == 1 else orders[t - 2])
        oh = np.maximum(0.0, oh + arr - ds.game_sales[:, t])
    return orders, ds


def main(out_dir: str | Path | None = None, seasonal_weights=(1.0, 0.7, 0.4)):
    out_dir = Path(out_dir or ARTIFACTS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    ds = load_dataset()
    y_raw = np.hstack([ds.history, ds.game_sales])
    y_cens, woy, _ = full_panel(ds)

    _, bench = run_organizer_benchmark(ds)

    rows = []
    for sw in seasonal_weights:
        cache = cache_forecasts(y_cens, woy, sw, CV_ORIGINS)
        for cov, buf in itertools.product(
            (0.9, 1.0, 1.1, 1.2, 1.3), (0.0, 0.15, 0.3, 0.45)
        ):
            rows.append(
                eval_policy(y_raw, y_cens, cache, cov, buf) | {"seasonal_weight": sw}
            )

    rows.sort(key=lambda r: r["robust"])
    best = rows[0]

    frozen = {
        "selected_on": "CV only (8 disjoint origins, 3 synthetic starts, +0/+10/+20% demand stress)",
        "best": best,
        "n_candidates": len(rows),
        "top5": rows[:5],
        # Every evaluated point, so the selection is auditable rather than asserted.
        "all_points": rows,
        "seasonal_weight": best["seasonal_weight"],
        "coverage": best["coverage"],
        "buffer_coef": best["buffer_coef"],
        "frozen_at_unix": time.time(),
        # Seeds that actually drove the CV draws above (synthetic start states).
        "cv_initial_state_seeds": list(CV_INITIAL_STATE_SEEDS),
        # Provenance only: SEED drives no RNG in this pipeline.
        "provenance_seed": SEED,
        "data_sha256": {p.name: sha256_file(p) for p in sorted(DATA_DIR.glob("*.csv"))},
    }
    (out_dir / "frozen_config.json").write_text(json.dumps(frozen, indent=2))

    orders, ds = game_orders(
        y_cens, woy, best["seasonal_weight"], best["coverage"], best["buffer_coef"]
    )
    res = simulate(orders, ds.game_sales.T, ds.on_hand0, ds.pipe_w1, ds.pipe_w2)
    paths = write_rounds(orders, Path(out_dir).parent / "submissions")

    replay = {
        "validity": VALIDITY,
        "warning": "Game weeks 1-8 labels were published before this project. NOT blind, NOT official.",
        "total_cost": round(res.total_cost, 2),
        "scored_cost": round(res.scored_cost, 2),
        "scored_holding": round(res.scored_holding, 2),
        "scored_shortage": round(res.scored_shortage, 2),
        "fill_rate_scored": round(res.fill_rate_scored, 4),
        "weekly_total": [round(x, 2) for x in res.weekly_total.tolist()],
        "organizer_benchmark_scored": round(bench.scored_cost, 2),
        "submission_files": [str(p) for p in paths],
        "elapsed_s": round(time.time() - t0, 1),
    }
    (out_dir / "retrospective_replay.json").write_text(json.dumps(replay, indent=2))

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    md = [
        "# VN2 from-scratch — results",
        "",
        f"**Validity: {VALIDITY}.** The game weeks 1-8 labels were published before",
        "this project existed, so nothing below is blind or official.",
        "",
        "| metric | value |",
        "| --- | ---: |",
        f"| Organizer benchmark, weeks 3-8 | EUR {bench.scored_cost:,.2f} |",
        f"| **This solution, weeks 3-8** | **EUR {res.scored_cost:,.2f}** |",
        f"| &nbsp;&nbsp;holding | EUR {res.scored_holding:,.2f} |",
        f"| &nbsp;&nbsp;shortage | EUR {res.scored_shortage:,.2f} |",
        f"| Fill rate (scored weeks) | {100 * res.fill_rate_scored:.2f}% |",
        "",
        "## Selected configuration (CV only)",
        "",
        f"- seasonal weight **{best['seasonal_weight']}**, coverage **{best['coverage']}**, "
        f"buffer **{best['buffer_coef']}**",
        f"- robust CV cost **{best['robust']}** (mean {best['mean']}, worst {best['worst']}) over "
        f"{len(rows)} candidates",
        "",
        "## Per-fold CV cost",
        "",
        "| origin | cost | min | max |",
        "| --- | ---: | ---: | ---: |",
    ]
    fold_rows = [
        f"| {f['origin']} | {f['cost']:,.2f} | {f['min']:,.2f} | {f['max']:,.2f} |"
        for f in best["per_fold"]
    ]
    md.extend(fold_rows)
    md.extend(
        [
            "",
            "Fold costs are measured on a harsher regime than the replay: synthetic",
            "opening states and +0/+10/+20% demand stress. They are therefore not",
            f"comparable in level to the EUR {res.scored_cost:,.2f} figure, which uses the real opening",
            "state and realised demand. The fold costs select the configuration; they",
            "do not estimate the replay score.",
            "",
            "## Forecast quality vs inventory cost",
            "",
            "Forecast error and inventory cost are reported separately and must not be",
            "conflated: a lower WAPE does not imply a lower cost, because cost depends",
            "on the holding/shortage trade-off at 0.20/1.00 EUR per unit, not on",
            "forecast accuracy alone.",
        ]
    )
    (REPORTS_DIR / "RESULTS.md").write_text("\n".join(md) + "\n")
    return frozen, replay


if __name__ == "__main__":
    main()
