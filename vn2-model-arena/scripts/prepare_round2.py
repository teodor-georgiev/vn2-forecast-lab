"""Prepare round 2 (the ordering game) and its reference baselines.

Builds the starting inventory in arena units from the real VN2 initial state
(scaled by the same per-series factor as the demand), writes the contestant-facing
file, and produces reference order plans so the leaderboard has a bar.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ARENA = Path(r"C:\Users\Toni\Documents\Default Project\vn2-model-arena")
PRIVATE = Path(r"C:\Users\Toni\AppData\Local\Temp\opencode\vn2-arena-private")
sys.path.insert(0, str(ARENA / "tools"))
import inventory_sim as sim  # noqa: E402

RAW_INITIAL = Path(
    r"C:\Users\Toni\DemandAhead-VN2Game\raw\Week 0 - 2024-04-08 - Initial State.csv"
)
ORDER_WEEKS = 6


def causal_forecasts(hist: np.ndarray) -> np.ndarray:
    """13-week trailing mean per decision week, recomputed from history alone."""
    forecast = np.zeros((ORDER_WEEKS, 3, hist.shape[0]))
    for w in range(ORDER_WEEKS):
        window = hist[:, 157 + w - 13 : 157 + w]
        forecast[w] = np.repeat(window.mean(axis=1)[None, :], 3, axis=0)
    return forecast


def main() -> None:
    mapping = pd.read_csv(PRIVATE / "series_mapping.csv")
    initial_raw = pd.read_csv(RAW_INITIAL).merge(
        mapping, on=["Store", "Product"], how="inner"
    )
    assert len(initial_raw) == 599, len(initial_raw)
    initial_raw = initial_raw.sort_values("series_id").reset_index(drop=True)

    state = (
        initial_raw[["End Inventory", "In Transit W+1", "In Transit W+2"]].to_numpy(
            float
        )
        * initial_raw["scale"].to_numpy(float)[:, None]
    )
    state = np.rint(state)
    initial = pd.DataFrame(
        {
            "series_id": initial_raw["series_id"],
            "on_hand": state[:, 0],
            "arrival_w1": state[:, 1],
            "arrival_w2": state[:, 2],
        }
    )
    initial.to_csv(ARENA / "data" / "initial_state.csv", index=False)

    sales = pd.read_csv(ARENA / "data" / "sales.csv")
    pivot = sales.pivot(index="series_id", columns="week", values="y")
    pivot = pivot.loc[sorted(pivot.index)]
    hist = pivot.to_numpy(float)
    series = pivot.index.to_numpy()

    truth = pd.read_csv(PRIVATE / "answer_panel.csv")
    truth_demand = (
        truth.pivot(index="horizon", columns="series_id", values="y_true")
        .loc[range(1, sim.WEEKS + 1), series]
        .to_numpy(float)
    )
    full = np.concatenate([hist, truth_demand.T], axis=1)

    def write_plan(name: str, orders: np.ndarray) -> None:
        rows = [
            (sid, w + 1, float(orders[w, i]))
            for w in range(ORDER_WEEKS)
            for i, sid in enumerate(series)
        ]
        out = ARENA / "results" / "round2_baselines"
        out.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows, columns=["series_id", "week", "quantity"]).to_csv(
            out / f"{name}.csv", index=False
        )

    zero = np.zeros((ORDER_WEEKS, len(series)))
    write_plan("zero_orders", zero)

    # Static plan: orders chosen up front from a causal 13-week mean forecast, with the
    # planner's own projected state evolution (no feedback). This is a plan a contestant
    # can actually produce.
    mean13 = hist[:, -13:].mean(axis=1)
    state_now = state.copy()
    orders = np.zeros((ORDER_WEEKS, len(series)))
    for w in range(ORDER_WEEKS):
        forecast = np.repeat(mean13[None, :], 3, axis=0)
        order = sim.reference_orders(forecast, state_now, 1.0, 1.0, 0.75)
        orders[w] = order
        on_hand, arr1, arr2 = state_now.T
        end = np.maximum(0.0, on_hand + arr1 - mean13)
        state_now = np.column_stack((end, arr2, order))
    write_plan("deficit_static_13wmean", orders)

    # Adaptive reference with true weekly feedback (headroom reference; a static
    # submission cannot match it, and it is reported separately, never as the bar).
    forecast_adapt = causal_forecasts(full)
    plan = sim.order_plan_from_policy(
        truth_demand,
        state,
        forecast_fn=lambda w, s, h: forecast_adapt[w],
        policy=lambda f, s: sim.reference_orders(f, s, 1.0, 1.0, 0.75),
    )
    write_plan("deficit_adaptive_feedback", plan["orders"])

    print("initial state written")
    print(initial.describe().to_string())
    print("baseline plans written to results/round2_baselines")


if __name__ == "__main__":
    main()
