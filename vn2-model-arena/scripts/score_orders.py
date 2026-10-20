"""Frozen scorer for round 2 (the ordering game).

Usage:
    python score_orders.py orders.csv [--label NAME]

Validates the plan strictly, replays it with the arena's inventory simulator against
the private answer panel, and prints one JSON block. Validity gate: exactly 599 series
x 6 order weeks, unique pairs, finite non-negative quantities. Invalid plans are still
replayed diagnostically (missing quantities treated as 0) but marked invalid.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ARENA = Path(r"C:\Users\Toni\Documents\Default Project\vn2-model-arena")
PRIVATE = Path(r"C:\Users\Toni\AppData\Local\Temp\opencode\vn2-arena-private")
sys.path.insert(0, str(ARENA / "tools"))
import inventory_sim as sim  # noqa: E402

ORDER_WEEKS = 6


def load_truth() -> tuple[np.ndarray, np.ndarray, list[str]]:
    truth = pd.read_csv(PRIVATE / "answer_panel.csv")
    initial = pd.read_csv(ARENA / "data" / "initial_state.csv")
    series = sorted(truth["series_id"].unique().tolist())
    demand = (
        truth.pivot(index="horizon", columns="series_id", values="y_true")
        .loc[range(1, sim.WEEKS + 1), series]
        .to_numpy(float)
    )
    state = (
        initial.set_index("series_id")
        .loc[series][["on_hand", "arrival_w1", "arrival_w2"]]
        .to_numpy(float)
    )
    return demand, state, series


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("orders", type=Path)
    parser.add_argument("--label", default=None)
    args = parser.parse_args()

    demand, state, series = load_truth()

    problems: list[str] = []
    try:
        plan = pd.read_csv(args.orders)
    except Exception as exc:  # noqa: BLE001
        print(
            json.dumps(
                {
                    "label": args.label,
                    "valid": False,
                    "problems": [f"unreadable: {exc}"],
                },
                indent=2,
            )
        )
        return

    plan.columns = [str(c).strip() for c in plan.columns]
    required = {"series_id", "week", "quantity"}
    if not required.issubset(plan.columns):
        print(
            json.dumps(
                {
                    "label": args.label,
                    "valid": False,
                    "problems": [
                        f"missing columns: {sorted(required - set(plan.columns))}"
                    ],
                },
                indent=2,
            )
        )
        return

    plan = plan[["series_id", "week", "quantity"]].copy()
    plan["series_id"] = plan["series_id"].astype(str)
    plan["week"] = pd.to_numeric(plan["week"], errors="coerce")
    plan["quantity"] = pd.to_numeric(plan["quantity"], errors="coerce")

    if plan["week"].isna().any():
        problems.append("non-numeric weeks")
    if plan["quantity"].isna().any():
        problems.append(f"{int(plan['quantity'].isna().sum())} non-numeric quantities")
    if (plan["quantity"].dropna() < 0).any():
        problems.append("negative quantities")
    if not np.isfinite(plan["quantity"].dropna().to_numpy()).all():
        problems.append("non-finite quantities")
    dupes = int(plan.duplicated(["series_id", "week"]).sum())
    if dupes:
        problems.append(f"{dupes} duplicate (series_id, week) rows")
    if set(plan["series_id"]) != set(series):
        problems.append("series_id set does not match the 599 arena series")
    if len(plan) != 599 * ORDER_WEEKS:
        problems.append(f"{len(plan)} rows, expected {599 * ORDER_WEEKS}")

    grid = pd.MultiIndex.from_product(
        [series, range(1, ORDER_WEEKS + 1)], names=["series_id", "week"]
    )
    filled = (
        plan.dropna(subset=["week"])
        .assign(week=lambda d: d["week"].astype(int))
        .drop_duplicates(["series_id", "week"], keep="last")
        .set_index(["series_id", "week"])["quantity"]
        .reindex(grid)
    )
    missing = int(filled.isna().sum())
    orders = np.rint(filled.fillna(0.0).to_numpy(float)).reshape(599, ORDER_WEEKS).T

    result = sim.simulate(demand, state, orders)
    holding = sum(r["holding"] for r in result["weekly"][2:])
    shortage = sum(r["shortage"] for r in result["weekly"][2:])
    print(
        json.dumps(
            {
                "label": args.label,
                "valid": not problems,
                "problems": problems,
                "rows": int(len(plan)),
                "missing_or_filled_rows": missing,
                "score": result["score"],
                "setup": result["setup"],
                "holding_weeks3_8": holding,
                "shortage_weeks3_8": shortage,
                "ordered_total": int(orders.sum()),
                "weekly": result["weekly"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
