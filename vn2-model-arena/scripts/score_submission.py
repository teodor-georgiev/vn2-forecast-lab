"""Frozen scorer for the demand-forecasting arena.

Usage:
    python score_submission.py predictions.csv [--label NAME]

Reads the private answer panel, validates the submission strictly, and prints one
JSON block with the scores. Validity gate: exactly 599 series x 8 horizons, unique
(series_id, horizon) pairs, finite non-negative numeric y_hat. An invalid submission
is still scored diagnostically (missing rows counted as forecasts of 0) but is
marked invalid.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

PRIVATE = Path(r"C:\Users\Toni\AppData\Local\Temp\opencode\vn2-arena-private")
HORIZON = 8


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--label", default=None)
    args = parser.parse_args()

    truth = pd.read_csv(PRIVATE / "answer_panel.csv")
    truth["series_id"] = truth["series_id"].astype(str)
    expected_series = set(truth["series_id"])

    problems: list[str] = []
    try:
        pred = pd.read_csv(args.predictions)
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

    pred.columns = [str(c).strip() for c in pred.columns]
    required = {"series_id", "horizon", "y_hat"}
    if not required.issubset(pred.columns):
        problems.append(f"missing columns: {sorted(required - set(pred.columns))}")
    if problems:
        print(
            json.dumps(
                {"label": args.label, "valid": False, "problems": problems}, indent=2
            )
        )
        return

    pred = pred[["series_id", "horizon", "y_hat"]].copy()
    pred["series_id"] = pred["series_id"].astype(str)
    pred["horizon"] = pd.to_numeric(pred["horizon"], errors="coerce")
    pred["y_hat"] = pd.to_numeric(pred["y_hat"], errors="coerce")

    if pred["horizon"].isna().any():
        problems.append("non-numeric horizons")
    if pred["y_hat"].isna().any():
        problems.append(f"{int(pred['y_hat'].isna().sum())} non-numeric y_hat values")
    if (pred["y_hat"].dropna() < 0).any():
        problems.append("negative y_hat values")
    if not np.isfinite(pred["y_hat"].dropna().to_numpy()).all():
        problems.append("non-finite y_hat values")
    dupes = int(pred.duplicated(["series_id", "horizon"]).sum())
    if dupes:
        problems.append(f"{dupes} duplicate (series_id, horizon) rows")
    unknown = set(pred["series_id"]) - expected_series
    if unknown:
        problems.append(f"{len(unknown)} unknown series_id values")
    if len(pred) != 599 * HORIZON:
        problems.append(f"{len(pred)} rows, expected {599 * HORIZON}")

    merged = truth.merge(pred, on=["series_id", "horizon"], how="left")
    merged["y_hat"] = merged["y_hat"].fillna(0.0)
    missing_rows = int(
        len(merged) - merged["y_hat"].notna().sum() + merged["y_hat"].isna().sum()
    )
    y = merged["y_true"].to_numpy(dtype=float)
    yhat = merged["y_hat"].to_numpy(dtype=float)
    err = yhat - y
    total = float(y.sum())
    wape = float(np.abs(err).sum() / total)
    mae = float(np.abs(err).mean())
    rmse = float(np.sqrt((err**2).mean()))
    pbias = float(err.sum() / total)
    per_h = (
        merged.assign(abs_err=err.__abs__())
        .groupby("horizon")
        .apply(
            lambda g: float(g["abs_err"].sum() / g["y_true"].sum()),
            include_groups=False,
        )
        .to_dict()
    )
    zero_forecast_wape = 1.0

    print(
        json.dumps(
            {
                "label": args.label,
                "valid": not problems,
                "problems": problems,
                "rows": int(len(pred)),
                "missing_or_filled_rows": missing_rows,
                "wape": wape,
                "mae": mae,
                "rmse": rmse,
                "pbias": pbias,
                "per_horizon_wape": {int(k): v for k, v in per_h.items()},
                "total_true_demand": total,
                "zero_forecast_wape": zero_forecast_wape,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
