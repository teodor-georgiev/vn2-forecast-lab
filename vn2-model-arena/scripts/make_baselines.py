"""Reference baselines for the demand-forecasting arena (scored like any contestant)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ARENA = Path(r"C:\Users\Toni\Documents\Default Project\vn2-model-arena")
OUT = ARENA / "results" / "baselines"
HORIZON = 8


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    sales = pd.read_csv(ARENA / "data" / "sales.csv")
    sales["week"] = pd.to_datetime(sales["week"])
    sales = sales.sort_values(["series_id", "week"])
    pivot = sales.pivot(index="series_id", columns="week", values="y")
    series = pivot.index.to_numpy()
    values = pivot.to_numpy(dtype=float)
    n_weeks = values.shape[1]

    subs: dict[str, np.ndarray] = {
        # horizon x series matrices
        "zero": np.zeros((HORIZON, len(series))),
        "last_value": np.repeat(values[:, -1:], HORIZON, axis=1).T,
        "mean_13": np.repeat(
            values[:, -13:].mean(axis=1, keepdims=True), HORIZON, axis=1
        ).T,
        "median_13": np.repeat(
            np.median(values[:, -13:], axis=1, keepdims=True), HORIZON, axis=1
        ).T,
        "seasonal_52": np.stack(
            [values[:, n_weeks - 52 + h] for h in range(HORIZON)], axis=1
        ).T,
    }

    for name, matrix in subs.items():
        rows = [
            (sid, h + 1, float(matrix[h, i]))
            for i, sid in enumerate(series)
            for h in range(HORIZON)
        ]
        frame = pd.DataFrame(rows, columns=["series_id", "horizon", "y_hat"])
        frame.to_csv(OUT / f"{name}.csv", index=False)
        print(f"wrote {OUT / (name + '.csv')}")


if __name__ == "__main__":
    main()
