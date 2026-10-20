"""Prepare an anonymised 599-series, 8-week demand-forecasting arena from the VN2 raw files.

Writes:
  arena/data/            what the competing models see (anonymised, truncated at the origin)
  private/               truth + mapping, kept outside the arena directory

Anonymisation: (Store, Product) -> salted hash series id; per-series random positive
scale factor applied to demand; synthetic weekly dates. This breaks any join back to
the original competition files that also exist on this machine.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path(r"C:\Users\Toni\DemandAhead-VN2Game\raw")
ARENA = Path(r"C:\Users\Toni\Documents\Default Project\vn2-model-arena")
PRIVATE = Path(r"C:\Users\Toni\AppData\Local\Temp\opencode\vn2-arena-private")

SALT = "vn2-arena-2026-10-19-8f3c1d7a"
SYNTH_START = pd.Timestamp("2019-01-07")  # synthetic weekly Mondays
HORIZON = 8


def salted_hex(*parts: object) -> str:
    h = hashlib.sha256(SALT.encode("utf-8"))
    for part in parts:
        h.update(str(part).encode("utf-8"))
        h.update(b"|")
    return h.hexdigest()


def main() -> None:
    week0 = pd.read_csv(RAW / "Week 0 - 2024-04-08 - Sales.csv")
    week8 = pd.read_csv(RAW / "Week 8 - 2024-06-03 - Sales.csv")
    in_stock = pd.read_csv(RAW / "Week 0 - In Stock.csv")
    master = pd.read_csv(RAW / "Week 0 - Master.csv")

    for name, frame in [("week0", week0), ("week8", week8), ("in_stock", in_stock)]:
        assert len(frame) == 599, f"{name}: expected 599 rows, got {len(frame)}"

    key = ["Store", "Product"]
    week0_dates = [c for c in week0.columns if c not in key]
    week8_dates = [c for c in week8.columns if c not in key]
    stock_dates = [c for c in in_stock.columns if c not in key]
    assert len(week0_dates) == 157, len(week0_dates)
    assert len(week8_dates) == 165, len(week8_dates)
    assert week8_dates[:157] == week0_dates, "history columns must align"
    holdout_dates = week8_dates[157:]
    assert len(holdout_dates) == HORIZON, holdout_dates
    assert holdout_dates == stock_dates[157:], "in-stock and sales must share dates"
    assert week0_dates[0] == "2021-04-12" and week0_dates[-1] == "2024-04-08"
    assert holdout_dates[0] == "2024-04-15" and holdout_dates[-1] == "2024-06-03"

    week0 = week0.sort_values(key).reset_index(drop=True)
    week8 = week8.sort_values(key).reset_index(drop=True)
    in_stock = in_stock.sort_values(key).reset_index(drop=True)
    assert (week0[key].to_numpy() == week8[key].to_numpy()).all()

    # --- anonymisation map -------------------------------------------------------
    series_ids: list[str] = []
    scales: list[float] = []
    for store, product in week0[key].itertuples(index=False):
        digest = salted_hex(store, product)
        series_ids.append("s" + digest[:12])
        u = int(digest[12:20], 16) / 16**8  # deterministic uniform in [0, 1)
        scales.append(0.60 + 0.80 * u)  # per-series positive scale
    series_ids_arr = np.array(series_ids)
    scales_arr = np.array(scales)
    assert len(set(series_ids)) == 599

    synth_dates = [SYNTH_START + pd.Timedelta(weeks=i) for i in range(len(week0_dates))]

    # --- what the models see -----------------------------------------------------
    hist = week0[week0_dates].to_numpy(dtype=float) * scales_arr[:, None]
    train_rows = []
    for i, sid in enumerate(series_ids_arr):
        for w, date in enumerate(synth_dates):
            train_rows.append((sid, date, float(hist[i, w])))
    train = pd.DataFrame(train_rows, columns=["series_id", "week", "y"])
    train["week"] = train["week"].dt.strftime("%Y-%m-%d")

    stock_hist = in_stock[week0_dates].to_numpy()
    avail_rows = []
    for i, sid in enumerate(series_ids_arr):
        for w, date in enumerate(synth_dates):
            avail_rows.append((sid, date, bool(stock_hist[i, w])))
    avail = pd.DataFrame(avail_rows, columns=["series_id", "week", "is_available"])
    avail["week"] = avail["week"].dt.strftime("%Y-%m-%d")

    attrs = master.sort_values(key).reset_index(drop=True)
    attr_cols = [c for c in attrs.columns if c not in key]
    products = attrs[attr_cols].copy()
    products.insert(0, "series_id", series_ids_arr)
    for col in attr_cols:
        products[col] = [salted_hex("attr", col, v)[:6] for v in attrs[col]]

    # --- private truth -----------------------------------------------------------
    truth_vals = week8[holdout_dates].to_numpy(dtype=float) * scales_arr[:, None]
    truth_stock = in_stock[holdout_dates].to_numpy()
    truth_rows = []
    for i, sid in enumerate(series_ids_arr):
        for h in range(HORIZON):
            truth_rows.append(
                (sid, h + 1, float(truth_vals[i, h]), bool(truth_stock[i, h]))
            )
    truth = pd.DataFrame(
        truth_rows, columns=["series_id", "horizon", "y_true", "is_available"]
    )

    mapping = pd.DataFrame({"series_id": series_ids_arr, "scale": scales_arr})
    mapping[key[0]] = week0[key[0]].to_numpy()
    mapping[key[1]] = week0[key[1]].to_numpy()

    data_dir = ARENA / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    train.to_csv(data_dir / "sales.csv", index=False)
    avail.to_csv(data_dir / "availability.csv", index=False)
    products.to_csv(data_dir / "product_attributes.csv", index=False)

    PRIVATE.mkdir(parents=True, exist_ok=True)
    truth.to_csv(PRIVATE / "answer_panel.csv", index=False)
    mapping.to_csv(PRIVATE / "series_mapping.csv", index=False)
    (PRIVATE / "meta.json").write_text(
        json.dumps(
            {
                "created": pd.Timestamp.now().isoformat(),
                "origin_raw_week": "2024-04-08",
                "holdout_raw_weeks": holdout_dates,
                "series": int(len(series_ids_arr)),
                "history_weeks": len(week0_dates),
                "horizon": HORIZON,
                "salt": SALT,
                "raw_sales_sha256": hashlib.sha256(
                    (RAW / "Week 0 - 2024-04-08 - Sales.csv").read_bytes()
                ).hexdigest(),
                "raw_truth_sha256": hashlib.sha256(
                    (RAW / "Week 8 - 2024-06-03 - Sales.csv").read_bytes()
                ).hexdigest(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"train rows      : {len(train)}")
    print(f"availability    : {len(avail)}")
    print(f"product attrs   : {len(products)}")
    print(f"private truth   : {len(truth)}")
    print(f"holdout truth   : total demand {truth['y_true'].sum():,.1f}")
    print(f"censored cells  : {(~truth['is_available']).sum()}")


if __name__ == "__main__":
    main()
