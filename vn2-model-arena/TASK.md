# Demand-forecasting arena — task brief

You are given the weekly demand history of **599 anonymised store-product series**.
Your job: **forecast the next 8 weeks for every series.**

## Data (read-only)

Folder: `C:\Users\Toni\Documents\Default Project\vn2-model-arena\data\`

| File | Columns | Notes |
|---|---|---|
| `sales.csv` | `series_id, week, y` | weekly (ISO Monday dates), 157 weeks of history, demand in anonymised units |
| `availability.csv` | `series_id, week, is_available` | whether the item was on the shelf that week |
| `product_attributes.csv` | `series_id, ...` | anonymised categorical attributes |

History ends at the last week present in `sales.csv`. Your forecasts cover the **8
consecutive weeks immediately after it**; `horizon = 1` is the first of those weeks.

## Deliverable

Exactly one file, in your own run directory:

```
predictions.csv
header: series_id,horizon,y_hat
```

- 599 series × 8 horizons = **4792 rows**, one per (series_id, horizon).
- `horizon` in 1..8, `y_hat` finite and >= 0, no duplicate (series_id, horizon).
- Long format, comma separated, exactly those three columns.

## Rules

- Work only inside your own run directory. Read only the arena `data` folder.
- No internet. Python 3.12 is available with pandas, numpy, scikit-learn, lightgbm,
  xgboost, catboost and statsmodels installed.
- You have a **15-minute working budget**. Submit even if your method is simple —
  a missing or malformed file scores nothing.
- Grading uses `predictions.csv` only. Do not report your own metrics as evidence;
  anything you claim about accuracy is ignored.
