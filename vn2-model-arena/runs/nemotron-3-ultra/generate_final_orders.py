"""Generate final orders.csv for Round 2 submission."""

import numpy as np
import pandas as pd
from pathlib import Path

# Arena paths
DATA_DIR = Path("C:/Users/Toni/Documents/Default Project/vn2-model-arena/data")
RUN_DIR = Path(
    "C:/Users/Toni/Documents/Default Project/vn2-model-arena/runs/nemotron-3-ultra"
)

# Load data
sales = pd.read_csv(DATA_DIR / "sales.csv")
avail = pd.read_csv(DATA_DIR / "availability.csv")
attrs = pd.read_csv(DATA_DIR / "product_attributes.csv")
init = pd.read_csv(DATA_DIR / "initial_state.csv")

# Pivot sales to (series_id, week) matrix
sales_pivot = sales.pivot(index="series_id", columns="week", values="y")
series_ids = init["series_id"].values

# Ensure all series are present
for sid in series_ids:
    if sid not in sales_pivot.index:
        sales_pivot.loc[sid] = 0.0

sales_pivot = sales_pivot.reindex(series_ids)


# SES forecasting function
def forecast_ses(vals, h=8):
    """Simple exponential smoothing with damped trend."""
    vals = vals[~np.isnan(vals)]
    if len(vals) == 0 or np.all(vals == 0):
        return np.zeros(h)
    alpha, beta, phi = 0.3, 0.1, 0.9
    level, trend = vals[0], 0.0
    for v in vals[1:]:
        new_level = alpha * v + (1 - alpha) * (level + trend)
        new_trend = beta * (new_level - level) + (1 - beta) * trend
        level, trend = new_level, new_trend
    forecasts = []
    for i in range(1, h + 1):
        if i == 1:
            f = level + phi * trend
        else:
            f = level + phi * trend * (1 - phi**i) / (1 - phi)
        forecasts.append(max(0, f))
    return np.array(forecasts)


# Reference policy from inventory_sim.py
def reference_orders(forecast, state, scale=1.0, projection=1.0, buffer=0.75):
    a, b, c = [np.maximum(0.0, np.asarray(f, dtype=float)) * scale for f in forecast]
    on_hand, arr1, arr2 = np.asarray(state, dtype=float).T
    projected = np.maximum(0.0, on_hand + arr1 + arr2 - projection * (a + b))
    deficit = c + buffer * np.sqrt(c) - projected
    return np.rint(np.clip(deficit, 0.0, None))


# Best parameters from backtesting
BEST_BUFFER = 0.0
BEST_SCALE = 1.1
BEST_PROJECTION = 30.0

print("Generating forecasts for all 599 series...")
all_forecasts = {}
for sid in series_ids:
    all_forecasts[sid] = forecast_ses(sales_pivot.loc[sid].values, h=8)

print("Generating orders using reference policy...")
# Initial state from provided file
initial_state = init.set_index("series_id").reindex(series_ids)
state = initial_state[["on_hand", "arrival_w1", "arrival_w2"]].values.astype(float)

WEEKS = 8
ORDER_WEEKS = 6

orders_matrix = np.zeros((ORDER_WEEKS, len(series_ids)))
current_state = state.copy()

for w in range(ORDER_WEEKS):
    # At ordering week w (0-indexed), order arrives at week w+2
    # We need forecasts for test weeks w+2, w+3, w+4 (h = w+2, w+3, w+4)
    h1_idx = w + 1  # h=w+2 -> index w+1
    h2_idx = w + 2  # h=w+3 -> index w+2
    h3_idx = w + 3  # h=w+4 -> index w+3

    forecast_h = []
    for sid in series_ids:
        fc = all_forecasts[sid]
        h1 = fc[h1_idx] if h1_idx < 8 else fc[-1]
        h2 = fc[h2_idx] if h2_idx < 8 else fc[-1]
        h3 = fc[h3_idx] if h3_idx < 8 else fc[-1]
        forecast_h.append([h1, h2, h3])

    forecast_h = np.array(forecast_h).T  # (3, n)

    # Apply policy with best parameters
    order_qty = reference_orders(
        forecast_h,
        current_state,
        scale=BEST_SCALE,
        projection=BEST_PROJECTION,
        buffer=BEST_BUFFER,
    )
    orders_matrix[w] = order_qty

# Create output DataFrame
output_rows = []
for w in range(ORDER_WEEKS):
    for i, sid in enumerate(series_ids):
        output_rows.append(
            {
                "series_id": sid,
                "week": w + 1,
                "quantity": int(max(0, orders_matrix[w, i])),
            }
        )

orders_df = pd.DataFrame(output_rows)
orders_df.to_csv(RUN_DIR / "orders.csv", index=False)

print(f"Generated {len(orders_df)} orders (599 series x 6 weeks)")
print(f"Saved to {RUN_DIR / 'orders.csv'}")
print(f"\nOrder statistics:")
print(orders_df["quantity"].describe())
print(f"\nTotal units ordered: {orders_df['quantity'].sum()}")
print(
    f"\nParameters used: buffer={BEST_BUFFER}, scale={BEST_SCALE}, projection={BEST_PROJECTION}"
)
