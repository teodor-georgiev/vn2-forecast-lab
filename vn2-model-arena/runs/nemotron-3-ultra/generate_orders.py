"""Generate orders.csv for Round 2 of the demand-forecasting arena."""

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

# Ensure all series are present (some may have no sales history)
for sid in series_ids:
    if sid not in sales_pivot.index:
        sales_pivot.loc[sid] = 0.0

sales_pivot = sales_pivot.reindex(series_ids)
weeks = sales_pivot.columns.tolist()

# Get the last 157 weeks (full history) - but we need to forecast weeks 1-8 of the test
# The history is 157 weeks, we need to forecast 8 weeks ahead
# For the actual submission, we use the full 157 weeks as history


# Build features for each series
def compute_series_features(series_data):
    """Compute features from a series' demand history."""
    vals = series_data.values
    vals = vals[~np.isnan(vals)]

    if len(vals) == 0:
        return {
            "mean": 0.0,
            "std": 0.0,
            "cv": 0.0,
            "median": 0.0,
            "max": 0.0,
            "recent_mean_4": 0.0,
            "recent_mean_8": 0.0,
            "recent_mean_13": 0.0,
            "recent_mean_26": 0.0,
            "recent_mean_52": 0.0,
            "zero_frac": 1.0,
            "last_nonzero": 0.0,
            "weeks_since_nonzero": len(vals),
        }

    # Recent windows
    recent_4 = vals[-4:] if len(vals) >= 4 else vals
    recent_8 = vals[-8:] if len(vals) >= 8 else vals
    recent_13 = vals[-13:] if len(vals) >= 13 else vals
    recent_26 = vals[-26:] if len(vals) >= 26 else vals
    recent_52 = vals[-52:] if len(vals) >= 52 else vals

    # Non-zero stats
    nonzero = vals[vals > 0]
    last_nonzero = nonzero[-1] if len(nonzero) > 0 else 0.0

    # Weeks since last non-zero
    weeks_since_nonzero = 0
    for v in reversed(vals):
        if v > 0:
            break
        weeks_since_nonzero += 1

    return {
        "mean": np.mean(vals),
        "std": np.std(vals) if len(vals) > 1 else 0.0,
        "cv": np.std(vals) / np.mean(vals) if np.mean(vals) > 0 else 0.0,
        "median": np.median(vals),
        "max": np.max(vals),
        "recent_mean_4": np.mean(recent_4),
        "recent_mean_8": np.mean(recent_8),
        "recent_mean_13": np.mean(recent_13),
        "recent_mean_26": np.mean(recent_26),
        "recent_mean_52": np.mean(recent_52),
        "zero_frac": np.mean(vals == 0),
        "last_nonzero": last_nonzero,
        "weeks_since_nonzero": weeks_since_nonzero,
    }


# Compute features for all series
print("Computing features...")
features_list = []
for sid in series_ids:
    feats = compute_series_features(sales_pivot.loc[sid])
    feats["series_id"] = sid
    features_list.append(feats)

features_df = pd.DataFrame(features_list)

# Merge with product attributes
features_df = features_df.merge(attrs, on="series_id", how="left")

# Merge with initial state
features_df = features_df.merge(init, on="series_id", how="left")

# Now generate forecasts for the next 8 weeks (h=1 to h=8)
# We'll use a simple but effective approach: exponential smoothing + seasonal adjustment


def forecast_series(series_data, h=8):
    """Generate h-step ahead forecasts for a series."""
    vals = series_data.values
    vals = vals[~np.isnan(vals)]

    if len(vals) == 0:
        return np.zeros(h)

    # Handle all-zero series
    if np.all(vals == 0):
        return np.zeros(h)

    # Simple exponential smoothing with trend
    alpha = 0.3
    beta = 0.1

    level = vals[0]
    trend = 0.0

    for v in vals[1:]:
        new_level = alpha * v + (1 - alpha) * (level + trend)
        new_trend = beta * (new_level - level) + (1 - beta) * trend
        level, trend = new_level, new_trend

    # Damped trend for longer horizon
    phi = 0.9
    forecasts = []
    for i in range(1, h + 1):
        if i == 1:
            f = level + phi * trend
        else:
            f = level + phi * trend * (1 - phi**i) / (1 - phi)
        forecasts.append(max(0, f))

    return np.array(forecasts)


print("Generating forecasts...")
# Generate forecasts for all series (h=1..8)
all_forecasts = {}
for sid in series_ids:
    all_forecasts[sid] = forecast_series(sales_pivot.loc[sid], h=8)

# Now simulate the ordering policy
# We need orders for weeks 1-6 (arriving weeks 3-8)
# Weeks 7-8 have no orders but we still incur costs

WEEKS = 8
ORDER_WEEKS = 6
SHORTAGE_COST = 1.00
HOLDING_COST = 0.20


# Reference policy from inventory_sim.py
def reference_orders(forecast, state, scale=1.0, projection=1.0, buffer=0.75):
    """The reference deficit rule."""
    a, b, c = [np.maximum(0.0, np.asarray(f, dtype=float)) * scale for f in forecast]
    on_hand, arr1, arr2 = np.asarray(state, dtype=float).T
    projected = np.maximum(0.0, on_hand + arr1 + arr2 - projection * (a + b))
    deficit = c + buffer * np.sqrt(c) - projected
    return np.rint(np.clip(deficit, 0.0, None))


# Prepare initial state
initial_state = init.set_index("series_id").reindex(series_ids)
state = initial_state[["on_hand", "arrival_w1", "arrival_w2"]].values.astype(float)

# We need forecasts for h=1,2,3 for each ordering week
# Week 1 order arrives week 3, so needs forecast for weeks 3,4,5 (h=3,4,5 from now)
# Actually, let's think carefully:
# - At decision week 1 (start of week 1), we need forecasts for weeks 3,4,5 (h=3,4,5)
# - At decision week 2, we need forecasts for weeks 4,5,6 (h=4,5,6)
# - ...
# - At decision week 6, we need forecasts for weeks 8,9,10 (h=8,9,10) but we only have 8 weeks of demand
#
# The simulator expects forecast_fn(week_index, state, history) -> (3, n) for h1,h2,h3
# where week_index is 0-based (0=week1, ..., 5=week6)
# and h1,h2,h3 are demands for week_index+1, week_index+2, week_index+3

# Since we have 8 weeks of demand in the test period, and we're forecasting all 8 upfront:
# For ordering week w (0-indexed), we need forecasts for test weeks w+2, w+3, w+4
# (since order arrives in w+2, and we care about that week + next 2 weeks)

# Let's build the demand matrix for the test period (8 weeks)
# We'll use the last 8 weeks of historical data as a proxy? No, we need actual forecasts.
# The forecasts we generated above are for the NEXT 8 weeks after the history.
# So forecast h=1 is week 1 of test, h=2 is week 2, ..., h=8 is week 8.

# Actually wait - the sales data goes up to 2022-01-03 (week 157).
# The test period is 8 weeks after that. So our forecasts h=1..8 are exactly the test weeks 1..8.

# For the ordering policy at week w (0-indexed, w=0..5):
# We need forecasts for test weeks (w+2), (w+3), (w+4) which are h=(w+2), (w+3), (w+4)
# But we only have h=1..8. For w=5 (week 6), we need h=7,8,9 - we only have 7,8.
# We'll extrapolate h=9 as same as h=8.

print("Running ordering policy...")
orders_matrix = np.zeros((ORDER_WEEKS, len(series_ids)))

# Track state through the 8 weeks
current_state = state.copy()
revealed_demand = []  # We don't know true demand, but policy doesn't use it in reference_orders

for w in range(WEEKS):
    if w < ORDER_WEEKS:
        # Get forecasts for h1, h2, h3 relative to this decision point
        # At week w, order arrives at w+2, so we need demand for w+2, w+3, w+4
        h1_idx = w + 1  # 0-indexed: week w+2 is h=w+2, so index w+1
        h2_idx = w + 2
        h3_idx = w + 3

        forecast_h = []
        for sid_idx, sid in enumerate(series_ids):
            fc = all_forecasts[sid]
            h1 = fc[h1_idx] if h1_idx < len(fc) else fc[-1]
            h2 = fc[h2_idx] if h2_idx < len(fc) else fc[-1]
            h3 = fc[h3_idx] if h3_idx < len(fc) else fc[-1]
            forecast_h.append([h1, h2, h3])

        forecast_h = np.array(forecast_h).T  # (3, n)

        # Apply policy
        order_qty = reference_orders(forecast_h, current_state)
        orders_matrix[w] = order_qty

    # For simulation we'd step through, but we only need the orders
    # The actual simulator will handle the state transitions

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

print(f"Generated {len(orders_df)} orders")
print(f"Saved to {RUN_DIR / 'orders.csv'}")
print("\nSample:")
print(orders_df.head(20))
print("\nOrder statistics:")
print(orders_df["quantity"].describe())
print(f"\nTotal units ordered: {orders_df['quantity'].sum()}")
