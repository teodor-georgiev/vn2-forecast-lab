import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
import warnings

warnings.filterwarnings("ignore")

# Load data
sales = pd.read_csv("../../data/sales.csv")
avail = pd.read_csv("../../data/availability.csv")
attr = pd.read_csv("../../data/product_attributes.csv")

sales["week"] = pd.to_datetime(sales["week"])
avail["week"] = pd.to_datetime(avail["week"])

# Merge availability
df = sales.merge(avail, on=["series_id", "week"], how="left")

# Create features
df = df.sort_values(["series_id", "week"]).reset_index(drop=True)

# Lag features
for lag in [1, 2, 3, 4, 8, 13, 26, 52]:
    df[f"lag_{lag}"] = df.groupby("series_id")["y"].shift(lag)

# Rolling stats
for window in [4, 8, 13, 26, 52]:
    df[f"roll_mean_{window}"] = df.groupby("series_id")["y"].transform(
        lambda x: x.shift(1).rolling(window, min_periods=1).mean()
    )
    df[f"roll_std_{window}"] = df.groupby("series_id")["y"].transform(
        lambda x: x.shift(1).rolling(window, min_periods=1).std()
    )

# Time features
df["week_of_year"] = df["week"].dt.isocalendar().week
df["month"] = df["week"].dt.month
df["quarter"] = df["week"].dt.quarter

# Merge attributes
df = df.merge(attr, on="series_id", how="left")

# Encode categorical attributes
cat_cols = [
    "ProductGroup",
    "Division",
    "Department",
    "DepartmentGroup",
    "StoreFormat",
    "Format",
]
for col in cat_cols:
    df[col] = df[col].astype("category").cat.codes

# Fill NaN
df = df.fillna(0)

# Train on all but last 8 weeks (for validation), but for final model use all data
# Actually, let's train on all available data and predict next 8 weeks
last_week = df["week"].max()
train = df[df["week"] <= last_week].copy()

# Feature columns
feature_cols = [c for c in df.columns if c not in ["series_id", "week", "y"]]

# Train model
X_train = train[feature_cols]
y_train = train["y"]

model = LGBMRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=6,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    verbosity=-1,
)
model.fit(X_train, y_train)

# Now create future predictions iteratively for 8 horizons
series_ids = sorted(df["series_id"].unique())
predictions = []

# Get the last known state for each series
last_data = df[df["week"] == last_week].set_index("series_id")

for h in range(1, 9):
    # Build features for horizon h
    future_rows = []
    for sid in series_ids:
        row = {"series_id": sid, "horizon": h}

        # Get historical lags from the last known data + previously predicted
        # For simplicity, use the last known values
        hist = df[df["series_id"] == sid].sort_values("week")

        # We'll build the feature vector using the last available data
        last_row = hist.iloc[-1]

        feat = {}
        for lag in [1, 2, 3, 4, 8, 13, 26, 52]:
            if h <= lag:
                feat[f"lag_{lag}"] = hist.iloc[-lag]["y"]
            else:
                # Would need previous predictions - skip for now, use lag_1 value
                feat[f"lag_{lag}"] = hist.iloc[-1]["y"]

        for window in [4, 8, 13, 26, 52]:
            feat[f"roll_mean_{window}"] = hist["y"].tail(window).mean()
            feat[f"roll_std_{window}"] = hist["y"].tail(window).std()

        # Time features for future week
        future_week = last_week + pd.Timedelta(weeks=h)
        feat["week_of_year"] = future_week.isocalendar().week
        feat["month"] = future_week.month
        feat["quarter"] = future_week.quarter

        # Attributes
        attr_row = attr[attr["series_id"] == sid].iloc[0]
        for col in cat_cols:
            feat[col] = attr_row[col]

        # is_available - assume available for future
        feat["is_available"] = 1

        future_rows.append(feat)

    future_df = pd.DataFrame(future_rows)
    for col in cat_cols:
        future_df[col] = future_df[col].astype("category").cat.codes
    future_df = future_df.fillna(0)

    # Ensure column order matches training
    future_df = future_df[feature_cols]

    preds = model.predict(future_df)
    preds = np.maximum(preds, 0)  # Non-negative

    for i, sid in enumerate(series_ids):
        predictions.append({"series_id": sid, "horizon": h, "y_hat": preds[i]})

pred_df = pd.DataFrame(predictions)
pred_df.to_csv("predictions.csv", index=False)
print(f"Generated {len(pred_df)} predictions")
print(pred_df.head(16))
