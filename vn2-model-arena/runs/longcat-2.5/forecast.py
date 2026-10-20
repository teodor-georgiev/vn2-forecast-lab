import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
import time
import warnings

warnings.filterwarnings("ignore")

start = time.time()

# Load data
sales = pd.read_csv(
    r"C:\Users\Toni\Documents\Default Project\vn2-model-arena\data\sales.csv"
)
avail = pd.read_csv(
    r"C:\Users\Toni\Documents\Default Project\vn2-model-arena\data\availability.csv"
)
attrs = pd.read_csv(
    r"C:\Users\Toni\Documents\Default Project\vn2-model-arena\data\product_attributes.csv"
)

sales["week"] = pd.to_datetime(sales["week"])
avail["week"] = pd.to_datetime(avail["week"])

# Get last week and create future weeks
last_week = sales["week"].max()
future_weeks = [last_week + pd.Timedelta(weeks=i) for i in range(1, 9)]

# Merge availability
sales = sales.merge(avail, on=["series_id", "week"], how="left")

# Sort
sales = sales.sort_values(["series_id", "week"]).reset_index(drop=True)


# Create features per series
def create_features(df):
    df = df.copy()
    # Lag features
    for lag in [1, 2, 3, 4, 8, 12, 26, 52]:
        df[f"lag_{lag}"] = df["y"].shift(lag)
    # Rolling statistics
    df["roll_mean_4"] = df["y"].shift(1).rolling(4, min_periods=1).mean()
    df["roll_mean_8"] = df["y"].shift(1).rolling(8, min_periods=1).mean()
    df["roll_mean_12"] = df["y"].shift(1).rolling(12, min_periods=1).mean()
    df["roll_std_4"] = df["y"].shift(1).rolling(4, min_periods=1).std().fillna(0)
    df["roll_std_8"] = df["y"].shift(1).rolling(8, min_periods=1).std().fillna(0)
    # Time features
    df["week_of_year"] = df["week"].dt.isocalendar().week.astype(int)
    df["month"] = df["week"].dt.month
    df["day_of_year"] = df["week"].dt.dayofyear
    # Trend
    df["trend"] = np.arange(len(df))
    # Availability
    df["is_avail"] = (
        df["is_available"].astype(int) if "is_available" in df.columns else 1
    )
    return df


# Build training data with features
print("Creating features...")
featured = sales.groupby("series_id", group_keys=False).apply(create_features)

# Merge attributes
featured = featured.merge(attrs, on="series_id", how="left")

# Feature columns
lag_cols = [c for c in featured.columns if c.startswith("lag_")]
roll_cols = [c for c in featured.columns if c.startswith("roll_")]
time_cols = ["week_of_year", "month", "day_of_year", "trend", "is_avail"]
cat_cols = [
    "ProductGroup",
    "Division",
    "Department",
    "DepartmentGroup",
    "StoreFormat",
    "Format",
]

feature_cols = lag_cols + roll_cols + time_cols + cat_cols

# Training data (drop rows with NaN in lag features)
train = featured.dropna(subset=lag_cols).copy()

# Encode categorical
for c in cat_cols:
    train[c] = train[c].astype("category")

print(f"Training samples: {len(train)}")
print(f"Features: {len(feature_cols)}")

# Train model
print("Training LightGBM...")
model = LGBMRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    verbose=-1,
)

X_train = train[feature_cols]
y_train = train["y"]

model.fit(X_train, y_train, categorical_feature=cat_cols)

print(f"Training done in {time.time() - start:.1f}s")

# Forecast each series
print("Forecasting...")
predictions = []

for sid in sales["series_id"].unique():
    # Get last 52 weeks of history for this series
    hist = sales[sales["series_id"] == sid].sort_values("week").tail(52).copy()

    if len(hist) == 0:
        # No history, predict 0
        for h in range(1, 9):
            predictions.append({"series_id": sid, "horizon": h, "y_hat": 0.0})
        continue

    # Get attributes
    attr = (
        attrs[attrs["series_id"] == sid].iloc[0].to_dict()
        if sid in attrs["series_id"].values
        else {}
    )

    # Iterative forecasting
    current_hist = hist.copy()
    for h in range(1, 9):
        # Create features for next week
        next_week = current_hist["week"].max() + pd.Timedelta(weeks=1)

        # Build feature row
        row = {"week": next_week, "y": current_hist["y"].iloc[-1], "is_available": True}

        # Lags
        for lag in [1, 2, 3, 4, 8, 12, 26, 52]:
            if len(current_hist) >= lag:
                row[f"lag_{lag}"] = current_hist["y"].iloc[-lag]
            else:
                row[f"lag_{lag}"] = current_hist["y"].mean()

        # Rolling
        row["roll_mean_4"] = current_hist["y"].tail(4).mean()
        row["roll_mean_8"] = current_hist["y"].tail(8).mean()
        row["roll_mean_12"] = current_hist["y"].tail(12).mean()
        row["roll_std_4"] = (
            current_hist["y"].tail(4).std() if len(current_hist) >= 4 else 0
        )
        row["roll_std_8"] = (
            current_hist["y"].tail(8).std() if len(current_hist) >= 8 else 0
        )

        # Time
        row["week_of_year"] = next_week.isocalendar().week
        row["month"] = next_week.month
        row["day_of_year"] = next_week.dayofyear
        row["trend"] = len(current_hist)
        row["is_avail"] = 1

        # Categorical
        for c in cat_cols:
            row[c] = attr.get(c, "")

        # Predict
        X_pred = pd.DataFrame([row])[feature_cols]
        for c in cat_cols:
            X_pred[c] = X_pred[c].astype("category")

        y_pred = model.predict(X_pred)[0]
        y_pred = max(0, y_pred)  # Non-negative

        predictions.append({"series_id": sid, "horizon": h, "y_hat": y_pred})

        # Append to history for next iteration
        new_row = pd.DataFrame(
            [{"series_id": sid, "week": next_week, "y": y_pred, "is_available": True}]
        )
        current_hist = pd.concat([current_hist, new_row], ignore_index=True)

# Save
pred_df = pd.DataFrame(predictions)
pred_df = pred_df[["series_id", "horizon", "y_hat"]]
pred_df.to_csv(
    r"C:\Users\Toni\Documents\Default Project\vn2-model-arena\runs\longcat-2.5\predictions.csv",
    index=False,
)

print(f"Done! {len(pred_df)} predictions saved in {time.time() - start:.1f}s")
print(pred_df.head(10))
