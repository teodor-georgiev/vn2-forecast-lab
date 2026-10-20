import pandas as pd, numpy as np, time, os, sys

BASE = "C:/Users/Toni/Documents/Default Project/vn2-model-arena"
DATA = BASE + "/data"
RUN = BASE + "/runs/beam"
sys.path.insert(0, BASE + "/tools")
from inventory_sim import simulate, reference_orders, order_plan_from_policy

sales = pd.read_csv(DATA + "/sales.csv")
avail = pd.read_csv(DATA + "/availability.csv")
attrs = pd.read_csv(DATA + "/product_attributes.csv")
df = sales.merge(avail, on=["series_id", "week"])
df["week"] = pd.to_datetime(df["week"])
series_ids = sorted(df.series_id.unique())
sid_idx = {s: i for i, s in enumerate(series_ids)}
n = len(series_ids)

# demand matrix (T, n)
weeks = sorted(df.week.unique())
T = len(weeks)
D = np.zeros((T, n))
for i, s in enumerate(series_ids):
    sub = df[df.series_id == s].sort_values("week")
    # align
    wmap = {pd.Timestamp(w): v for w, v in zip(sub.week, sub.y)}
    for t, w in enumerate(weeks):
        D[t, i] = wmap.get(pd.Timestamp(w), 0.0)

print("T,n", T, n)


# initial state zeros for backtest windows? We'll use zeros.
def make_initial(n):
    return np.zeros((n, 3))


# Forecast function: seasonal naive (same week last year) with fallback recent mean
def forecast_sn(history_weeks_indices, horizon=8):
    # history_weeks_indices: list of week indices available (0-based) up to current
    # returns (3,n) for h1,h2,h3? For order_plan_from_policy, forecast_fn(week_index,state,history) should forecast next 3 horizons
    pass


# Instead, implement a simple policy test using simulate on a fixed 8-week window with a forecast array.
# We'll create forecast for window using seasonal naive from data before window.
def seasonal_naive_forecast(train_D, steps):
    # train_D: (t, n) history up to before window
    # forecast h steps using y at t-52 (or recent mean)
    Tt = train_D.shape[0]
    out = np.zeros((steps, n))
    for h in range(1, steps + 1):
        idx = Tt - 52 + (h - 1)  # same week last year? approximate
        if idx >= 0 and idx < Tt:
            out[h - 1] = train_D[idx]
        else:
            out[h - 1] = train_D[-4:].mean(axis=0)
    # fallback zeros
    out = np.nan_to_num(out, nan=0.0)
    out = np.clip(out, 0, None)
    return out


# Backtest: for each window start t0w (need at least 52 history), take 8 weeks demand, initial state zeros, forecast from history.
# Generate orders via reference policy rolling with forecasted demand as revealed?
# We'll do adaptive: forecast_fn uses seasonal naive based on history (actual revealed in simulation) -> but in backtest order_plan_from_policy reveals actual demand, so forecast updates with actuals. That's optimistic but okay for tuning policy params.
def make_forecast_fn(train_D0):
    # returns function for a specific starting history length L
    # We'll just use seasonal naive on revealed history
    def forecast_fn(week_index, state, history):
        if history is None:
            hist = train_D0.copy()
        else:
            h = np.asarray(history, dtype=float)
            if h.ndim == 1:
                h = h.reshape(1, -1)
            hist = np.vstack([train_D0, h])
        # forecast 3 horizons
        Tt = hist.shape[0]
        f = np.zeros((3, n))
        for k in range(3):
            idx = Tt - 52 + k  # hmm
            if 0 <= idx < Tt:
                f[k] = hist[idx]
            else:
                f[k] = hist[-4:].mean(axis=0) if Tt > 0 else 0.0
        f = np.clip(np.nan_to_num(f, nan=0.0), 0, None)
        return f

    return forecast_fn


# Simple: evaluate one window
results = []
for start in [100, 120, 140]:
    if start + 8 > T:
        continue
    train_D0 = D[:start]
    demand_window = D[start : start + 8]
    init = make_initial(n)
    # forecast_fn
    ff = make_forecast_fn(train_D0)
    # tune params by trying several
    for buffer in [0.5, 0.75, 1.0]:
        for scale in [1.0]:
            for projection in [1.0]:

                def policy(forecast, state):
                    return reference_orders(
                        forecast,
                        state,
                        scale=scale,
                        projection=projection,
                        buffer=buffer,
                    )

                out = order_plan_from_policy(demand_window, init, ff, policy=policy)
                results.append((start, buffer, scale, projection, out["score"]))
print(results)
print("avg by buffer")
from collections import defaultdict

agg = defaultdict(list)
for r in results:
    agg[r[1]].append(r[4])
for b, v in agg.items():
    print(b, np.mean(v))
