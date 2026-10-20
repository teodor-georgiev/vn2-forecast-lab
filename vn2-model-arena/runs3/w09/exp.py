import sys, time, numpy as np, pandas as pd

sys.path.insert(0, r"C:\Users\Toni\Documents\Default Project\vn2-model-arena\tools")
sys.path.insert(0, r"C:\Users\Toni\Documents\Default Project\vn2-model-arena\runs3\w09")
import helper_backtest as hb

# Load data
ids = pd.read_csv(
    r"C:\Users\Toni\Documents\Default Project\vn2-model-arena\data\initial_state.csv"
)["series_id"].tolist()
sales = pd.read_csv(
    r"C:\Users\Toni\Documents\Default Project\vn2-model-arena\data\sales.csv"
)
avail = pd.read_csv(
    r"C:\Users\Toni\Documents\Default Project\vn2-model-arena\data\availability.csv"
)
wks = sorted(sales.week.unique())
wi = {w: i for i, w in enumerate(wks)}
S = np.zeros((len(ids), len(wks)))
A = np.zeros((len(ids), len(wks)))
id2i = {sid: i for i, sid in enumerate(ids)}
for sid, g in sales.groupby("series_id"):
    i = id2i[sid]
    for _, r in g.iterrows():
        S[i, wi[r["week"]]] = r["y"]
for sid, g in avail.groupby("series_id"):
    i = id2i[sid]
    for _, r in g.iterrows():
        A[i, wi[r["week"]]] = 1 if r["is_available"] else 0
INIT = (
    pd.read_csv(
        r"C:\Users\Toni\Documents\Default Project\vn2-model-arena\data\initial_state.csv"
    )
    .set_index("series_id")
    .loc[ids][["on_hand", "arrival_w1", "arrival_w2"]]
    .values.astype(float)
)


# Censoring correction: impute avail=0 weeks
def correct(S, A, method="mean_season"):
    Y = S.copy()
    # week-of-year index (157 weeks ~ 3 years). Use 52-week seasonality.
    T = S.shape[1]
    if method == "none":
        return Y
    if method == "mean_all":
        # replace zeros on censored weeks with series mean of available weeks
        ser_mean = np.where(
            A.sum(axis=1) > 0, (S * A).sum(axis=1) / np.maximum(A.sum(axis=1), 1), 0
        )
        mask = A == 0
        Y[mask] = np.broadcast_to(ser_mean[:, None], S.shape)[mask]
        return Y
    if method == "seasonal_mean":
        # for each censored week t, use mean of available weeks with same (t mod 52)
        for i in range(S.shape[0]):
            for t in range(T):
                if A[i, t] == 0:
                    same = [k for k in range(T) if k % 52 == t % 52 and A[i, k] == 1]
                    if same:
                        Y[i, t] = S[i, same].mean()
                    else:
                        Y[i, t] = S[i, A[i] == 1].mean() if A[i].sum() > 0 else 0
        return Y
    if method == "highq":
        # impute with 75th percentile of available-week demands (same season)
        for i in range(S.shape[0]):
            for t in range(T):
                if A[i, t] == 0:
                    same = [k for k in range(T) if k % 52 == t % 52 and A[i, k] == 1]
                    if same:
                        Y[i, t] = np.percentile(S[i, same], 75)
                    else:
                        av = S[i, A[i] == 1]
                        Y[i, t] = np.percentile(av, 75) if len(av) > 0 else 0
        return Y
    raise ValueError(method)


# Global corrected history
CORR_METHOD = "seasonal_mean"
SH = correct(S, A, CORR_METHOD)


# EWMA forecast
def ewma_forecast(history, spans=(4, 8, 13, 26, 52)):
    # history: (n, T) array (already includes revealed appended by caller)
    # returns level per series for each span? We'll compute combined level.
    n, T = history.shape
    levels = []
    for sp in spans:
        alpha = 2.0 / (sp + 1)
        # vectorized EWMA
        # state
        level = np.zeros(n)
        # initialize with first
        level = history[:, 0].copy()
        for t in range(1, T):
            level = alpha * history[:, t] + (1 - alpha) * level
        levels.append(level.copy())
    L = np.stack(levels, axis=0)  # (nspans, n)
    # combine: median across spans? or mean
    comb = np.median(L, axis=0)
    return comb, L


def make_decide(
    corr_method,
    spans,
    use_seasonal,
    seasonal_w,
    qparam,
    projection,
    buffer,
    scale,
):
    SH = correct(S, A, corr_method)

    def decide(week_index, state, revealed):
        state = np.asarray(state, float)
        n = state.shape[0]
        revealed = np.asarray(revealed, float)
        if revealed.size == 0:
            rev = np.zeros((0, n))
        else:
            rev = revealed.reshape(-1, n) if revealed.ndim == 2 else revealed
        w = rev.shape[0]
        # Build extended history per series if n matches
        if n == S.shape[0]:
            # static corrected history + revealed (actual)
            hist = SH.copy()  # (n,157)
            if w > 0:
                # append revealed
                hist = np.column_stack([hist, rev.T])  # (n, 157+w)
        else:
            # fallback: use revealed only + zeros
            if w > 0:
                hist = rev.T.copy()
            else:
                hist = np.zeros((n, 1))
        T = hist.shape[1]
        # point forecast per horizon h=1..3 (constant EWMA level)
        comb_level, levels = ewma_forecast(hist, spans)
        # seasonal naive: for horizon h, use value at t-52+h
        # Build seasonal forecast array (n,) for each horizon
        fc = np.zeros((3, n))
        for hi in range(3):
            h = hi + 1
            # EWMA constant
            base = comb_level.copy()
            if use_seasonal and T >= 52:
                # seasonal naive: demand at (last_index + h) approx hist[:, -52 + h] if available
                idx = (
                    T - 52 + h
                )  # 0-based in hist for target? target position T-1+h; seasonal lag 52 -> T-1+h-52 = T-53+h
                idx = T - 52 + (h - 1)
                if 0 <= idx < T:
                    seas = hist[:, idx].copy()
                else:
                    seas = base.copy()
                # blend
                fc[hi] = (1 - seasonal_w) * base + seasonal_w * seas
            else:
                fc[hi] = base
        fc = np.maximum(fc, 0)
        # residual quantile calibration: scale up by factor?
        fc = fc * scale
        # reference_orders
        a, b, c = [np.maximum(0.0, fc[i]) for i in range(3)]
        on_hand, arr1, arr2 = state.T
        projected = np.maximum(0.0, on_hand + arr1 + arr2 - projection * (a + b))
        deficit = c + buffer * np.sqrt(np.maximum(c, 0)) - projected
        order = np.clip(deficit, 0.0, None)
        # residual quantile calibration additive?
        # add per-horizon quantile residual
        return order

    return decide


# Test harness
def adaptive_backtest(decide, history, start, initial):
    history = np.asarray(history, float)
    n = history.shape[0]
    demand = history[:, start : start + 8].T
    state = np.asarray(initial, float).copy()
    revealed = []
    weekly = []
    orders = np.zeros((6, n))
    for w in range(8):
        if w < 6:
            rev_arr = (
                np.array(revealed, dtype=float).reshape(w, n)
                if w > 0
                else np.zeros((0, n))
            )
            decision = decide(w, state.copy(), rev_arr)
            orders[w] = np.rint(np.maximum(0.0, np.asarray(decision, float)))
        order = orders[w] if w < 6 else np.zeros(n)
        on_hand, arr1, arr2 = state.T.copy()
        supply = on_hand + arr1
        end = np.maximum(0.0, supply - demand[w])
        missed = np.maximum(0.0, demand[w] - supply)
        weekly.append((1.0 * missed.sum()) + 0.2 * end.sum())
        state = np.column_stack((end, arr2, order))
        revealed.append(demand[w].copy())
    score = sum(weekly[2:])
    return score, weekly, orders


if __name__ == "__main__":
    # quick baseline
    decide = make_decide(
        "seasonal_mean", (4, 8, 13, 26, 52), True, 0.5, None, 1.0, 0.75, 1.0
    )
    for st in [100, 130]:
        sc, _, _ = adaptive_backtest(decide, S, st, INIT)
        print(f"start {st} raw: {sc:.1f}")
    print("done")
