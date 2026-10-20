import sys, time
import numpy as np
import pandas as pd

ARENA = "C:/Users/Toni/Documents/Default Project/vn2-model-arena"
sys.path.insert(0, ARENA + "/tools")
import inventory_sim as sim

sys.path.insert(0, "C:/Users/Toni/Documents/Default Project/vn2-model-arena/runs3/w04")
import helper_backtest as hb

sales = pd.read_csv(ARENA + "/data/sales.csv")
avail = pd.read_csv(ARENA + "/data/availability.csv")
initial = pd.read_csv(ARENA + "/data/initial_state.csv")
series_list = sorted(sales["series_id"].unique().tolist())
sid_to_idx = {s: i for i, s in enumerate(series_list)}
n = len(series_list)
weeks = sorted(sales["week"].unique().tolist())
T = len(weeks)
w_to_t = {w: t for t, w in enumerate(weeks)}
D = np.zeros((n, T))
A = np.zeros((n, T), dtype=bool)
for _, r in sales.iterrows():
    D[sid_to_idx[r["series_id"]], w_to_t[r["week"]]] = r["y"]
for _, r in avail.iterrows():
    A[sid_to_idx[r["series_id"]], w_to_t[r["week"]]] = bool(r["is_available"])
init_state = (
    initial.set_index("series_id")
    .loc[series_list][["on_hand", "arrival_w1", "arrival_w2"]]
    .to_numpy(float)
)

# corrected seasonal
Ycorr = D.copy()
woy = np.array([t % 52 for t in range(T)])
for i in range(n):
    for t in range(T):
        if not A[i, t]:
            mask = (woy == woy[t]) & A[i]
            if mask.sum() > 0:
                Ycorr[i, t] = D[i][mask].mean()
            elif A[i].sum() > 0:
                Ycorr[i, t] = D[i][A[i]].mean()


def forecast_matrix(week_index, state, revealed):
    n_series = state.shape[0]
    if n_series != n:
        return np.zeros((3, n_series))
    rev = np.asarray(revealed, dtype=float)
    if rev.size == 0:
        rev2 = np.zeros((0, n))
    else:
        rev2 = rev.reshape(-1, n)
    ext = np.hstack([Ycorr, rev2.T]) if rev2.shape[0] > 0 else Ycorr.copy()
    out = np.zeros((3, n_series))
    cur_len = T + week_index
    for hi, h in enumerate([1, 2, 3]):
        target_idx = cur_len - 1 + h
        tw = target_idx % 52
        # seasonal average over history for same woy
        for i in range(n_series):
            y = ext[i]
            hist_idx = [t for t in range(cur_len) if t % 52 == tw]
            if len(hist_idx) > 0:
                out[hi, i] = max(0.0, np.mean(y[hist_idx]))
            else:
                out[hi, i] = max(0.0, np.mean(y))
    return out


def decide(week_index, state, revealed):
    fc = forecast_matrix(week_index, state, revealed)
    a, b, c = [np.maximum(0.0, fc[i]) for i in range(3)]
    st = np.asarray(state, float)
    on_hand, arr1, arr2 = st.T
    buffer = 0.97
    projection = 1.0
    projected = np.maximum(0.0, on_hand + arr1 + arr2 - projection * (a + b))
    deficit = c + buffer * np.sqrt(np.maximum(c, 0)) - projected
    return np.clip(deficit, 0, None)


for start in [100, 130]:
    res = hb.backtest(decide, D, start_week=start, initial=init_state)
    print(f"start {start}: score {res['score']:.2f}")
