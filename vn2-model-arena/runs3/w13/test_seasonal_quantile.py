import numpy as np, pandas as pd, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
D = Path(__file__).resolve().parents[2] / "data"
sales = pd.read_csv(D / "sales.csv", parse_dates=["week"])
avail = pd.read_csv(D / "availability.csv", parse_dates=["week"])
init = pd.read_csv(D / "initial_state.csv")
Y = sales.pivot(index="series_id", columns="week", values="y")
A = avail.pivot(index="series_id", columns="week", values="is_available")
ids = list(Y.index)
y = Y.values.astype(float)
a = A.values.astype(float)
woy = np.array([d.isocalendar().week for d in Y.columns])
init = init.set_index("series_id").loc[ids]
st0 = init[["on_hand", "arrival_w1", "arrival_w2"]].values.astype(float)
n = y.shape[0]


def backtest(decide, history, start_week=0, weeks=8, initial=None):
    history = np.asarray(history, float)
    n = history.shape[0]
    if initial is None:
        state = np.zeros((n, 3))
    else:
        state = np.asarray(initial, float).copy()
    demand = history[:, start_week : start_week + weeks].T
    revealed = []
    orders = np.zeros((6, n))
    weekly = []
    for w in range(weeks):
        if w < 6:
            decision = decide(
                w,
                state.copy(),
                np.array(revealed, dtype=float).reshape(w, n)
                if revealed
                else np.empty((0, n)),
            )
            orders[w] = np.rint(np.maximum(0.0, np.asarray(decision, float)))
        order = orders[w] if w < 6 else np.zeros(n)
        on_hand, arr1, arr2 = state.T.copy()
        supply = on_hand + arr1
        end = np.maximum(0.0, supply - demand[w])
        missed = np.maximum(0.0, demand[w] - supply)
        weekly.append(
            {"week": w + 1, "holding": 0.2 * end.sum(), "shortage": 1.0 * missed.sum()}
        )
        state = np.column_stack((end, arr2, order))
        revealed.append(demand[w])
    score = sum(r["holding"] + r["shortage"] for r in weekly[2:])
    return float(score)


def make_policy(q=0.8, buffer=0.0, projection=1.0):
    # precompute seasonal quantiles per series per woy
    # censored: use only available weeks
    # seasonal quantile matrix (n, 53) for given q
    sq = np.zeros((n, 53))
    for wi in range(1, 53):
        maskw = woy == wi
        if maskw.sum() == 0:
            sq[:, wi] = np.nanmean(y, axis=1)
            continue
        # values across weeks with this woy, only available
        vals = y[:, maskw]  # (n, m)
        avs = a[:, maskw]
        # compute quantile ignoring unavailable (set to nan)
        arr = np.where(avs > 0.5, vals, np.nan)
        # nanquantile
        sq[:, wi] = np.nanquantile(arr, q, axis=1)
    # fallback for nan
    sq = np.where(np.isnan(sq), np.nanmean(y, axis=1)[:, None], sq)
    sq = np.maximum(sq, 0)

    def decide(week_index, state, revealed):
        state = np.asarray(state, float)
        revealed = np.asarray(revealed, float)
        if n == state.shape[0]:
            hist = y.copy()
            av = a.copy()
            woyh = woy.copy()
        else:
            hist = np.zeros((state.shape[0], 0))
            av = np.ones((state.shape[0], 0))
            woyh = np.array([])
        if revealed.size:
            hist = np.column_stack([hist, revealed.T]) if hist.size else revealed.T
            av = (
                np.column_stack([av, np.ones((state.shape[0], revealed.shape[0]))])
                if av.size
                else np.ones((state.shape[0], revealed.shape[0]))
            )
            # extend woyh
            woyh = (
                np.concatenate(
                    [
                        woyh,
                        np.array(
                            [
                                (woyh[-1] + i) % 52 + 1
                                for i in range(1, revealed.shape[0] + 1)
                            ]
                        ),
                    ]
                )
                if woyh.size
                else np.array([])
            )
        # forecast h1..h3 by seasonal quantile of target woy
        T = hist.shape[1]
        fc = []
        for hh in range(1, 4):
            pos = T + hh - 1
            if woyh.size > pos:
                tw = woyh[pos]
            else:
                tw = woy[pos % len(woy)]
            tw = int(tw)
            if tw < 1 or tw > 52:
                tw = 1
            m = sq[:, tw]
            fc.append(m)
        fc = np.array(fc)
        # ordering with buffer 0 (quantile already safety)
        a1, b, c = [f for f in fc]
        on_hand, arr1, arr2 = np.asarray(state, float).T
        projected = np.maximum(0.0, on_hand + arr1 + arr2 - projection * (a1 + b))
        deficit = c - projected  # quantile target already high
        # add small buffer sqrt
        deficit = deficit + 0.75 * np.sqrt(np.maximum(c, 0)) * 0.0
        return np.clip(deficit, 0, None)

    return decide


for q in [0.7, 0.75, 0.8, 0.85, 0.9]:
    decide = make_policy(q=q)
    scs = [backtest(decide, y, start_week=sw, initial=st0) for sw in [100, 130]]
    print(f"q={q}: scores={[round(s) for s in scs]} avg={np.mean(scs):.0f}")
