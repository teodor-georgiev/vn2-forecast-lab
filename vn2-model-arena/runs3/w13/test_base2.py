import numpy as np, pandas as pd, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
D = Path(__file__).resolve().parents[2] / "data"
sales = pd.read_csv(D / "sales.csv", parse_dates=["week"])
avail = pd.read_csv(D / "availability.csv", parse_dates=["week"])
Y = sales.pivot(index="series_id", columns="week", values="y")
A = avail.pivot(index="series_id", columns="week", values="is_available")
y = Y.values.astype(float)
a = A.values.astype(float)
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


# impute unavailable weeks with series mean
yc = np.where(a > 0.5, y, np.nan)
series_mean = np.nanmean(yc, axis=1)
y_imp = np.where(a > 0.5, y, series_mean[:, None])


def make_decide(k=13, buffer=0.75, use_imp=False, censor_mean=False):
    def decide(week_index, state, revealed):
        state = np.asarray(state, float)
        revealed = np.asarray(revealed, float)
        if n == state.shape[0]:
            hist = (y_imp if use_imp else y).copy()
            av = a.copy()
        else:
            hist = np.zeros((state.shape[0], 0))
            av = np.ones((state.shape[0], 0))
        if revealed.size:
            hist = np.column_stack([hist, revealed.T]) if hist.size else revealed.T
            av = (
                np.column_stack([av, np.ones((state.shape[0], revealed.shape[0]))])
                if av.size
                else np.ones((state.shape[0], revealed.shape[0]))
            )
        # 13-week mean skipping unavailable
        h = hist.copy()
        if av.shape == h.shape:
            h = np.where(av > 0.5, h, np.nan)
        m = np.nanmean(h[:, -k:], axis=1)
        m = np.where(np.isnan(m), series_mean if n == state.shape[0] else 0.0, m)
        m = np.maximum(m, 0)
        fc = np.tile(m, (3, 1))
        a1, b, c = [np.maximum(0.0, f) for f in fc]
        on_hand, arr1, arr2 = np.asarray(state, float).T
        projected = np.maximum(0.0, on_hand + arr1 + arr2 - 1.0 * (a1 + b))
        deficit = c + buffer * np.sqrt(np.maximum(c, 0)) - projected
        return np.clip(deficit, 0, None)

    return decide


for use_imp in [False, True]:
    for init in [None]:
        decide = make_decide(k=13, buffer=0.75, use_imp=use_imp)
        for sw in [100, 130]:
            sc = backtest(decide, y, start_week=sw, initial=init)
            print(f"use_imp={use_imp} init=zeros sw={sw} score={sc:.0f}")
