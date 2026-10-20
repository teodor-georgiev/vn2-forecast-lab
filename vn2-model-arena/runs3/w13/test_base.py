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
    return {"score": float(score), "weekly": weekly, "orders": orders}


def make_decide(k=13, buffer=0.75, projection=1.0, scale=1.0, use_avail=False):
    def decide(week_index, state, revealed):
        state = np.asarray(state, float)
        revealed = np.asarray(revealed, float)
        if n == state.shape[0]:
            hist = y.copy()
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
        # impute unavailable weeks with 0? For now mean over available weeks only
        if hist.shape[1] > 0:
            # 13-week mean skipping unavailable (set unavailable to nan)
            h = hist.copy()
            # mark unavailable as nan
            if av.shape == h.shape:
                h = np.where(av > 0.5, h, np.nan)
            # rolling last k
            window = h[:, -k:]
            m = np.nanmean(window, axis=1)
            m = np.where(np.isnan(m), 0.0, m)
        else:
            m = np.zeros(state.shape[0])
        m = np.maximum(m, 0)
        fc = np.tile(m, (3, 1))
        a1, b, c = [np.maximum(0.0, f) * scale for f in fc]
        on_hand, arr1, arr2 = np.asarray(state, float).T
        projected = np.maximum(0.0, on_hand + arr1 + arr2 - projection * (a1 + b))
        deficit = c + buffer * np.sqrt(np.maximum(c, 0)) - projected
        return np.clip(deficit, 0, None)

    return decide


for k in [4, 8, 13, 26, 52]:
    for buf in [0.75]:
        decide = make_decide(k=k, buffer=buf)
        scs = [
            backtest(decide, y, start_week=sw, initial=st0)["score"]
            for sw in [100, 130]
        ]
        print(
            f"k={k} buf={buf} scores={[round(s) for s in scs]} avg={np.mean(scs):.0f}"
        )
