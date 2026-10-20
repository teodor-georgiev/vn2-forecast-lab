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

# censored demand: set unavailable to nan
yc = np.where(a > 0.5, y, np.nan)


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


def make_policy(fc_func, buffer=0.75, projection=1.0, scale=1.0):
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
            # extend woy
            if woyh.size:
                last = woyh[-1]
                ext = []
                d = last
                for i in range(revealed.shape[0]):
                    # add 7 days approximated by +1 week index mod 52
                    pass
                woyh = np.concatenate(
                    [
                        woyh,
                        np.array(
                            [
                                (woyh[-1] + i + 1 - 1) % 52 + 1
                                for i in range(revealed.shape[0])
                            ]
                        ),
                    ]
                )  # rough
        # compute forecast h1,h2,h3 via fc_func
        fc = fc_func(hist, av, woyh, week_index)  # returns (3,n)
        fc = np.asarray(fc, float)
        a1, b, c = [np.maximum(0.0, f) * scale for f in fc]
        on_hand, arr1, arr2 = np.asarray(state, float).T
        projected = np.maximum(0.0, on_hand + arr1 + arr2 - projection * (a1 + b))
        deficit = c + buffer * np.sqrt(np.maximum(c, 0)) - projected
        return np.clip(deficit, 0, None)

    return decide


# Forecast functions: return (3,n)
def fc_mean(hist, av, woyh, wi, k=13):
    h = hist.copy()
    if av.shape == h.shape:
        h = np.where(av > 0.5, h, np.nan)
    m = np.nanmean(h[:, -k:], axis=1)
    m = np.where(np.isnan(m), 0.0, m)
    m = np.maximum(m, 0)
    return np.tile(m, (3, 1))


def fc_seasonal(hist, av, woyh, wi):
    # same week-of-year mean over history
    # current week index in extended history for h1: len(hist) (0-based next week)
    T = hist.shape[1]
    out = []
    for hh in range(1, 4):
        idx = (
            T + hh - 1
        )  # target week position in extended timeline if future? need woy
        # target woy: use woyh extended if available else woy of (idx mod 52)
        if woyh.size > idx:
            tw = woyh[idx]
        else:
            tw = woy[idx % len(woy)]
        # find historical weeks with same woy
        mask = (woyh == tw) if woyh.size == T else (woy == tw)
        # historical demand matrix aligned: use yc? We'll compute from hist with avail
        h = hist.copy()
        if av.shape == h.shape:
            h = np.where(av > 0.5, h, np.nan)
        vals = h[:, mask] if mask.sum() > 0 else h
        m = np.nanmean(vals, axis=1)
        m = np.where(np.isnan(m), 0.0, m)
        m = np.maximum(m, 0)
        out.append(m)
    return np.array(out)


def fc_seasonal_recent(hist, av, woyh, wi):
    # seasonal naive: last year's same week (t-52) for each horizon
    T = hist.shape[1]
    out = []
    for hh in range(1, 4):
        pos = T + hh - 1
        lag = 52
        src = pos - lag
        if src >= 0 and src < T:
            vals = hist[:, src]
            # if unavailable, use nanmean of same woy
            if av.shape == hist.shape and av[:, src].mean() < 0.5:
                # fallback mean of same woy
                tw = woyh[src] if woyh.size > src else woy[src % len(woy)]
                mask = woyh == tw
                h = hist.copy()
                if av.shape == h.shape:
                    h = np.where(av > 0.5, h, np.nan)
                vals = np.nanmean(h[:, mask], axis=1)
        else:
            vals = np.zeros(hist.shape[0])
        vals = np.where(np.isnan(vals), 0.0, vals)
        vals = np.maximum(vals, 0)
        out.append(vals)
    return np.array(out)


candidates = {
    "mean13": lambda h, a, w, wi: fc_mean(h, a, w, wi, k=13),
    "seasonal_woy": lambda h, a, w, wi: fc_seasonal(h, a, w, wi),
    "seasonal_naive": lambda h, a, w, wi: fc_seasonal_recent(h, a, w, wi),
}
for name, ff in candidates.items():
    decide = make_policy(ff)
    scs = [backtest(decide, y, start_week=sw, initial=st0) for sw in [100, 130]]
    print(f"{name}: scores={[round(s) for s in scs]} avg={np.mean(scs):.0f}")
