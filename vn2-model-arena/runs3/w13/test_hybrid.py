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
woy = np.array([d.isocalendar().week for d in Y.columns])
n = y.shape[0]
T = y.shape[1]


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


# seasonal quantile by woy
def seasonal_quantile(q):
    sq = np.zeros((n, 53))
    for wi in range(1, 53):
        maskw = woy == wi
        if maskw.sum() == 0:
            sq[:, wi] = np.nanmean(np.where(a > 0.5, y, np.nan), axis=1)
            continue
        vals = y[:, maskw]
        avs = a[:, maskw]
        arr = np.where(avs > 0.5, vals, np.nan)
        sq[:, wi] = np.nanquantile(arr, q, axis=1)
    sq = np.where(
        np.isnan(sq), np.nanmean(np.where(a > 0.5, y, np.nan), axis=1)[:, None], sq
    )
    return np.maximum(sq, 0)


sq08 = seasonal_quantile(0.8)


def make_decide(q=0.8, buffer=0.0, projection=1.0, trend=True):
    sq = seasonal_quantile(q)

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
            if woyh.size:
                woyh = np.concatenate(
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
        Tcur = hist.shape[1]
        # recent mean and seasonal mean for trend
        recent = np.nanmean(np.where(av[:, -13:] > 0.5, hist[:, -13:], np.nan), axis=1)
        recent = np.where(
            np.isnan(recent),
            series_mean if False else np.nanmean(np.where(a > 0.5, y, np.nan), axis=1),
            recent,
        )
        # seasonal mean same woy
        fc = []
        for hh in range(1, 4):
            pos = Tcur + hh - 1
            if woyh.size > pos:
                tw = int(woyh[pos])
            else:
                tw = int(woy[pos % len(woy)])
            if tw < 1 or tw > 52:
                tw = 1
            base = sq[:, tw].copy()
            if trend:
                # seasonal mean for this woy
                maskw = woy == tw
                if maskw.sum() > 0:
                    seas_mean = np.nanmean(
                        np.where(a[:, maskw] > 0.5, y[:, maskw], np.nan), axis=1
                    )
                    seas_mean = np.where(np.isnan(seas_mean), 1.0, seas_mean)
                    ratio = recent / (seas_mean + 1e-6)
                    ratio = np.clip(ratio, 0.5, 2.0)
                    base = base * ratio
            fc.append(np.maximum(base, 0))
        fc = np.array(fc)
        a1, b, c = [f for f in fc]
        on_hand, arr1, arr2 = state.T
        projected = np.maximum(0.0, on_hand + arr1 + arr2 - projection * (a1 + b))
        deficit = c + buffer * np.sqrt(np.maximum(c, 0)) - projected
        return np.clip(deficit, 0, None)

    return decide


series_mean_global = np.nanmean(np.where(a > 0.5, y, np.nan), axis=1)
# monkey patch series_mean name used above
import builtins


# We'll just replace in function by closure variable
# redefine make_decide with series_mean captured
def make_decide2(q=0.8, buffer=0.0, projection=1.0, trend=True):
    sq = seasonal_quantile(q)
    sm = series_mean_global

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
            if woyh.size:
                woyh = np.concatenate(
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
        Tcur = hist.shape[1]
        recent = np.nanmean(np.where(av[:, -13:] > 0.5, hist[:, -13:], np.nan), axis=1)
        recent = np.where(np.isnan(recent), sm, recent)
        fc = []
        for hh in range(1, 4):
            pos = Tcur + hh - 1
            tw = int(woyh[pos]) if woyh.size > pos else int(woy[pos % len(woy)])
            if tw < 1 or tw > 52:
                tw = 1
            base = sq[:, tw].copy()
            if trend:
                maskw = woy == tw
                if maskw.sum() > 0:
                    seas_mean = np.nanmean(
                        np.where(a[:, maskw] > 0.5, y[:, maskw], np.nan), axis=1
                    )
                    seas_mean = np.where(np.isnan(seas_mean), 1.0, seas_mean)
                    ratio = recent / (seas_mean + 1e-6)
                    ratio = np.clip(ratio, 0.5, 2.0)
                    base = base * ratio
            fc.append(np.maximum(base, 0))
        fc = np.array(fc)
        a1, b, c = [f for f in fc]
        on_hand, arr1, arr2 = state.T
        projected = np.maximum(0.0, on_hand + arr1 + arr2 - projection * (a1 + b))
        deficit = c + buffer * np.sqrt(np.maximum(c, 0)) - projected
        return np.clip(deficit, 0, None)

    return decide


for trend in [True, False]:
    for buffer in [0.0, 0.5]:
        decide = make_decide2(q=0.8, buffer=buffer, projection=1.0, trend=trend)
        scs = [backtest(decide, y, start_week=sw, initial=None) for sw in [100, 130]]
        print(
            f"trend={trend} buf={buffer} scores={[round(s) for s in scs]} avg={np.mean(scs):.0f}"
        )
