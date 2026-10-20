"""Diagnose why verify.py replay scores differ from tune.py backtests."""

import numpy as np
import pandas as pd
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent.parent / "tools"))
import inventory_sim as sim

import mybt
import tune as T

Y, INIT = T.Y, T.INIT

# reference rule via mybt (tuner path)
ref = dict(scale=1.0, buffer=0.75, projection=1.0, base="raw", adapt=0.0, trend=False)


def ref_decide(w, state, revealed, H=Y):
    avail = np.concatenate([H, revealed.T], axis=1) if w else H
    a = np.maximum(avail[:, -13:].mean(axis=1), 0.0)
    return T.ref_orders([a, a, a], state, 1.0, 0.75, 1.0)


for s in (145, 100):
    r1 = mybt.backtest(
        lambda w, st, rv: ref_decide(w, st, rv, Y), Y, start_week=s, initial=INIT
    )
    print("mybt  start", s, round(r1["score"], 1))

# now verify path: replay loop + sim.simulate
sales = pd.read_csv(HERE.parent.parent / "data" / "sales.csv")
weeks = sorted(sales["week"].unique())
series = sorted(sales["series_id"].unique())
sidx = {x: i for i, x in enumerate(series)}
widx = {x: i for i, x in enumerate(weeks)}
n, t = len(series), len(weeks)
y2 = np.zeros((n, t))
y2[sales["series_id"].map(sidx).to_numpy(), sales["week"].map(widx).to_numpy()] = sales[
    "y"
].to_numpy()
print("Y from npy == y2 from csv:", np.allclose(Y, y2))
init2 = pd.read_csv(HERE.parent.parent / "data" / "initial_state.csv")[
    ["on_hand", "arrival_w1", "arrival_w2"]
].to_numpy(float)
print("INIT equal init2:", np.allclose(INIT, init2), INIT.shape, init2.shape)

for s in (145, 100):
    dem = y2[:, s : s + 8].T
    state = init2.copy()
    orders = np.zeros((6, n))
    for w in range(8):
        if w < 6:
            d = ref_decide(w, state.copy(), np.zeros((w, n)), y2[:, :s])
            orders[w] = np.rint(np.maximum(0.0, d))
        on_hand, arr1, arr2 = state.T.copy()
        end = np.maximum(0.0, on_hand + arr1 - dem[w])
        state = np.column_stack((end, arr2, orders[w] if w < 6 else np.zeros(n)))
    res = sim.simulate(dem, init2, orders)
    print("verify-path start", s, round(res["score"], 1))
    # compare orders with mybt's
    r = mybt.backtest(
        lambda w, st, rv: ref_decide(w, st, rv, Y), Y, start_week=s, initial=INIT
    )
    print("  max |order diff|:", float(np.abs(r["orders"] - orders).max()))
