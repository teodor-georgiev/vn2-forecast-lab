"""Compare tune2.make_decide orders vs policy.decide orders window by window."""

import numpy as np
import importlib.util
from pathlib import Path

import tune as T
import tune2 as T2

HERE = Path(__file__).parent
spec = importlib.util.spec_from_file_location("pol", HERE / "policy.py")
pol = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pol)

cfg = dict(scale=0.95, buffer=0.0, projection=1.35, base="corr", adapt=6.0, trend=False)
Y, YC, INIT = T.Y, T.YC, T.INIT

for s in (145, 100):
    d_tune = T2.make_decide(YC[:, :s], cfg)
    state = INIT.copy()
    st2 = INIT.copy()
    print(f"--- window {s} ---")
    for w in range(6):
        rev = Y[:, s : s + w].T
        o1 = np.asarray(d_tune(w, state.copy(), rev), dtype=float)
        o2 = np.asarray(pol.decide(w, st2.copy(), rev), dtype=float)
        print(
            f"  w{w}: maxdiff {np.abs(o1 - o2).max():.1f}  sum1 {o1.sum():.0f} sum2 {o2.sum():.0f}"
        )
        # advance both states with their own orders (identical if orders equal)
        for st, o in ((state, o1), (st2, o2)):
            on_hand, arr1, arr2 = st.T.copy()
            end = np.maximum(0.0, on_hand + arr1 - Y[:, s + w])
            st[:] = np.column_stack((end, arr2, np.rint(np.maximum(0.0, o))))
