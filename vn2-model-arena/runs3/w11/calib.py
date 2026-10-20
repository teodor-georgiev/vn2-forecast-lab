"""Calibration: score the manager's reference rule (13-week mean forecasts) at cutoffs."""

import sys
import numpy as np
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "tools"))
import hb2
from inventory_sim import reference_orders
from diag import get_data

Y, init = get_data()


def make_policy(window=13, buffer=0.75, scale=1.0, projection=1.0):
    from policy import _load

    C = _load()

    def decide(w, state, revealed):
        d = C["d"][:, : decide.base]
        if w:
            d = np.column_stack((d, np.asarray(revealed).T))
        k = min(window, d.shape[1])
        f = d[:, -k:].mean(axis=1) * scale
        return reference_orders([f, f, f], state, buffer=buffer, projection=projection)

    decide.base = None
    return decide


if __name__ == "__main__":
    cuts = [53, 79, 105, 130, 149]
    for buf in (0.75, 1.0, 1.5):
        for win in (13, 26):
            row = []
            for c in cuts:
                pol = make_policy(window=win, buffer=buf)
                pol.base = c
                r = hb2.backtest(pol, Y, start_week=c, initial=init)
                row.append(f"{r['score']:7.0f}")
            print(f"ref win={win:3d} buf={buf}: " + " ".join(row))
    print("cutoffs:", cuts)
