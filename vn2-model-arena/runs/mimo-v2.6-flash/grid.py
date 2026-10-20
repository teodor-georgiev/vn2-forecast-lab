# -*- coding: utf-8 -*-
"""Grid the reference rule (scale x buf) at two historical cutoffs, true simulated score."""

import sys, time
import numpy as np, pandas as pd

sys.path.insert(0, r"C:\Users\Toni\Documents\Default Project\vn2-model-arena\tools")
from inventory_sim import simulate

RUN = r"C:\Users\Toni\Documents\Default Project\vn2-model-arena\runs\mimo-v2.6-flash"
DATA = r"C:\Users\Toni\Documents\Default Project\vn2-model-arena\data"
t0 = time.time()

src = open(RUN + r"\final.py").read()
head = src.split("# ---- OOS multi-step residual ratios")[0]
exec(head)

init = pd.read_csv(DATA + r"\initial_state.csv")
init_state = (
    init.set_index("series_id")
    .loc[sids][["on_hand", "arrival_w1", "arrival_w2"]]
    .values.astype(float)
)


def reference_plan(mu9v, buf):
    stt = init_state.copy()
    ords = np.zeros((6, N))
    for w in range(6):
        a = np.maximum(0, mu9v[w + 1])
        b = np.maximum(0, mu9v[w + 2])
        c = np.maximum(0, mu9v[w + 3])
        oh, a1, a2 = stt.T
        proj = np.maximum(0.0, oh + a1 + a2 - (a + b))
        o = np.rint(np.clip(c + buf * np.sqrt(c) - proj, 0, None))
        ords[w] = o
        end = np.maximum(0.0, (oh + a1) - np.maximum(0, mu9v[w]))
        stt = np.column_stack([end, a2, o])
    return ords


# bias estimate from OOS windows
def oos_ratio(c2):
    m2 = train_model(c2)
    fc2 = recursive_forecast(m2, c2, 8)
    return np.clip((Yf[c2 : c2 + 8] + 0.5) / (fc2 + 0.5), 0.05, 10.0)


summary = []
for cut, rescuts in [(W - 8, (W - 32, W - 24)), (W - 16, (W - 40, W - 32))]:
    rr = np.vstack([oos_ratio(c) for c in rescuts])
    mu_b = float(np.exp(np.log(rr).mean()))
    print("cut", weeks[cut], "bias ratio", round(mu_b, 3), flush=True)
    mcut = train_model(cut)
    mu9 = np.vstack([recursive_forecast(mcut, cut, 9)])
    dem_true = Yf[cut : cut + 8]
    for scale in (0.85, 0.95, 1.0, 1.05):
        for buf in (0.75, 1.0, 1.25, 1.5):
            o = reference_plan(mu9 * scale, buf)
            s = simulate(dem_true, init_state, o)["score"]
            summary.append((cut, scale, buf, s))
            print(f"  scale {scale} buf {buf} -> {s:.1f}", flush=True)

df = pd.DataFrame(summary, columns=["cut", "scale", "buf", "score"])
print(df.groupby(["scale", "buf"])["score"].mean().sort_values())
print("elapsed", time.time() - t0)
