# -*- coding: utf-8 -*-
"""Final submission: reference-deficit rule (scale=1.0, buf=0.75) on full-history LGBM forecast."""

import sys, time
import numpy as np, pandas as pd

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

model = train_model(W)
mu9 = recursive_forecast(model, W, 9)

SCALE, BUF = 1.0, 0.75
stt = init_state.copy()
ords = np.zeros((6, N))
for w in range(6):
    a = np.maximum(0, mu9[w + 1]) * SCALE
    b = np.maximum(0, mu9[w + 2]) * SCALE
    c = np.maximum(0, mu9[w + 3]) * SCALE
    oh, a1, a2 = stt.T
    proj = np.maximum(0.0, oh + a1 + a2 - (a + b))
    o = np.rint(np.clip(c + BUF * np.sqrt(c) - proj, 0, None))
    ords[w] = o
    end = np.maximum(0.0, (oh + a1) - np.maximum(0, mu9[w]) * SCALE)
    stt = np.column_stack([end, a2, o])

df = pd.DataFrame(
    {
        "series_id": np.repeat(sids, 6),
        "week": np.tile(np.arange(1, 7), N),
        "quantity": np.rint(ords.T).ravel().astype(int),
    }
)
assert len(df) == N * 6 == 3594
assert (df["quantity"] >= 0).all() and df["quantity"].notna().all()
assert set(df["week"]) == set(range(1, 7)) and df["series_id"].nunique() == 599
df.to_csv(RUN + r"\orders.csv", index=False)
print(
    "wrote",
    len(df),
    "rows; total units",
    int(df["quantity"].sum()),
    "per-week:",
    df.groupby("week")["quantity"].sum().tolist(),
    "elapsed",
    round(time.time() - t0, 1),
)
