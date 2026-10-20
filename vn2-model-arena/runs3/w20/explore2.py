"""Which past-level estimator best predicts UNCENSORED future demand?"""

import numpy as np
import pandas as pd
from pathlib import Path

DATA = Path(r"C:\Users\Toni\Documents\Default Project\vn2-model-arena\data")
sales = pd.read_csv(DATA / "sales.csv")
avail = pd.read_csv(DATA / "availability.csv")
init = pd.read_csv(DATA / "initial_state.csv")
order = init["series_id"].tolist()
piv = sales.pivot_table(index="series_id", columns="week", values="y").reindex(
    index=order
)
Y = piv.to_numpy(dtype=float)
A = (
    avail.pivot_table(index="series_id", columns="week", values="is_available")
    .reindex(index=order, columns=list(piv.columns))
    .to_numpy(dtype=bool)
)
T, N = Y.shape[1], Y.shape[0]

rows = []
for i in range(N):
    y, a = Y[i], A[i]
    for w in range(52, T - 8, 2):
        hist = y[w - 26 : w]
        hista = a[w - 26 : w]
        tgt = y[w + 1]
        tA = a[w + 1]
        if not tA:  # only evaluate against uncensored targets
            continue
        m_all = hist.mean()
        m_av = hist[hista].mean() if hista.any() else m_all
        n_c = (~hista).sum()
        rows.append((i, w, tgt, m_all, m_av, n_c / 26, m_all * 1.13))
E = pd.DataFrame(rows, columns=["i", "w", "tgt", "m_all", "m_av", "pc", "m_up"])
lvl = E.groupby("i")["tgt"].transform("mean") + 1e-9
for c in ["m_all", "m_av", "m_up"]:
    print(c, "relMAE %.4f" % ((E[c] - E["tgt"]).abs().div(lvl).mean()))
# tune uplift on censored weeks only
sub = E[E.pc > 0.05]
lvl2 = sub.groupby("i")["tgt"].transform("mean") + 1e-9
best = None
for u in np.arange(1.0, 2.01, 0.05):
    v = (sub.m_av * u - sub.tgt).abs().div(lvl2).mean()
    if best is None or v < best[1]:
        best = (u, v)
    print("u=%.2f relMAE %.4f" % (u, v))
print("best u", best)
# shrink: combine all-week mean and censored-adjusted
for lam in [0.3, 0.5, 0.7]:
    f = lam * E.m_all + (1 - lam) * E.m_av * best[0]
    print("mix %.1f relMAE %.4f" % (lam, ((f - E.tgt).abs().div(lvl).mean())))
