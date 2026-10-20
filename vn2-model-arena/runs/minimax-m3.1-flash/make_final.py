"""FINAL: anchored seasonal forecast + static newsvendor plan -> orders.csv."""

from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
from core import load, plan_news, sigma_from
import inventory_sim as sim

RUN = Path(
    r"C:\Users\Toni\Documents\Default Project\vn2-model-arena\runs\minimax-m3.1-flash"
)
H, series, I = load()
T_HIST, N = H.shape


def fc_anchor(H, T, alpha=0.5, recent=4, anchor_w=0.5):
    """8-week demand forecast: year-ago seasonal shape (level-rescaled) blended with
    the recent 4-week mean, then the 8-week level pulled 50% toward the trailing
    13-week mean (robust to a one-off spike in the year-ago window)."""
    hist = H[:T]
    cur_base = hist[-13:].mean(0)
    mu_rec = hist[-recent:].mean(0)
    out = np.zeros((8, N))
    for k in range(1, 9):
        idx = T + k - 52
        if idx - 12 >= 0:
            base = hist[idx - 12 : idx + 1].mean(0)
            ratio = np.clip(cur_base / np.maximum(base, 0.3), 0.4, 2.5)
            seas = hist[idx] * ratio
            if idx - 52 - 12 >= 0:
                b2 = hist[idx - 52 - 12 : idx - 52 + 1].mean(0)
                r2 = np.clip(cur_base / np.maximum(b2, 0.3), 0.4, 2.5)
                seas = 0.7 * seas + 0.3 * (hist[idx - 52] * r2)
        else:
            seas = hist[-13:].mean(0)
        out[k - 1] = alpha * mu_rec + (1 - alpha) * seas
    lvl = out.mean(0)
    tgt = anchor_w * cur_base + (1 - anchor_w) * lvl
    out = out * (tgt / np.maximum(lvl, 0.2))
    return out


Z, SIG_WIN = 0.5, 26
fc = fc_anchor(H, T_HIST, alpha=0.5, recent=4, anchor_w=0.5)
sig = np.repeat(sigma_from(H, T_HIST, N, SIG_WIN)[None, :], 8, axis=0)
Q = plan_news(fc, sig, I, z=Z)

rows = [
    (sid, w + 1, float(max(0.0, Q[w, i])))
    for w in range(6)
    for i, sid in enumerate(series)
]
df = pd.DataFrame(rows, columns=["series_id", "week", "quantity"])
df.to_csv(RUN / "orders.csv", index=False)

# ---------------- validation
chk = pd.read_csv(RUN / "orders.csv")
piv = chk.pivot(index="series_id", columns="week", values="quantity").loc[list(series)]
assert list(chk.columns) == ["series_id", "week", "quantity"]
assert len(chk) == 3594 and piv.shape == (599, 6)
assert not piv.isna().any().any() and (piv.to_numpy() >= 0).all()
assert sorted(chk.week.unique()) == [1, 2, 3, 4, 5, 6]
print(
    "OK rows=%d series=%d weeks=%s"
    % (len(chk), chk.series_id.nunique(), sorted(chk.week.unique()))
)
print(
    "ordered per order-week:",
    piv.sum().to_numpy().round(1),
    "total",
    round(chk.quantity.sum()),
)
print("forecast per game week :", fc.sum(1).round(1))
print("trailing 13w mean level:", round(H[-13:].mean(0).sum()))
print(
    "initial state: on_hand %.0f arr1 %.0f arr2 %.0f"
    % (I[:, 0].sum(), I[:, 1].sum(), I[:, 2].sum())
)
# expected leftover into week 3 under the forecast
E1 = np.maximum(0.0, I[:, 0] - fc[0])
E3 = np.maximum(0.0, E1 + I[:, 2] - fc[1])
print("projected leftover entering week 3: %.0f units" % E3.sum())

# ---------------- sanity: cost of the plan under hypothetical demand paths
for name, dem in [
    ("forecast", fc),
    ("forecast x1.15", fc * 1.15),
    ("forecast x0.85", fc * 0.85),
]:
    r = sim.simulate(dem, I, Q)
    print("%-14s score=%8.0f setup=%8.0f  (weeks3-8)" % (name, r["score"], r["setup"]))
