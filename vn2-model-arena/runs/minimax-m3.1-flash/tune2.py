"""Anchored seasonal forecast (shape from year-ago, level from trailing mean) + final pick."""

from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
from core import load, FORECASTS, plan_news, plan_ref, sigma_from
import inventory_sim as sim

RUN = Path(
    r"C:\Users\Toni\Documents\Default Project\vn2-model-arena\runs\minimax-m3.1-flash"
)
H, series, I = load()
T_HIST, N = H.shape
WINDOWS = [
    w
    for w in sorted(set(list(range(60, 150, 6)) + [104, 52, 140, 146]))
    if w + 8 <= 157
]


def proxy_state(mu):
    return np.column_stack([np.rint(1.45 * mu), np.zeros(len(mu)), np.rint(1.10 * mu)])


def fc_anchor(H, T, alpha=0.5, recent=4, anchor_w=1.0, **kw):
    """Seasonal shape (year-ago, rescaled) blended with recent level, then the 8-week
    mean is pulled back toward the trailing 13-week mean by anchor_w."""
    hist = H[:T]
    cur_base = hist[-13:].mean(0)
    mu_rec = hist[-recent:].mean(0)
    out = np.zeros((8, H.shape[1]))
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


FORECASTS["anchor"] = fc_anchor


def run(fcname, fc_kw, z, sw=26, per_window=False):
    rows = []
    for T in WINDOWS:
        fc = FORECASTS[fcname](H, T, **fc_kw)
        sig = np.repeat(sigma_from(H, T, N, sw)[None, :], 8, axis=0)
        I0 = proxy_state(H[max(0, T - 13) : T].mean(0))
        Q = plan_news(fc, sig, I0, z=z)
        res = sim.simulate(H[T : T + 8], I0, Q)
        rows.append((T, res["score"]))
    df = pd.DataFrame(rows, columns=["T", "score"])
    return df


base = []
for T in WINDOWS:
    mu = H[max(0, T - 13) : T].mean(0)
    I0 = proxy_state(mu)
    base.append(sim.simulate(H[T : T + 8], I0, plan_ref(H, T, I0, mu))["score"])
base = np.array(base)
print("baseline total %.0f" % base.sum())

cfgs = [
    ("mean13", {}, 0.3, 26),
    ("mean13", {}, 0.3, 52),
    ("mean13", {}, 0.5, 26),
    ("blend2", dict(alpha=0.5, recent=4), 0.5, 26),
    ("anchor", dict(alpha=0.5, recent=4, anchor_w=1.0), 0.3, 26),
    ("anchor", dict(alpha=0.5, recent=4, anchor_w=1.0), 0.5, 26),
    ("anchor", dict(alpha=0.5, recent=4, anchor_w=0.5), 0.5, 26),
    ("anchor", dict(alpha=0.35, recent=4, anchor_w=1.0), 0.5, 26),
    ("anchor", dict(alpha=0.65, recent=4, anchor_w=1.0), 0.5, 26),
]
tab = {}
for fcname, fckw, z, sw in cfgs:
    df = run(fcname, fckw, z, sw)
    r = df["score"].to_numpy() / base
    tab[(fcname, str(fckw), z, sw)] = (df["score"].to_numpy(), r)
    print(
        "%-12s %-42s z=%.1f sw=%d  total=%.0f med=%.3f worst=%.3f trim=%.0f"
        % (
            fcname,
            str(fckw),
            z,
            sw,
            df["score"].sum(),
            np.median(r),
            r.max(),
            np.sort(df["score"].to_numpy())[:12].sum(),
        )
    )

print("\nper-window ratios (rows=window, cols=config):")
keys = list(tab)
print(
    "T     " + "  ".join("%-8s" % (k[0][:4] + k[2].__str__() + str(k[3])) for k in keys)
)
for i, T in enumerate(WINDOWS):
    print("%-5d " % T + "  ".join("%-8.3f" % tab[k][1][i] for k in keys))
