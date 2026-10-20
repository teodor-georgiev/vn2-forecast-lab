"""Diagnostics + candidate comparison + final orders.csv write."""

from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
import core
from core import (
    load,
    fc_mean13,
    fc_blend,
    fc_blend2,
    plan_news,
    plan_ref,
    plan_cover,
    sigma_from,
    FORECASTS,
)
import inventory_sim as sim

RUN = Path(
    r"C:\Users\Toni\Documents\Default Project\vn2-model-arena\runs\minimax-m3.1-flash"
)
H, series, I = load()
T_HIST, N = H.shape
R_ON, R_AR = 1.45, 1.10
WINDOWS = sorted(set(list(range(60, 150, 6)) + [104, 52, 140, 146]))
WINDOWS = [w for w in WINDOWS if w + 8 <= T_HIST]


def proxy_state(mu):
    return np.column_stack([np.rint(R_ON * mu), np.zeros(len(mu)), np.rint(R_AR * mu)])


def plan_sqrt(F, sig, I0, b=0.75, shrink=1.0):
    """target = mu + b*sqrt(mu), safety proportional to the mean (Poisson-like)."""
    n = F.shape[1]
    on_hand, arr1, arr2 = I0.T
    Ew = np.zeros((9, n))
    Ew[0] = on_hand
    arr = {1: np.array(arr1, float), 2: np.array(arr2, float)}
    Q = np.zeros((6, n))
    for t in range(1, 9):
        supply = Ew[t - 1] + arr.get(t, np.zeros(n))
        Ew[t] = np.maximum(0.0, supply - F[t - 1])
        if t <= 6:
            tw = t + 2
            proj = np.maximum(0.0, Ew[t] + arr.get(t + 1, np.zeros(n)) - F[t])
            mu = F[tw - 1]
            target = mu + b * np.sqrt(np.maximum(mu, 0.0))
            Q[t - 1] = np.maximum(0.0, target - proj)
            arr[tw] = arr.get(tw, np.zeros(n)) + Q[t - 1]
    return Q


def plan_hybrid(F, sig, I0, b=0.75, z=0.0):
    n = F.shape[1]
    on_hand, arr1, arr2 = I0.T
    Ew = np.zeros((9, n))
    Ew[0] = on_hand
    arr = {1: np.array(arr1, float), 2: np.array(arr2, float)}
    Q = np.zeros((6, n))
    for t in range(1, 9):
        supply = Ew[t - 1] + arr.get(t, np.zeros(n))
        Ew[t] = np.maximum(0.0, supply - F[t - 1])
        if t <= 6:
            tw = t + 2
            proj = np.maximum(0.0, Ew[t] + arr.get(t + 1, np.zeros(n)) - F[t])
            mu = F[tw - 1]
            target = mu + b * np.sqrt(np.maximum(mu, 0.0)) + z * sig[tw - 1]
            Q[t - 1] = np.maximum(0.0, target - proj)
            arr[tw] = arr.get(tw, np.zeros(n)) + Q[t - 1]
    return Q


def evaluate(planner, fcname, fc_kw, p_kw, windows=WINDOWS, sig_win=26, detail=False):
    rows = []
    for T in windows:
        fc = FORECASTS[fcname](H, T, **fc_kw)
        sig = np.repeat(sigma_from(H, T, N, sig_win)[None, :], 8, axis=0)
        mu = H[max(0, T - 13) : T].mean(0)
        I0 = proxy_state(mu)
        Q = planner(fc, sig, I0, **p_kw)
        res = sim.simulate(H[T : T + 8], I0, Q)
        dem = H[T + 2 : T + 8].sum()
        rows.append(
            dict(
                T=T,
                score=res["score"],
                dem=dem,
                short=sum(w["shortage"] for w in res["weekly"][2:]),
                hold=sum(w["holding"] for w in res["weekly"][2:]),
                ordered=sum(w["ordered"] for w in res["weekly"]),
            )
        )
    return pd.DataFrame(rows)


base_df = evaluate(lambda fc, sig, I0: plan_ref(H, 0, I0, fc[0]), "mean13", {}, {})
print("=== baseline ref_static ===")
print(base_df[["T", "dem", "score", "short", "hold", "ordered"]].to_string(index=False))

CANDS = {}
for fcname, fckw in [
    ("mean13", {}),
    ("blend", dict(alpha=0.5, recent=8)),
    ("blend", dict(alpha=0.5, recent=4)),
    ("blend2", dict(alpha=0.5, recent=4)),
    ("blend2", dict(alpha=0.35, recent=8)),
]:
    for b in [0.0, 0.5, 0.75, 1.0, 1.5]:
        CANDS[(fcname, str(fckw), f"sqrt b={b}")] = (plan_sqrt, fcname, fckw, dict(b=b))
    for z in [0.0, 0.3, 0.5, 0.8]:
        CANDS[(fcname, str(fckw), f"news z={z}")] = (plan_news, fcname, fckw, dict(z=z))

out = []
for key, (pl, fcname, fckw, pkw) in CANDS.items():
    df = evaluate(pl, fcname, fckw, pkw)
    ratio = df["score"].to_numpy() / base_df["score"].to_numpy()
    out.append(
        dict(
            fc=fcname,
            fckw=str(fckw),
            pol=key[2],
            total=df["score"].sum(),
            medratio=np.median(ratio),
            trim=sum(np.sort(df["score"].to_numpy())[:12]),
            short=df["short"].sum(),
            hold=df["hold"].sum(),
        )
    )
res_df = pd.DataFrame(out).sort_values("medratio")
print("\n=== candidates sorted by median ratio vs baseline ===")
print(res_df.head(20).to_string(index=False))
print("\n=== sorted by trimmed total ===")
print(res_df.sort_values("trim").head(12).to_string(index=False))
res_df.to_csv(RUN / "cand_grid.csv", index=False)
