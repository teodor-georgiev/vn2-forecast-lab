"""Final config search; writes orders.csv with the winner."""

from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
from core import load, FORECASTS, plan_news, sigma_from, plan_ref
import inventory_sim as sim

WINDOWS = [
    w
    for w in sorted(set(list(range(60, 150, 6)) + [104, 52, 140, 146]))
    if w + 8 <= 157
]


def proxy_state(mu):
    return np.column_stack([np.rint(1.45 * mu), np.zeros(len(mu)), np.rint(1.10 * mu)])


def plan_sqrt(F, sig, I0, b=0.75):
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


RUN = Path(
    r"C:\Users\Toni\Documents\Default Project\vn2-model-arena\runs\minimax-m3.1-flash"
)
H, series, I = load()
T_HIST, N = H.shape


def evaluate(fcname, fc_kw, z=None, b=None, sig_win=26):
    rows = []
    for T in WINDOWS:
        fc = FORECASTS[fcname](H, T, **fc_kw)
        sig = np.repeat(sigma_from(H, T, N, sig_win)[None, :], 8, axis=0)
        I0 = proxy_state(H[max(0, T - 13) : T].mean(0))
        if b is None:
            Q = plan_news(fc, sig, I0, z=z)
        else:
            Q = plan_sqrt(fc, sig, I0, b=b)
        res = sim.simulate(H[T : T + 8], I0, Q)
        rows.append((T, res["score"]))
    df = pd.DataFrame(rows, columns=["T", "score"])
    return df


# baseline per window
base = []
for T in WINDOWS:
    mu = H[max(0, T - 13) : T].mean(0)
    I0 = proxy_state(mu)
    from core import plan_ref

    base.append(sim.simulate(H[T : T + 8], I0, plan_ref(H, T, I0, mu))["score"])
base = np.array(base)

cands = []
for fcname, fckw in [
    ("blend2", dict(alpha=0.5, recent=4)),
    ("blend2", dict(alpha=0.35, recent=4)),
    ("blend2", dict(alpha=0.65, recent=4)),
    ("blend2", dict(alpha=0.5, recent=8)),
    ("blend", dict(alpha=0.5, recent=4)),
    ("blend", dict(alpha=0.5, recent=8)),
    ("blend", dict(alpha=0.7, recent=4)),
    ("mean13", {}),
]:
    for z in [0.3, 0.5, 0.7, 1.0]:
        for sw in [13, 26, 52]:
            df = evaluate(fcname, fckw, z=z, sig_win=sw)
            r = df["score"].to_numpy() / base
            cands.append(
                dict(
                    fc=fcname,
                    fckw=str(fckw),
                    pol=f"z={z}",
                    sw=sw,
                    total=df["score"].sum(),
                    med=np.median(r),
                    trim=np.sort(df["score"].to_numpy())[:12].sum(),
                    worst=r.max(),
                )
            )
    for b in [0.5, 0.75, 1.0]:
        df = evaluate(fcname, fckw, b=b)
        r = df["score"].to_numpy() / base
        cands.append(
            dict(
                fc=fcname,
                fckw=str(fckw),
                pol=f"sqrt b={b}",
                sw=26,
                total=df["score"].sum(),
                med=np.median(r),
                trim=np.sort(df["score"].to_numpy())[:12].sum(),
                worst=r.max(),
            )
        )

c = pd.DataFrame(cands)
c["rank"] = c["med"] + 0.15 * (c["worst"] - 1.0)
print(c.sort_values("med").head(20).to_string(index=False))
print()
print(c.sort_values("rank").head(20).to_string(index=False))
c.to_csv(RUN / "final_grid.csv", index=False)
