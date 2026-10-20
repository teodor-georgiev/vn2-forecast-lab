"""Write final orders.csv (and validate)."""

from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
from core import load, FORECASTS, plan_news, sigma_from
import inventory_sim as sim

RUN = Path(
    r"C:\Users\Toni\Documents\Default Project\vn2-model-arena\runs\minimax-m3.1-flash"
)
H, series, I = load()
T_HIST, N = H.shape


def build(fcname="blend2", fc_kw=dict(alpha=0.5, recent=4), z=0.5, sig_win=26):
    T = T_HIST
    fc = FORECASTS[fcname](H, T, **fc_kw)
    sig = np.repeat(sigma_from(H, T, N, sig_win)[None, :], 8, axis=0)
    Q = plan_news(fc, sig, I, z=z)
    return fc, Q


def write_orders(Q, path=RUN / "orders.csv"):
    rows = []
    for w in range(6):
        for i, sid in enumerate(series):
            rows.append((sid, w + 1, float(Q[w, i])))
    df = pd.DataFrame(rows, columns=["series_id", "week", "quantity"])
    df["quantity"] = df["quantity"].clip(lower=0.0)
    df.to_csv(path, index=False)
    return df


if __name__ == "__main__":
    import sys

    kw = dict(
        fcname=sys.argv[1] if len(sys.argv) > 1 else "blend2",
        fc_kw=eval(sys.argv[2]) if len(sys.argv) > 2 else dict(alpha=0.5, recent=4),
        z=float(sys.argv[3]) if len(sys.argv) > 3 else 0.5,
    )
    fc, Q = build(**kw)
    df = write_orders(Q)
    # ---- validation
    piv = df.pivot(index="series_id", columns="week", values="quantity").loc[series]
    assert piv.shape == (599, 6), piv.shape
    assert not piv.isna().any().any()
    assert (piv.to_numpy() >= 0).all()
    print("rows", len(df), "cols", list(df.columns))
    print(
        "total units ordered",
        df.quantity.sum(),
        " per week:",
        piv.sum().to_numpy().round(1),
    )
    print("weeks", sorted(df.week.unique()), "series", df.series_id.nunique())
    print("forecast mean per game week:", fc.sum(1).round(1))
    print(
        "real initial state totals: on_hand %.0f arr1 %.0f arr2 %.0f"
        % (I[:, 0].sum(), I[:, 1].sum(), I[:, 2].sum())
    )
