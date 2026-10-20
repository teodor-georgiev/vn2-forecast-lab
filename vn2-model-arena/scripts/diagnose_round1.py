"""Sanity diagnostics for round-1 submissions: shape of the predictions, not just WAPE.

Checks that a weak score is a weak method rather than a misaligned or degenerate file:
zero share, mean level vs truth, per-horizon WAPE, and correlation with the truth.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ARENA = Path(r"C:\Users\Toni\Documents\Default Project\vn2-model-arena")
PRIVATE = Path(r"C:\Users\Toni\AppData\Local\Temp\opencode\vn2-arena-private")

truth = pd.read_csv(PRIVATE / "answer_panel.csv")
truth["series_id"] = truth["series_id"].astype(str)
t = truth.pivot(index="series_id", columns="horizon", values="y_true").sort_index()
t = t.loc[:, range(1, 9)]

rows = []
for run in sorted((ARENA / "runs").iterdir()):
    path = run / "predictions.csv"
    if not path.exists():
        continue
    p = pd.read_csv(path)
    p["series_id"] = p["series_id"].astype(str)
    m = p.pivot(index="series_id", columns="horizon", values="y_hat").sort_index()
    m = m.loc[t.index, range(1, 9)]
    y = t.to_numpy(float)
    pv = m.to_numpy(float)
    per_h = [float(np.abs(pv[:, h] - y[:, h]).sum() / y[:, h].sum()) for h in range(8)]
    rows.append(
        {
            "model": run.name,
            "pred_mean": pv.mean(),
            "true_mean": y.mean(),
            "zero_share": float((pv < 1e-9).mean()),
            "true_zero_share": float((y < 1e-9).mean()),
            "corr": float(np.corrcoef(pv.ravel(), y.ravel())[0, 1]),
            "wape_h1": per_h[0],
            "wape_h8": per_h[7],
            "wape_spread": max(per_h) - min(per_h),
        }
    )

out = pd.DataFrame(rows).sort_values("wape_h1")
print(out.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
