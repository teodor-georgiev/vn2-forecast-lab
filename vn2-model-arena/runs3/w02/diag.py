"""w02 diagnostics: forecast quality, quantile levels, order/score breakdown."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import tune  # noqa: E402
import policy as pol  # noqa: E402

start = int(sys.argv[1]) if len(sys.argv) > 1 else 100
pol.HISTORY_END = start
pol.GAME_DOY = tune.DOY[start : start + 8].tolist()
P = pol._prepare(start, tune.DOY[start : start + 8].tolist())
f, factor = pol._forecast(P, 0, None)

actual = tune.SALES[:, start : start + 8].T  # (8, n)
print("history weeks used:", P["T"], " game DOY:", P["gdoy"])
print("\nweekly totals: forecast vs actual")
for g in range(8):
    print(
        f"  w{g + 1}: fc {f[g].sum():9,.1f}  act {actual[g].sum():9,.1f}  ratio {actual[g].sum() / max(f[g].sum(), 1e-9):5.2f}"
    )

print("\nseries-level level estimate vs actual 8-week mean")
print("  mean level:", P["level"].mean(), " mean actual 8w:", actual.mean())

for W in (1, 2, 3):
    pool = P["pools"][W]
    for q in (0.5, 0.7, 0.8, 0.85, 0.9, 0.95):
        kap = np.quantile(pool, q, axis=1)
        print(
            f"  W={W} q={q}: kappa median {np.median(kap):5.2f}  p90 {np.quantile(kap, 0.9):5.2f}  max {kap.max():6.2f}  mean {kap.mean():5.2f}"
        )

# what does the actual 3-week aggregate look like vs the forecast?
W = 3
agg_act = actual[0:W].sum(axis=0)
prot = f[0:W].sum(axis=0)
kap = np.quantile(P["pools"][W], 0.85, axis=1)
S = prot * kap
print("\norder-up-to vs actual 3w demand (start window):")
print(
    "  S median",
    np.median(S),
    " actual 3w sum median",
    np.median(agg_act),
    " prot median",
    np.median(prot),
)
print("  ratio S/actual median", np.median(S / np.maximum(agg_act, 1e-9)))

res = tune.run(start)
print("\nweekly breakdown:")
for w in res["weekly"]:
    print("  ", w)
print("score", res["score"], "setup", res["setup"])


