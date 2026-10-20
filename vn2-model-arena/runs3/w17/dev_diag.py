from __future__ import annotations
import numpy as np, hb_local as hb
from dev_bt import S, INIT, truncate, N

import policy

H = 130
truncate(H)
res = hb.backtest(policy.decide, S, H, initial=INIT)
for r in res["weekly"]:
    print(
        f"w{r['week']} ordered {r['ordered']:7d} hold {r['holding']:8.0f} short {r['shortage']:8.0f}"
    )
dem = S[:, H : H + 8]
print("total demand 8w", dem.sum(), "total ordered", res["orders"].sum())

prof = policy._profile(policy._CFG)
print(
    "kappa quantiles",
    np.round(np.quantile(prof["kappa"], [0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1]), 2),
)
print(
    "level_eff quantiles",
    np.round(np.quantile(prof["level_eff"], [0, 0.25, 0.5, 0.75, 0.9, 0.99, 1]), 2),
)
print(
    "series mean quantiles",
    np.round(np.quantile(S[:, :H].mean(1), [0, 0.25, 0.5, 0.75, 0.9, 0.99, 1]), 2),
)

# top offenders: holding over the game
state = INIT.copy()
hold_by_series = np.zeros(N)
short_by_series = np.zeros(N)
orders = res["orders"]
for w in range(8):
    order = orders[w] if w < 6 else np.zeros(N)
    onhand, arr1, arr2 = state.T.copy()
    supply = onhand + arr1
    end = np.maximum(0, supply - dem[:, w])
    missed = np.maximum(0, dem[:, w] - supply)
    hold_by_series += 0.2 * end
    short_by_series += missed
    state = np.column_stack((end, arr2, order))
idx = np.argsort(-hold_by_series)[:15]
print("top holding series: lev, kap, mean13, hold, short")
mn13 = S[:, H - 13 : H].mean(1)
for i in idx:
    print(
        f"  {prof['level_eff'][i]:8.2f} {prof['kappa'][i]:6.2f} {mn13[i]:6.2f} hold {hold_by_series[i]:8.1f} short {short_by_series[i]:7.1f}"
    )
print(
    "holding share of top 15:",
    round(hold_by_series[idx].sum() / max(hold_by_series.sum(), 1), 3),
)
print(
    "total hold",
    round(hold_by_series.sum()),
    "total short",
    round(short_by_series.sum()),
)
