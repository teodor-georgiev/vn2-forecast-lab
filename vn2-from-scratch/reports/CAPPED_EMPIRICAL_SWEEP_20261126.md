# Capped empirical replenishment sweep — 2026-11-26

## Results and decision
Retain the EUR 3,775.20 previously verified CatBoost ensemble. No new ML models were fitted. This is a standalone in-memory research comparison, not an official submission or VPS training run.

| Policy | Mean CV EUR | Robust CV EUR | CV fill | Holdout EUR | Holdout fill | Replay EUR | Holding EUR | Lost units | Replay fill |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Empirical q=.85/.85, mix=1, cap=1 | 5042.78 | 5913.96 | 79.56% | 6172.89 | 80.09% | 3806.00 | 2556.00 | 1250 | 88.11% |
| Hybrid q=.85/.75, mix=.5, cap=1.35 | 5059.75 | 5979.06 | 78.75% | 6272.23 | 78.90% | 3789.40 | 2417.40 | 1372 | 86.95% |
| Seasonal incumbent | 5203.86 | 6211.53 | 78.97% | 6458.21 | 78.45% | 3899.00 | 2510.00 | 1389 | 86.79% |
| Organizer | 5363.29 | 6509.32 | 83.84% | 6732.49 | 82.54% | 4334.00 | 3256.00 | 1078 | 89.75% |

## Reproduction contract
Use the same immutable dataset, key alignment, historical origins, stress scenarios, LCG opening state RNG, setup-week exclusion, and robust objective as PRODUCT_TREND_PROJECTION_20261126.md. The opening state generator differs from the Python benchmark. Reproduced organizer 4334, seasonal 3899, fixed setup 913.8. Eight original CV folds and two reserved later historical windows; those windows and competition replay are not globally untouched because of previous experiments.

At each decision cutoff t:
- Reuse the global multiplicative ISO-week seasonal factors fitted strictly before t; compute the mean b of uncensored deseasonalized sales over the preceding 13 weeks, or zero when absent.
- Take preceding 52 weeks of deseasonalized observations h, keeping stockout observations missing. Let mu be their valid mean. Scale historical demand paths by clip((b+mu)/(2*mu), 0.5, 2) if mu>0, else 1.
- Extract complete consecutive 3-week paths, reseasonalized with the three target week factors. If fewer than five complete paths, generate 50 deterministic triples from valid observations using indices j, 7j+3, 11j+5 modulo number of observations; all zero when none exist.
- Intermittent segment: at least 35% of valid historical observations are zero.
- For each path (x,y,z), compute arrival-week required order r=z-max(0,max(0,OH+A0-x)+A1-y). Use the segment quantile of r, with linear interpolation between adjacent sorted values.
- Point order P=max(0,1.1*(d1+d2+d3)+0.3*sqrt(d1+d2+d3)-OH-A0-A1).
- Final order before current demand reveal: banker-round(min((1-mix)*P+mix*max(0,quantile(r)),cap*P+1)), clipped at zero.
- Grid: intermittent and regular quantiles each in (.65,.75,.85); mix (.5,1); cap (1,1.35). 36 candidates plus organizer and seasonal baseline, 38 total.
- Freeze global robust-CV winner and best candidate per mix plus baselines before reporting replay.

## Next step
Compare this capped empirical policy against the same calibrated five-model ensemble using identical forecasts and scenarios. Do not select by competition replay cost. Full ML benchmark still awaits review of compute PR #1049. VPS scratch runtime probe failed with exit 1 and empty output (job a221a96a8a3044f76e8cc1cd54c87d59).
