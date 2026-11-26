# Product-trend and state-projection research — 2026-11-26

## Decision
Do not promote this policy over the verified EUR 3,775.20 CatBoost/seasonal incumbent. This standalone, in-memory statistical experiment is useful evidence for a future ensemble and uncertainty-policy comparison, not a Linux training run or a new official competition submission. The best replay among evaluated, preselected new candidates was EUR 4,002.40; the historical winner cost EUR 4,262.40 on replay. The incumbent's cost was not re-fit in this experiment.

## Data and validation
- Immutable input source: teodor-georgiev/vn2-forecast-lab at 204d237ab25dc827f8fce14c4be6cf5a3727104b, vn2-from-scratch/data/raw.
- 599 Store x Product series; 297 distinct Product codes; 157 historical weeks plus 8 replay weeks.
- Align by Store/Product keys and treat stockout observations as missing in forecast features.
- Selection on original historical CV origins 63,72,81,90,99,108,117,126. Later historical origins 135,144 were excluded from this sweep's selection, but have been used in earlier experiments.
- Three synthetic opening states and demand stress 0%,10%,20%. Opening base is rounded trailing 13-week uncensored mean, copied into on-hand and both pipeline slots. For seeds 1/2, each opening array independently multiplied by a seeded LCG uniform(.6,1.4) draw (seed = 1000*seed + origin; LCG multiplier 1664525, increment 1013904223, modulus 2^32). This differs from the Python benchmark's RNG. Stress affects simulated demand only, not forecast inputs.
- Robust cost = fold mean + 0.45 * population std + 0.15 * (worst fold - mean).
- Organizer EUR 4,334.00 and seasonal incumbent EUR 3,899.00 reproduced; first two setup weeks EUR 913.80 excluded. Prefix orders invariant to current/future demand perturbations for all five selected configurations when forecasts are held fixed.

## Statistical forecasts and policy grid
Three point-forecast families:
1. Global multiplicative ISO-week seasonal factors, trailing 13-week deseasonalized mean.
2. Product-specific factors, from per-week mean of in-stock store observations within Product code. Divide product week means by product overall mean; shrink each week-of-year estimate toward the global factor using four pseudo-observations. Trailing 13-week deseasonalized mean.
3. Trend level = 0.60 * family 2 trailing 13-week level + 0.25 * trailing 6-week deseasonalized mean + 0.15 * preceding 20-week deseasonalized mean. Reseasonalize per horizon. Fall back to the 13-week level if a subwindow has no observations.

For each family, estimate per-series horizon-3 squared residuals from rolling forecasts at origins o-26 through o-3 inclusive, using only uncensored labels before o. Shrink variance toward the pooled variance with five pseudo-observations; floor sigma at 0.5. Freeze sigma for the six decision rounds.
State projection: E1=max(0,OH+A0-d1); E2=max(0,E1+A1-d2); order=max(0,round(multiplier*(d3+z*sigma)-E2)). Order before current demand reveal, with banker rounding and arrival t+2.
45 projected policies: three families x z in (0,0.25,0.5,0.75,1) x multiplier in (1,1.15,1.3). Nine coverage policies: three families x coverage in (1,1.1,1.2) with 0.3*sqrt(three-week demand) buffer. Two reference configurations; one duplicates seasonal coverage=1.1. Total evaluated configurations 56 (55 unique).
Freeze overall historical winner, best projected policy per forecast family and reference policies before replay.

## Verified results (EUR)
| Policy | CV mean | Robust CV | CV fill | Holdout cost | Holdout fill | Replay cost | Holding | Lost units | Replay fill |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Trend projection z=.5 mult=1.3 | 4971.18 | 5834.41 | 83.07% | 6059.30 | 82.22% | 4262.40 | 3333.40 | 929 | 91.17% |
| Global projection z=.5 mult=1.15 | 5082.57 | 5977.18 | 81.36% | 6326.10 | 80.13% | 4097.40 | 3078.40 | 1019 | 90.31% |
| Product projection z=.5 mult=1.15 | 5007.01 | 5882.77 | 80.14% | 6086.59 | 79.62% | 4002.40 | 2876.40 | 1126 | 89.29% |
| Seasonal incumbent | 5203.86 | 6211.53 | 78.97% | 6458.21 | 78.45% | 3899.00 | 2510.00 | 1389 | 86.79% |
| Organizer | 5363.29 | 6509.32 | 83.84% | 6732.49 | 82.54% | 4334.00 | 3256.00 | 1078 | 89.75% |

## Interpretation
Historical generalization and fill improve at the cost of excessive inventory in replay. The replay is known, so do not retune against its outcomes. Integrate product trend forecasts as a candidate feature or blend in a future independently reviewed ML lane; select parameters via chronological CV. The next experiment should compare the five-model causal Opus adaptation and existing CatBoost incumbent under the same scenarios. This report does not modify the pending workflow's pinned source or claim that ML training has been executed.
