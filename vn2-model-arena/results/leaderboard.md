# VN2 arena — model leaderboard

Primary metric: holdout WAPE (lower is better). Truth: 599 series x 8 weeks, 4792 cells, total demand 13866.6.

| rank | model | WAPE | PBIAS | MAE | valid | notes |
|---:|---|---:|---:|---:|:--:|---|
| 1 | DeepSeek V4.1 Flash Free | 0.5378 | -0.137 | 1.556 | yes |  |
| 2 | Beam Free | 0.5518 | -0.026 | 1.597 | yes |  |
| 3 | GLM-5.3 Flash Free | 0.5534 | -0.130 | 1.601 | yes |  |
| 4 | Nemotron 3.5 Super VL Free | 0.5560 | -0.141 | 1.609 | yes |  |
| 5 | Nemotron 3 Ultra Free | 0.5691 | +0.011 | 1.647 | yes |  |
| 6 | MiMo-V2.6 Flash Free | 0.5776 | -0.067 | 1.671 | yes |  |
| 7 | LongCat 2.5 Preview Free | 0.5932 | +0.139 | 1.717 | yes |  |
| 8 | MiniMax M3.1 Flash Free | 0.8002 | +0.179 | 2.315 | yes |  |

## Reference points

| baseline | WAPE | PBIAS | MAE |
|---|---:|---:|---:|
| median_13 | 0.5238 | -0.239 | 1.516 |
| mean_13 | 0.5444 | -0.096 | 1.575 |
| last_value | 0.6613 | -0.143 | 1.914 |
| seasonal_52 | 0.8021 | +0.171 | 2.321 |
| zero | 1.0000 | -1.000 | 2.894 |
