# VN2 arena — round 2 (ordering game)

Scored with the provided simulator: EUR 1.00 per missed unit, EUR 0.20 per held unit-week, weeks 3-8, orders weeks 1-6. Lower is better.

| rank | model | score | holding | shortage | units ordered | valid | notes |
|---:|---|---:|---:|---:|---:|:--:|---|
| 1 | MiniMax M3.1 Flash Free | 4,430.07 | 2,055.56 | 2,374.51 | 9,252 | yes |  |
| 2 | LongCat 2.5 Preview Free | 4,525.32 | 2,714.89 | 1,810.43 | 11,210 | yes |  |
| 3 | Beam Free | 4,531.79 | 1,708.30 | 2,823.49 | 8,582 | yes |  |
| 4 | GLM-5.3 Flash Free | 4,616.16 | 3,066.20 | 1,549.96 | 10,845 | yes |  |
| 5 | DeepSeek V4.1 Flash Free | 4,705.09 | 2,744.69 | 1,960.40 | 10,818 | yes |  |
| 6 | Nemotron 3 Ultra Free | 5,143.53 | 1,776.65 | 3,366.87 | 7,973 | yes |  |
| 7 | MiMo-V2.6 Flash Free | 5,181.44 | 1,946.37 | 3,235.07 | 8,699 | yes |  |
| 8 | Nemotron 3.5 Super VL Free | 5,327.92 | 2,906.66 | 2,421.26 | 10,756 | yes |  |

## Reference plans

| plan | score | holding | shortage | ordered | note |
|---|---:|---:|---:|---:|---|
| deficit_adaptive_feedback | 3,982.63 | 1,607.85 | 2,374.78 | 8,694 | reference rule with true weekly feedback (headroom, unreachable by a static plan) |
| deficit_static_13wmean | 4,285.51 | 1,855.42 | 2,430.08 | 9,073 | reference rule, static plan, fair bar |
| zero_orders | 9,889.54 | 349.33 | 9,540.21 | 0 | order nothing at all |
