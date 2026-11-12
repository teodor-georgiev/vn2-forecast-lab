# VN2 from-scratch — results

**Validity: retrospective competition replay.** The game weeks 1-8 labels were published before
this project existed, so nothing below is blind or official.

| metric | value |
| --- | ---: |
| Organizer benchmark, weeks 3-8 | EUR 4,334.00 |
| **This solution, weeks 3-8** | **EUR 3,899.00** |
| &nbsp;&nbsp;holding | EUR 2,510.00 |
| &nbsp;&nbsp;shortage | EUR 1,389.00 |
| Fill rate (scored weeks) | 86.79% |

## Selected configuration (CV only)

- seasonal weight **1.0**, coverage **1.1**, buffer **0.3**
- robust CV cost **6209.1936** (mean 5200.3, worst 8293.53) over 60 candidates

## Per-fold CV cost

| origin | cost | min | max |
| --- | ---: | ---: | ---: |
| 63 | 4,914.53 | 4,460.20 | 5,423.00 |
| 72 | 4,358.03 | 4,096.31 | 4,678.44 |
| 81 | 8,293.53 | 7,685.40 | 8,976.97 |
| 90 | 4,889.13 | 4,634.00 | 5,136.89 |
| 99 | 5,432.26 | 4,706.60 | 6,159.08 |
| 108 | 4,640.93 | 4,009.09 | 5,342.48 |
| 117 | 4,418.79 | 3,968.00 | 4,965.52 |
| 126 | 4,655.18 | 4,296.40 | 5,077.24 |

Fold costs are measured on a harsher regime than the replay: synthetic
opening states and +0/+10/+20% demand stress. They are therefore not
comparable in level to the EUR 3,899.00 figure, which uses the real opening
state and realised demand. The fold costs select the configuration; they
do not estimate the replay score.

## Forecast quality vs inventory cost

Forecast error and inventory cost are reported separately and must not be
conflated: a lower WAPE does not imply a lower cost, because cost depends
on the holding/shortage trade-off at 0.20/1.00 EUR per unit, not on
forecast accuracy alone.
