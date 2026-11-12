# VN2 From-Scratch — Frozen Interface Contract

**Status: FROZEN.** Every worker builds against this document. Do not change a
signature without re-running the integration gate.

## Ground truth

- Competition: VN2 Inventory Planning Challenge (DataSource.ai).
- 599 store x product series, weekly history 2021-04-12 .. 2024-04-08 (157 weeks).
- 8 revealed game weeks 2024-04-15 .. 2024-06-03.
- 6 ordering decisions. Cost = holding (EUR 0.20/unit/week on ENDING inventory,
  no cost while in transit) + shortage (EUR 1.00 per LOST sale, no backorder).
- **Only game weeks 3-8 are scored.** Weeks 1-2 are excluded.

## Lead time (the single most important derivation)

An order placed at the end of game week X is received at the start of game week
X+3 and is sellable in game week X+3. Round `d` (1-based) therefore arrives in
game week `d+2`.

0-indexed arrival schedule over game weeks `t = 0..7`:

```
arr[0] = pipe_w1        # "In Transit W+1" already in flight at game start
arr[1] = pipe_w2        # "In Transit W+2"
arr[t] = orders[t - 2]  # for t = 2..7; orders has shape (6, N_SERIES)
```

**Invariant:** game weeks 1 and 2 costs depend only on the initial state and on
demand, never on any order. Use this as an assertion.

## Data columns (verified against the downloaded files)

| File | Shape | Notes |
| --- | --- | --- |
| `Week 0 - 2024-04-08 - Initial State.csv` | (599, 12) | `Start Inventory`, `In Transit W+1` are **all zero**. Game start uses `End Inventory` as opening on-hand, `In Transit W+2` as `pipe_w2`. |
| `Week 0 - 2024-04-08 - Sales.csv` | (599, 159) | Store, Product + 157 date columns. |
| `Week 0 - In Stock.csv` | (599, 167) | Store, Product + 165 booleans through 2024-06-03; ~10.6% False (out of stock). |
| `Week 0 - Master.csv` | (599, 8) | ProductGroup, Division, Department, DepartmentGroup, StoreFormat, Format. Note: this file has **no date in its name**, unlike the others. |
| `Week 0 - Submission Template.csv` | (599, 3) | Store, Product, `0`. Defines the required submission row order. |
| `Week k - <date> - Sales.csv` | (599, 159+k) | Same wide format, one extra appended date column. |

Row order `(Store, Product)` is identical across every file and equals the
submission template order. Assert this.

## Module signatures

```python
# config.py  -- ALREADY WRITTEN, do not modify
N_SERIES, HISTORY_WEEKS, GAME_WEEKS, N_DECISIONS
HOLDING_COST, SHORTAGE_COST, SCORE_WEEKS_START, SCORE_WEEKS_END
CV_ORIGINS, CV_WINDOW, CV_INITIAL_STATE_SEEDS, DEMAND_STRESS
GATE_MIN_ROBUST_IMPROVEMENT, SEED, verify_data_hashes()

# data.py
@dataclass(frozen=True)
class Dataset:
    keys: np.ndarray           # (599, 2) int64 [store, product], submission order
    history_dates: list[str]   # 157
    history: np.ndarray        # (599, 157) float64 observed weekly sales
    in_stock: np.ndarray       # (599, 157) bool availability aligned to history
    in_stock_full: np.ndarray  # (599, 165) bool availability through 2024-06-03
    game_dates: list[str]      # 8
    game_sales: np.ndarray     # (599, 8) float64 EXPOSED labels, game weeks 1..8
    on_hand0: np.ndarray       # (599,) float64 opening on-hand (End Inventory)
    pipe_w1: np.ndarray        # (599,) float64 In Transit W+1 (all zeros)
    pipe_w2: np.ndarray        # (599,) float64 In Transit W+2
    master: pd.DataFrame       # index-aligned to keys, 6 hierarchy columns

def load_dataset(data_dir: Path | None = None) -> Dataset: ...
def assert_contract(ds: Dataset) -> None: ...   # raises AssertionError on any mismatch

# simulator.py
@dataclass
class SimResult:
    total_cost: float; holding_cost: float; shortage_cost: float
    weekly_total: np.ndarray; weekly_holding: np.ndarray; weekly_shortage: np.ndarray  # (8,)
    scored_cost: float; scored_holding: float; scored_shortage: float   # weeks 3-8
    fill_rate_scored: float; demand_scored: float; lost_units_scored: float; held_units_scored: float

def simulate(orders, demand, on_hand0, pipe_w1, pipe_w2) -> SimResult: ...
    # orders  : (6, 599) float64, index 0 = round 1
    # demand  : (8, 599) float64, index 0 = game week 1
    # on_hand0, pipe_w1, pipe_w2: (599,)

def simulate_many(demand, on_hand0, pipe_w1, pipe_w2, order_fn, ...) -> list[SimResult]: ...

# features.py
def make_features(history: np.ndarray, in_stock: np.ndarray, upto: int) -> np.ndarray:
    """Causal features using ONLY columns [0:upto]. Never reads [upto:]."""

# forecast.py
def fit_horizons(features_tr, targets_tr, horizons=(1, 2, 3)) -> Model
def predict(model, features) -> np.ndarray   # (n_series, 3) forecasts for h=1,2,3
def seasonal_reference(history, upto, lags=(13, 26)) -> np.ndarray  # (n_series, 3)

# policy.py
def decide(decision_idx, on_hand, arrivals_now, arrivals_next, fc_h1, fc_h2, fc_h3, ctx) -> np.ndarray
    # returns (599,) non-negative orders for round decision_idx (1-based)

# cv.py
def run_cv(config, ds) -> CVReport   # robust cost, per-fold, stress, WAPE, bias

# submit.py
def build_submission(orders: np.ndarray) -> pd.DataFrame  # (599, 3) Store, Product, order
def validate_submission(df, template) -> None

# verify.py
def independent_simulate(...) -> tuple[float, float, float]  # second implementation
```

## Anti-leakage rule (enforced, not aspirational)

Any function that computes features or orders at decision time `t` may read only
history columns `[0:t]`. The mandatory test: mutate `history` and `game_sales`
columns at and after `t`, recompute, and assert decision-time outputs are
byte-identical. `test_forecast_causality.py` must implement and pass this.

## Reproduction gates (must all pass before any model result counts)

1. `verify_data_hashes()` returns all True.
2. `simulate` with **any** policy reproduces `weekly_total[0] == 380.6` and
   `weekly_total[1] == 533.2` on the true initial state (±1e-6).
3. The organizer benchmark policy reproduces `scored_cost == 4334.00` (±1e-6).
4. `verify.independent_simulate` agrees with `simulate.simulate` to < 1e-9 on
   randomized orders and demand across all CV folds.

## Validity vocabulary (report exactly these terms)

- **CV**: chronological, non-overlapping, causal. Only basis for selection.
- **Retrospective competition replay**: the game weeks 1-8 labels were published
  before this project existed. **Never blind. Never official. Never a win.**
- Published anchors for context only: organizer baseline EUR 4,334.00,
  published winner EUR 3,763.20.

## Reporting rule

Always report forecast quality (WAPE, bias) separately from inventory cost
(holding vs shortage split). A better WAPE does not imply a better cost.
