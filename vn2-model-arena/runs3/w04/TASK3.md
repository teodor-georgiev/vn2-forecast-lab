# Round 3 — VN2 played for real: the adaptive ordering game

Same 599 anonymised series. This round replicates the actual VN2 competition format
(https://www.datasource.ai/en/home/data-science-competitions-for-startups/vn2-inventory-planning-challenge/description):

- You place **one order per week for 6 weeks** (the game runs 8 weeks; the last order
  arrives in week 8). Lead time **2 weeks**: an order placed for week X is available
  for sales in week X+2.
- **You get feedback**: at each decision you are told the actual demand observed so
  far in THIS game (the revealed sales of the weeks that already happened) and your
  current inventory state. In the real competition participants received exactly this.
- Costs: **1.00 EUR per missed unit, 0.20 EUR per held unit-week**, scored over
  **weeks 3-8** (weeks 1-2 are a settling period and are excluded; they are identical
  for every policy by construction).

## Your job

Write the decision policy that plays this game. You write code — the manager replays
your code against the hidden demand sequence, calling it once per round with the live
state and the revealed history, exactly like the competition server did.

Deliverable, in your assigned work directory:

```
policy.py
```

It must define exactly this function:

```python
def decide(week_index, state, revealed):
    """week_index: int 0..7 — index of the week about to start (0 = before week 1).
    state:        np.ndarray (n_series, 3): on_hand, delivery_during_this_week,
                  delivery_during_next_week.
    revealed:     np.ndarray (week_index, n_series): actual demand observed so far
                  within this game; empty (0, n) array at week_index 0.
    returns:      np.ndarray (n_series,) of non-negative order quantities (float,
                  will be rounded to whole units).
    Timing: an order you return at week_index w is delivered during week w+2.
    """
```

The manager runs an 8-week replay in which `decision` is called with `week_index`
0..7, but only the decisions at 0..5 become orders (7 and 8 have no order slot).
Your `policy.py` must not write files, must not read anything outside your own work
directory plus the arena `data\` and `tools\` folders (read-only), must not access the
internet, and must finish the whole game in under 30 seconds of compute. numpy,
pandas, scikit-learn, lightgbm, xgboost, catboost and statsmodels are installed.

## What you know going in

- History: `..\..\data\sales.csv` (157 weeks), `..\..\data\availability.csv`
  (1 = this product had stock, 0 = it was sold out — sales during week 0 are
  censored, the unobserved demand was higher), `..\..\data\product_attributes.csv`.
- Start: `..\..\data\initial_state.csv` (per series: on_hand, arrival_w1, arrival_w2).
- The exact scoring engine `..\..\tools\inventory_sim.py` (read it; its
  `order_plan_from_policy` documents the flow).
- A ready backtester `.\helper_backtest.py` and an interface check `.\selfcheck.py`
  (both are the manager's, read-only).
- Reference bars, measured by the manager on this exact arena: a decent adaptive
  rule with 13-week-mean forecasts scores about **3,983**; the winning historical
  competition replay is equivalent to about **3,481** in arena units. A blind static
  plan scored 4,211. You may cite these as targets. You do NOT know the hidden
  demand and must not try to guess it — you will never see it. Defeat the adaptive
  reference bar of 3,983 honestly.

## Method hints (from the research profile)

- Sales are censored by availability: weeks with availability 0 under-report demand.
  Correcting for censoring is one of the biggest levers.
- Demand is intermittent and weekly seasonal (yearly pattern matters). ABC/XYZ
  segmentation (high-volume vs intermittent, stable vs seasonal) lets different
  series use different methods.
- A per-series newsvendor quantile chosen on simulated cost, not a rule of thumb,
  usually beats fixed safety stock.
- Budget **25 minutes**. Run `python .\selfcheck.py` first, backtest on at least two
  historical weekly cutoffs with `.\helper_backtest.py`, then finalise. Correctness
  and validity beat cleverness: a broken policy scores like zero orders (~9,890).
