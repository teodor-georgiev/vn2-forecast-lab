"""Replay round-3 adaptive policies against the hidden demand (manager tool).

For each agent directory under runs3\\: load policy.py, call decide(w, state,
revealed) for w = 0..5 with truthful feedback (the revealed weeks' actual demand —
exactly what VN2 competition participants received), then score the resulting order
plan with the frozen simulator. Also builds consensus policies (median/mean across
agents) and scores them the same way, as the pre-registered combination rule.

Usage:
    python scripts\\score_adaptive.py            (score all + combine)
    python scripts\\score_adaptive.py runs3\\w01  (score one agent)
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ARENA = Path(r"C:\Users\Toni\Documents\Default Project\vn2-model-arena")
PRIVATE = Path(r"C:\Users\Toni\AppData\Local\Temp\opencode\vn2-arena-private")
sys.path.insert(0, str(ARENA / "tools"))
import inventory_sim as sim  # noqa: E402


def load_truth() -> tuple[np.ndarray, np.ndarray, list[str]]:
    truth = pd.read_csv(PRIVATE / "answer_panel.csv")
    initial = pd.read_csv(ARENA / "data" / "initial_state.csv")
    series = sorted(truth["series_id"].unique().tolist())
    demand = (
        truth.pivot(index="horizon", columns="series_id", values="y_true")
        .loc[range(1, sim.WEEKS + 1), series]
        .to_numpy(float)
    )
    state = (
        initial.set_index("series_id")
        .loc[series][["on_hand", "arrival_w1", "arrival_w2"]]
        .to_numpy(float)
    )
    return demand, state, series


def load_decide(path: Path):
    spec = importlib.util.spec_from_file_location(f"policy_{path.parent.name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.decide


def replay(decide, demand: np.ndarray, state: np.ndarray) -> dict:
    n = demand.shape[1]
    orders = np.zeros((sim.ORDER_WEEKS, n))
    t0 = time.perf_counter()
    failures = []
    for w in range(sim.ORDER_WEEKS):
        try:
            decision = decide(w, state.copy(), demand[:w].T.copy())
            arr = np.asarray(decision, dtype=float).ravel()
            if arr.shape != (n,):
                raise ValueError(f"shape {arr.shape} != ({n},)")
            if not np.isfinite(arr).all():
                raise ValueError("non-finite values")
            orders[w] = np.rint(np.maximum(0.0, arr))
        except Exception as exc:  # noqa: BLE001
            failures.append(f"week {w}: {exc.__class__.__name__}: {exc}")
            orders[w] = np.zeros(n)
        state = np.column_stack(
            (
                np.maximum(0.0, state[:, 0] + state[:, 1] - demand[w]),
                state[:, 2],
                orders[w],
            )
        )
    elapsed = time.perf_counter() - t0
    return {"orders": orders, "elapsed": elapsed, "failures": failures}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("target", nargs="?", default=None)
    args = parser.parse_args()
    demand, state, series = load_truth()

    if args.target:
        target_path = ARENA / args.target
        targets = (
            sorted(target_path.glob("policy.py"))
            if target_path.is_dir()
            else [target_path]
        )
    else:
        targets = sorted((ARENA / "runs3").glob("w*/policy.py"))

    plans: dict[str, np.ndarray] = {}
    rows: list[dict] = []
    for path in targets:
        name = path.parent.name
        failures = []
        try:
            decide = load_decide(path)
            replayed = replay(decide, demand, state)
            orders, elapsed, failures = (
                replayed["orders"],
                replayed["elapsed"],
                replayed["failures"],
            )
        except Exception as exc:  # noqa: BLE001
            orders = np.zeros((sim.ORDER_WEEKS, len(series)))
            failures = [f"load: {exc.__class__.__name__}: {exc}"]
            elapsed = 0.0
        plans[name] = orders
        result = sim.simulate(demand, state, orders)
        holding = sum(r["holding"] for r in result["weekly"][2:])
        shortage = sum(r["shortage"] for r in result["weekly"][2:])
        rows.append(
            {
                "agent": name,
                "score": result["score"],
                "holding": holding,
                "shortage": shortage,
                "ordered": int(orders.sum()),
                "decide_seconds": round(elapsed, 2),
                "failures": "; ".join(failures) if failures else "",
            }
        )

    stacked = np.stack(list(plans.values()))
    consensus = {"median_all": np.median(stacked, axis=0)}
    if len(stacked) >= 2:
        consensus["mean_all"] = stacked.mean(axis=0)
    if len(stacked) >= 3:
        consensus["trimmed_all"] = np.sort(stacked, axis=0)[1:-1].mean(axis=0)
    for name, orders in consensus.items():
        result = sim.simulate(demand, state, orders)
        rows.append(
            {
                "agent": f"CONSENSUS_{name}",
                "score": result["score"],
                "holding": sum(r["holding"] for r in result["weekly"][2:]),
                "shortage": sum(r["shortage"] for r in result["weekly"][2:]),
                "ordered": int(orders.sum()),
                "decide_seconds": 0.0,
                "failures": "",
            }
        )

    frame = pd.DataFrame(rows).sort_values("score")
    (ARENA / "results" / "round3_results.csv").write_text(frame.to_csv(index=False))
    lines = ["# VN2 arena — round 3 (adaptive game)", ""]
    lines.append(
        "| rank | agent | score | holding | shortage | ordered | decide s | failures |"
    )
    lines.append("|---:|---|---:|---:|---:|---:|---:|---|")
    for rank, row in enumerate(frame.itertuples(), start=1):
        lines.append(
            f"| {rank} | {row.agent} | {row.score:,.2f} | {row.holding:,.2f} | "
            f"{row.shortage:,.2f} | {row.ordered:,} | {row.decide_seconds} | "
            f"{row.failures or '—'} |"
        )
    (ARENA / "results" / "round3_leaderboard.md").write_text("\n".join(lines) + "\n")
    print(frame.to_string(index=False, max_colwidth=60))


if __name__ == "__main__":
    main()
