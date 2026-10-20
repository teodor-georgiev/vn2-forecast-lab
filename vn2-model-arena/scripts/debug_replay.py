"""Debug a worker policy in the real replay with full tracebacks (manager tool).

Usage: python scripts\\debug_replay.py runs3\\w05 runs3\\w13 ...
"""

from __future__ import annotations

import importlib.util
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

ARENA = Path(r"C:\Users\Toni\Documents\Default Project\vn2-model-arena")
PRIVATE = Path(r"C:\Users\Toni\AppData\Local\Temp\opencode\vn2-arena-private")
sys.path.insert(0, str(ARENA / "tools"))
import inventory_sim as sim  # noqa: E402


def load_truth() -> tuple[np.ndarray, np.ndarray]:
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
    return demand, state


def load_decide(path: Path):
    spec = importlib.util.spec_from_file_location(f"debug_{path.parent.name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.decide


def replay_trace(decide, demand, state) -> None:
    n = demand.shape[1]
    orders = np.zeros((sim.ORDER_WEEKS, n))
    for w in range(sim.ORDER_WEEKS):
        try:
            decision = decide(w, state.copy(), demand[:w].T.copy())
            arr = np.asarray(decision, dtype=float).ravel()
            print(f"  week {w}: ok, first3={arr[:3]}")
            orders[w] = np.rint(np.maximum(0.0, arr))
        except Exception:
            print(f"  week {w}: FAILED")
            traceback.print_exc()
            return
        state = np.column_stack(
            (
                np.maximum(0.0, state[:, 0] + state[:, 1] - demand[w]),
                state[:, 2],
                orders[w],
            )
        )


def main() -> None:
    demand, state = load_truth()
    for target in sys.argv[1:]:
        path = Path(target)
        policy_path = path / "policy.py"
        print(f"=== {path.parent.name} ===")
        try:
            decide = load_decide(policy_path)
            replay_trace(decide, demand, state)
        except Exception:
            print("load failed:")
            traceback.print_exc()


if __name__ == "__main__":
    main()
