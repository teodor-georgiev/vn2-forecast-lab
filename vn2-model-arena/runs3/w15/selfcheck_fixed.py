"""Manager's selfcheck.py with its one broadcasting bug fixed (line 42:
(12,20) * (12,) does not broadcast). Same checks, same seeds, same contract."""

from __future__ import annotations

import importlib.util
import time
from pathlib import Path

import numpy as np

rng = np.random.default_rng(20261019)
N = 12
HIST = 20


def build_policy():
    here = Path(__file__).parent
    spec = importlib.util.spec_from_file_location("agent_policy", here / "policy.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "decide"):
        raise SystemExit(
            "FAIL: policy.py must define decide(week_index, state, revealed)"
        )
    return mod.decide


def main() -> None:
    decide = build_policy()
    started = time.perf_counter()
    scale = rng.uniform(0.05, 3.0, size=N)
    history = (
        np.maximum(0, rng.negative_binomial(1, 0.55, size=(N, HIST))) * scale[:, None]
    )
    state = np.zeros((N, 3))
    state[:, 0] = scale * rng.uniform(0, 8)
    state[:, 1] = scale * rng.uniform(0, 6)
    state[:, 2] = scale * rng.uniform(0, 6)
    orders = {}
    for w in range(6):
        revealed = history[:, :w].T.copy() if w else np.empty((0, N))
        decision = decide(w, state.copy(), revealed)
        arr = np.asarray(decision, dtype=float).ravel()
        if arr.shape != (N,):
            raise SystemExit(f"FAIL: week {w}: expected shape ({N},), got {arr.shape}")
        if not np.isfinite(arr).all():
            raise SystemExit(f"FAIL: week {w}: non-finite or NaN values")
        if (arr < 0).any():
            raise SystemExit(f"FAIL: week {w}: negative order quantities")
        orders[w] = np.rint(np.maximum(0.0, arr))
        on_hand, arr1, arr2 = state.T.copy()
        supply = on_hand + arr1
        demand_w = history[:, w].astype(float) if w < HIST else np.zeros(N)
        end = np.maximum(0.0, supply - demand_w)
        state = np.column_stack((end, arr2, orders[w]))
    elapsed = time.perf_counter() - started
    if elapsed > 20:
        print(f"WARN: decide loop took {elapsed:.1f}s — keep the replay < 30s")
    total = int(sum(o.sum() for o in orders.values()))
    print(
        f"PASS in {elapsed:.2f}s — decisions OK, total ordered units (synthetic): {total}"
    )


if __name__ == "__main__":
    main()
