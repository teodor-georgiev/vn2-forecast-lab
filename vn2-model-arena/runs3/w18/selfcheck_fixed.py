"""LOCAL COPY of the manager's selfcheck.py with the one broadcasting bug fixed.

The manager's copy has `rng.negative_binomial(1, 0.55, size=(N, HIST)) * scale`
where `scale` has shape (N,) -> ValueError (cannot broadcast (N,HIST) with (N,)).
The intent is clearly `* scale[:, None]`.  Nothing else is changed; the interface
contract checked here is identical to the manager's.
"""

from __future__ import annotations

import importlib.util
import time
from pathlib import Path

import numpy as np

rng = np.random.default_rng(20261019)
N = 12  # synthetic series
HIST = 20  # weeks of synthetic history the policy may not even need


def build_policy():
    here = Path(__file__).parent
    spec = importlib.util.spec_from_file_location("agent_policy", here / "policy.py")
    if spec is None or spec.loader is None:
        raise SystemExit("FAIL: policy.py not found next to selfcheck.py")
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
    state[:, 0] = scale * rng.uniform(0, 8)  # on hand
    state[:, 1] = scale * rng.uniform(0, 6)  # delivery during week 1
    state[:, 2] = scale * rng.uniform(0, 6)  # delivery during week 2

    orders = {}
    for w in range(6):
        revealed = history[:, :w].T.copy() if w else np.empty((0, N))
        decision = decide(w, state.copy(), revealed)
        arr = np.asarray(decision, dtype=float).ravel()
        if arr.shape != (N,):
            raise SystemExit(
                "FAIL: week %d: expected shape (%d,), got %s" % (w, N, arr.shape)
            )
        if not np.isfinite(arr).all():
            raise SystemExit(
                "FAIL: week %d: non-finite or NaN values in the order vector" % w
            )
        if (arr < 0).any():
            raise SystemExit("FAIL: week %d: negative order quantities" % w)
        orders[w] = np.rint(np.maximum(0.0, arr))
        on_hand, arr1, arr2 = state.T.copy()
        supply = on_hand + arr1
        demand_w = history[:, w].astype(float) if w < HIST else np.zeros(N)
        end = np.maximum(0.0, supply - demand_w)
        state = np.column_stack((end, arr2, orders[w]))

    elapsed = time.perf_counter() - started
    if elapsed > 20:
        print(
            "WARN: decide loop took %.1fs on a tiny game — keep the replay < 30s"
            % elapsed
        )
    total = int(sum(o.sum() for o in orders.values()))
    print(
        "PASS in %.2fs — decisions OK, total ordered units (synthetic): %d"
        % (elapsed, total)
    )


if __name__ == "__main__":
    main()
