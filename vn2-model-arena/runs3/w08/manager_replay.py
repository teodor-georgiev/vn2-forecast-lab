"""Manager-style replay verification: import policy.py by path, call decide 8x."""

from __future__ import annotations

import importlib.util
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from final_check import load_initial

spec = importlib.util.spec_from_file_location("agent_policy", HERE / "policy.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
decide = mod.decide

initial = load_initial()
n = initial.shape[0]
state = initial.copy()
revealed = []

t0 = time.perf_counter()
for w in range(8):
    rev = np.array(revealed).reshape(w, n) if revealed else np.empty((0, n))
    q = decide(w, state.copy(), rev)
    q = np.asarray(q, dtype=float).ravel()
    assert q.shape == (n,), q.shape
    assert np.isfinite(q).all()
    assert (q >= 0).all()
    on_hand, arr1, arr2 = state.T.copy()
    # pretend the next week's demand is the series' recent average
    d = np.zeros(n)
    end = np.maximum(0.0, on_hand + arr1 - d)
    state = np.column_stack((end, arr2, q if w < 6 else np.zeros(n)))
    revealed.append(d)
    print(f"  week {w}: order total={q.sum():9.0f}")
el = time.perf_counter() - t0
print(f"REPLAY OK — 8 decide() calls, total wall time {el:.2f}s")
