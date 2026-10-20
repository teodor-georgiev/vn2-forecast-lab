"""Fresh-process replay timing + sanity check of the deliverable (scratch)."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

t0 = time.perf_counter()
import policy  # noqa: E402

import importlib.util  # noqa: E402

spec = importlib.util.spec_from_file_location("agent_policy", HERE / "policy.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
import_time = time.perf_counter() - t0

n = 599
state = np.zeros((n, 3))
state[:, 0] = 1.5
state[:, 1] = 0.5
state[:, 2] = 0.25
revealed = np.zeros((0, n))
orders = []
t1 = time.perf_counter()
for w in range(6):
    o = mod.decide(w, state.copy(), revealed)
    assert o.shape == (n,) and np.isfinite(o).all() and (o >= 0).all()
    orders.append(o)
    # fake reveal so the next round has data
    revealed = np.vstack([revealed, np.full((1, n), 2.0)])
replay_time = time.perf_counter() - t1
print(f"import policy.py: {import_time:.2f}s")
print(f"6 decide calls (n=599, full history): {replay_time:.2f}s")
print(f"TOTAL game compute: {import_time + replay_time:.2f}s  (limit 30s)")
print(f"total units ordered: {sum(o.sum() for o in orders):.0f}")
print(f"per-round units: {[round(float(o.sum())) for o in orders]}")

# sanity: policy must not write files
import ast  # noqa: E402

src = (HERE / "policy.py").read_text(encoding="utf-8")
tree = ast.parse(src)
bad = [
    n
    for n in ast.walk(tree)
    if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "open"
]
print("open() calls in policy.py:", len(bad))
print(
    "writes data files:",
    any(
        isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr in ("to_csv", "write", "save", "dump", "writelines")
        for n in ast.walk(tree)
    ),
)
