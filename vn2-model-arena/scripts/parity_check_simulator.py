"""Parity check: the arena's extracted inventory simulator vs the proven research one.

Runs the same forecasts and policy through
  A) `simulate()` in C:\\Users\\Toni\\vn2-direct-lgbm-20261008\\run_vn2_lgbm.py  (proven)
  B) `inventory_sim.py` in this arena's tools directory                     (extracted)
and requires identical weekly costs and scores. Uses the real 599-series VN2 panel.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

RESEARCH = Path(r"C:\Users\Toni\vn2-direct-lgbm-20261008")
TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(RESEARCH))
sys.path.insert(0, str(TOOLS))

import inventory_sim as arena  # noqa: E402
import run_vn2_lgbm as proven  # noqa: E402


def causal_forecasts(panel: dict, cutoff: int) -> np.ndarray:
    """13-week causal mean per decision week and horizon (same input for both simulators)."""
    forecast = np.zeros((6, 3, panel["y"].shape[1]))
    for decision in range(6):
        hist = proven.imputed(panel["y"], panel["instock"], cutoff + decision)
        window = hist[cutoff + decision - 13 : cutoff + decision]
        for i in range(3):
            forecast[decision, i] = window.mean(axis=0)
    return forecast


def compare(panel: dict, forecast: np.ndarray, params: dict) -> None:
    cutoff = 157
    truth = panel["y"][cutoff : cutoff + 8]
    initial = panel["initial"]

    a = proven.simulate(
        panel, cutoff, forecast, params, competition=True, detailed=True
    )

    b = arena.order_plan_from_policy(
        truth,
        initial,
        forecast_fn=lambda w, state, hist: forecast[w],
        policy=lambda f, s: arena.reference_orders(
            f, s, params["scale"], params["projection"], params["buffer"]
        ),
    )
    c = arena.simulate(truth, initial, b["orders"])

    print(f"policy={params}")
    print(f"  proven   score={a['score']:.6f} setup={a['setup']:.6f}")
    print(f"  arena    score={b['score']:.6f} setup={b['setup']:.6f}")
    print(f"  arena(simulate) score={c['score']:.6f} setup={c['setup']:.6f}")

    assert abs(a["score"] - b["score"]) < 1e-9, "score mismatch"
    assert abs(a["setup"] - b["setup"]) < 1e-9, "setup mismatch"
    assert abs(a["score"] - c["score"]) < 1e-9, "fixed-plan mismatch"
    assert abs(b["score"] - c["score"]) < 1e-9, "policy vs fixed-plan mismatch"
    for wa, wb, wc in zip(a["weekly"], b["weekly"], c["weekly"]):
        for key in ("holding", "shortage", "total", "ordered"):
            assert abs(wa[key] - wb[key]) < 1e-9, f"week {wa['week']} {key} mismatch"
            assert abs(wa[key] - wc[key]) < 1e-9, (
                f"week {wa['week']} {key} mismatch (fixed)"
            )
    print("  weekly rows identical (holding, shortage, total, ordered): PASS")


def main() -> None:
    panel = proven.load()
    print(f"panel: {panel['y'].shape[1]} series, {panel['y'].shape[0]} weeks")
    forecast = causal_forecasts(panel, 157)
    for params in (
        {"policy": "deficit", "scale": 1.0, "projection": 1.0, "buffer": 0.75},
        {"policy": "deficit", "scale": 0.93, "projection": 1.03, "buffer": 0.20},
        {"policy": "deficit", "scale": 1.45, "projection": 0.75, "buffer": 2.80},
    ):
        compare(panel, forecast, params)
    print("PARITY: extracted simulator matches the proven research simulator")


if __name__ == "__main__":
    main()
