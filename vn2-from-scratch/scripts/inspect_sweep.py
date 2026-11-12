"""Read the frozen sweep and report whether the LightGBM branch earns its place."""

import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
d = json.loads((ROOT / "artifacts" / "frozen_config.json").read_text())

print("n_candidates:", d["n_candidates"])
print()

by_sw = defaultdict(list)
for r in d["all_points"]:
    by_sw[r["seasonal_weight"]].append(r["robust"])

print("robust CV cost by seasonal weight (lower = better):")
for sw in sorted(by_sw, reverse=True):
    vals = by_sw[sw]
    best = min(vals)
    mean = sum(vals) / len(vals)
    print(f"  sw={sw:<4}  n={len(vals):>3}  best={best:>9.2f}  mean_of_n={mean:>9.2f}")

print()
print("best coverage+buffer AT EACH seasonal weight:")
for sw in sorted(by_sw, reverse=True):
    rows = [r for r in d["all_points"] if r["seasonal_weight"] == sw]
    b = min(rows, key=lambda r: r["robust"])
    print(
        f"  sw={sw:<4} cov={b['coverage']:<4} buf={b['buffer_coef']:<5} "
        f"robust={b['robust']:>9.2f} mean={b['mean']:>8.2f} worst={b['worst']:>9.2f}"
    )

print()
print("does LightGBM help at a FIXED policy? (best coverage/buffer per weight)")
print("gap of best-LGBM-config vs best-pure-seasonal-config:")
best_pure = min(r["robust"] for r in d["all_points"] if r["seasonal_weight"] == 1.0)
for sw in sorted(by_sw, reverse=True):
    if sw == 1.0:
        continue
    best = min(r["robust"] for r in d["all_points"] if r["seasonal_weight"] == sw)
    print(
        f"  sw={sw:<4}  best={best:>9.2f}  vs pure-seasonal {best_pure:>9.2f}  delta={best - best_pure:>+9.2f}"
    )

print()
print("worst-fold outlier driving the 8293 fold, at best config:")
b = d["best"]
print("  per_fold:", b["per_fold"])
print(
    "  fold 2 (origin 81) is an outlier; other folds:",
    [c for i, c in enumerate(b["per_fold"]) if i != 2],
)
