"""Ensemble the contestants' round-2 order plans and score the combinations.

Each contestant decided all six order weeks blind. This script combines their plans
(per series, per week) with plain mean / median / trimmed mean / 25th percentile,
scores every combination with the frozen round-2 scorer, and writes the plans to
results/ensembles2/.

The primary table is decision-free (no tuning against the answer panel). A clearly
labelled scaling diagnostic follows it: the ensemble median multiplied by a constant
chosen with hindsight, reported as headroom, not as a blind rule.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

ARENA = Path(r"C:\Users\Toni\Documents\Default Project\vn2-model-arena")
RUNS = ARENA / "runs"
OUT = ARENA / "results" / "ensembles2"
SCORER = ARENA / "scripts" / "score_orders.py"
ORDER_WEEKS = 6

# distinct, stable short names for combination labels (two slugs start with "nemotron")
TAGS = {
    "beam": "beam",
    "deepseek-v4.1-flash": "deepseek",
    "glm-5.3-flash": "glm",
    "minimax-m3.1-flash": "minimax",
    "nemotron-3-ultra": "n3u",
    "nemotron-3.5-super-vl": "n35vl",
    "longcat-2.5": "longcat",
    "mimo-v2.6-flash": "mimo",
}


def tag(slug: str) -> str:
    return TAGS.get(slug, slug.split("-")[0])


def discover_slugs() -> list[str]:
    slugs = sorted(
        d.name for d in RUNS.iterdir() if d.is_dir() and (d / "orders.csv").exists()
    )
    return slugs


def load_matrix(slug: str) -> pd.DataFrame:
    frame = pd.read_csv(RUNS / slug / "orders.csv")
    frame.columns = [str(c).strip() for c in frame.columns]
    frame["series_id"] = frame["series_id"].astype(str)
    frame["week"] = pd.to_numeric(frame["week"], errors="coerce").astype(int)
    frame["quantity"] = pd.to_numeric(frame["quantity"], errors="coerce").fillna(0.0)
    return (
        frame.drop_duplicates(["series_id", "week"], keep="last")
        .pivot(index="series_id", columns="week", values="quantity")
        .sort_index()
    )


def write_plan(name: str, matrix: pd.DataFrame | np.ndarray) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    if not isinstance(matrix, pd.DataFrame):
        raise TypeError("write_plan expects a DataFrame")
    frame = (
        matrix.stack()
        .rename("quantity")
        .reset_index()
        .rename(columns={"level_1": "week"})
        .sort_values(["series_id", "week"])
    )
    path = OUT / f"{name}.csv"
    frame[["series_id", "week", "quantity"]].to_csv(path, index=False)
    return path


def score(path: Path, label: str) -> dict:
    proc = subprocess.run(
        [sys.executable, str(SCORER), str(path), "--label", label],
        capture_output=True,
        text=True,
        cwd=str(ARENA),
    )
    out = proc.stdout.strip()
    return json.loads(out[out.index("{") :])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--diag-only",
        action="store_true",
        help="skip the combination table and only run the scaling diagnostic",
    )
    args = parser.parse_args()

    slugs = discover_slugs()
    matrices = {slug: load_matrix(slug) for slug in slugs}
    index = next(iter(matrices.values())).index
    for slug, matrix in matrices.items():
        assert matrix.index.equals(index), f"series mismatch in {slug}"
        assert list(matrix.columns) == list(range(1, ORDER_WEEKS + 1)), slug

    stack = np.stack([m.to_numpy(float) for m in matrices.values()])
    results: list[dict] = []

    def record(name: str, array: np.ndarray) -> None:
        matrix = pd.DataFrame(array, index=index, columns=range(1, ORDER_WEEKS + 1))
        payload = score(write_plan(name, matrix), name)
        results.append(
            {
                "ensemble": name,
                "score": payload["score"],
                "holding": payload["holding_weeks3_8"],
                "shortage": payload["shortage_weeks3_8"],
                "ordered": payload["ordered_total"],
                "valid": payload["valid"],
            }
        )

    record("mean_all", stack.mean(axis=0))
    record("median_all", np.median(stack, axis=0))
    record("trimmed_all", np.sort(stack, axis=0)[1:-1].mean(axis=0))
    record("q25_all", np.quantile(stack, 0.25, axis=0))

    for size in (3, 4):
        for combo in combinations(range(len(slugs)), size):
            subset = stack[list(combo)]
            name = "median_" + "_".join(tag(slugs[i]) for i in combo)
            record(name, np.median(subset, axis=0))

    names = [r["ensemble"] for r in results]
    assert len(names) == len(set(names)), "combination labels collided"

    if args.diag_only:
        results.clear()

    if results:
        frame = pd.DataFrame(results).sort_values("score")
        frame.to_csv(ARENA / "results" / "ensemble2_results.csv", index=False)
        print(f"members: {slugs}")
        print(frame.head(15).to_string(index=False, float_format=lambda v: f"{v:,.2f}"))
        print(f"\n{len(frame)} combinations scored")

    print("\n--- scaling diagnostic on median_all (hindsight, not a blind rule) ---")
    median_plan = np.median(stack, axis=0)
    diag = []
    for scale in (0.4, 0.6, 0.8, 0.9, 1.0, 1.1, 1.2, 1.5, 2.0):
        plan = pd.DataFrame(
            median_plan * scale, index=index, columns=range(1, ORDER_WEEKS + 1)
        )
        payload = score(write_plan("diag", plan), f"median_all x{scale}")
        diag.append(
            {
                "scale": scale,
                "score": payload["score"],
                "holding": payload["holding_weeks3_8"],
                "shortage": payload["shortage_weeks3_8"],
            }
        )
    diag_frame = pd.DataFrame(diag)
    diag_frame.to_csv(ARENA / "results" / "ensemble2_scaling.csv", index=False)
    print(diag_frame.to_string(index=False, float_format=lambda v: f"{v:,.2f}"))


if __name__ == "__main__":
    main()
