"""Ensemble the contestants' round-1 forecasts and score the combinations.

Writes each ensemble to results/ensembles/ and scores it with the frozen scorer, so
the combination is judged exactly like an individual entry.
"""

from __future__ import annotations

import json
import subprocess
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

ARENA = Path(r"C:\Users\Toni\Documents\Default Project\vn2-model-arena")
RUNS = ARENA / "runs"
OUT = ARENA / "results" / "ensembles"
SCORER = ARENA / "scripts" / "score_submission.py"
PRIVATE = Path(r"C:\Users\Toni\AppData\Local\Temp\opencode\vn2-arena-private")

SLUGS = [
    "minimax-m3.1-flash",
    "deepseek-v4.1-flash",
    "glm-5.3-flash",
    "mimo-v2.6-flash",
    "beam",
    "longcat-2.5",
    "nemotron-3-ultra",
    "nemotron-3.5-super-vl",
]


def load_matrix(slug: str) -> pd.DataFrame:
    frame = pd.read_csv(RUNS / slug / "predictions.csv")
    frame["series_id"] = frame["series_id"].astype(str)
    return frame.pivot(
        index="series_id", columns="horizon", values="y_hat"
    ).sort_index()


def write_submission(name: str, matrix: pd.DataFrame) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [
        (sid, int(h), float(value))
        for sid, series in matrix.iterrows()
        for h, value in series.items()
    ]
    path = OUT / f"{name}.csv"
    pd.DataFrame(rows, columns=["series_id", "horizon", "y_hat"]).to_csv(
        path, index=False
    )
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
    matrices = {slug: load_matrix(slug) for slug in SLUGS}
    common_index = matrices[SLUGS[0]].index
    for slug, matrix in matrices.items():
        assert matrix.index.equals(common_index), slug

    stack = np.stack(
        [m.to_numpy(float) for m in matrices.values()]
    )  # (8, series, horizon)
    results = []

    def record(name: str, array: np.ndarray) -> None:
        matrix = pd.DataFrame(
            array, index=common_index, columns=matrices[SLUGS[0]].columns
        )
        payload = score(write_submission(name, matrix), name)
        results.append(
            {
                "ensemble": name,
                "wape": payload["wape"],
                "pbias": payload["pbias"],
                "mae": payload["mae"],
                "valid": payload["valid"],
            }
        )

    record("mean_all8", stack.mean(axis=0))
    record("median_all8", np.median(stack, axis=0))
    record("trimmed_all8", np.sort(stack, axis=0)[1:-1].mean(axis=0))

    # best-subset search: median of every 3-, 4- and 5-model combination
    for size in (3, 4):
        for combo in combinations(range(len(SLUGS)), size):
            subset = stack[list(combo)]
            matrix = np.median(subset, axis=0)
            name = "median_" + "_".join(SLUGS[i].split("-")[0] for i in combo)
            record(name, matrix)

    frame = pd.DataFrame(results).sort_values("wape")
    frame.to_csv(ARENA / "results" / "ensemble_results.csv", index=False)
    print(frame.head(15).to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\n{len(frame)} combinations scored")


if __name__ == "__main__":
    main()
