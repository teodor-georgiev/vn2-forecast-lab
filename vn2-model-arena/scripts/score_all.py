"""Score every arena run and write the leaderboard.

Usage:
    python score_all.py

Reads runs/<slug>/predictions.csv for each expected slug, scores against the private
answer panel, and writes results/arena_results.csv and results/leaderboard.md.
Runs with a missing or invalid submission are kept in the table, marked INVALID.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd

ARENA = Path(r"C:\Users\Toni\Documents\Default Project\vn2-model-arena")
RUNS = ARENA / "runs"
RESULTS = ARENA / "results"
SCORER = ARENA / "scripts" / "score_submission.py"
PRIVATE = Path(r"C:\Users\Toni\AppData\Local\Temp\opencode\vn2-arena-private")

MODELS = {
    "minimax-m3.1-flash": "MiniMax M3.1 Flash Free",
    "deepseek-v4.1-flash": "DeepSeek V4.1 Flash Free",
    "glm-5.3-flash": "GLM-5.3 Flash Free",
    "mimo-v2.6-flash": "MiMo-V2.6 Flash Free",
    "beam": "Beam Free",
    "longcat-2.5": "LongCat 2.5 Preview Free",
    "nemotron-3-ultra": "Nemotron 3 Ultra Free",
    "nemotron-3.5-super-vl": "Nemotron 3.5 Super VL Free",
}


def score_one(slug: str) -> dict:
    pred = RUNS / slug / "predictions.csv"
    if not pred.exists():
        return {
            "slug": slug,
            "valid": False,
            "problems": ["predictions.csv missing"],
            "wape": None,
        }
    proc = subprocess.run(
        [sys.executable, str(SCORER), str(pred), "--label", slug],
        capture_output=True,
        text=True,
        cwd=str(ARENA),
    )
    out = proc.stdout.strip()
    try:
        payload = json.loads(out[out.index("{") :])
    except Exception:  # noqa: BLE001
        return {
            "slug": slug,
            "valid": False,
            "problems": [f"scorer failed: {proc.stderr.strip()[:200] or out[:200]}"],
            "wape": None,
        }
    payload["slug"] = slug
    return payload


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    truth = pd.read_csv(PRIVATE / "answer_panel.csv")

    rows = [score_one(slug) for slug in MODELS]
    frame = pd.DataFrame(rows)
    frame["model"] = frame["slug"].map(MODELS)
    frame = frame[
        ["slug", "model", "valid", "wape", "pbias", "mae", "rmse", "rows", "problems"]
    ].sort_values("wape", na_position="last")
    frame.to_csv(RESULTS / "arena_results.csv", index=False)

    baselines = RESULTS / "baselines"
    baseline_rows = []
    for path in sorted(baselines.glob("*.csv")):
        proc = subprocess.run(
            [sys.executable, str(SCORER), str(path), "--label", path.stem],
            capture_output=True,
            text=True,
            cwd=str(ARENA),
        )
        out = proc.stdout.strip()
        payload = json.loads(out[out.index("{") :])
        baseline_rows.append(
            {
                "name": path.stem,
                "wape": payload.get("wape"),
                "pbias": payload.get("pbias"),
                "mae": payload.get("mae"),
            }
        )
    base = pd.DataFrame(baseline_rows).sort_values("wape")
    base.to_csv(RESULTS / "baseline_reference.csv", index=False)

    lines = [
        "# VN2 arena — model leaderboard",
        "",
        "Primary metric: holdout WAPE (lower is better). Truth: 599 series x 8 weeks, "
        "4792 cells, total demand %.1f." % float(truth["y_true"].sum()),
        "",
        "| rank | model | WAPE | PBIAS | MAE | valid | notes |",
        "|---:|---|---:|---:|---:|:--:|---|",
    ]
    for i, row in enumerate(frame.itertuples(index=False), start=1):
        wape = f"{row.wape:.4f}" if pd.notna(row.wape) else "—"
        pb = f"{row.pbias:+.3f}" if pd.notna(row.pbias) else "—"
        mae = f"{row.mae:.3f}" if pd.notna(row.mae) else "—"
        notes = "" if row.valid else "; ".join(row.problems or [])[:120]
        lines.append(
            f"| {i} | {row.model} | {wape} | {pb} | {mae} | "
            f"{'yes' if row.valid else 'NO'} | {notes} |"
        )
    lines += [
        "",
        "## Reference points",
        "",
        "| baseline | WAPE | PBIAS | MAE |",
        "|---|---:|---:|---:|",
    ]
    for row in base.itertuples(index=False):
        lines.append(
            f"| {row.name} | {row.wape:.4f} | {row.pbias:+.3f} | {row.mae:.3f} |"
        )
    (RESULTS / "leaderboard.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
