"""Score every round-2 order plan and write the game leaderboard."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

ARENA = Path(r"C:\Users\Toni\Documents\Default Project\vn2-model-arena")
RUNS = ARENA / "runs"
RESULTS = ARENA / "results"
SCORER = ARENA / "scripts" / "score_orders.py"

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


def score_path(path: Path, label: str) -> dict:
    if not path.exists():
        return {
            "label": label,
            "valid": False,
            "problems": [f"{path.name} missing"],
            "score": None,
        }
    proc = subprocess.run(
        [sys.executable, str(SCORER), str(path), "--label", label],
        capture_output=True,
        text=True,
        cwd=str(ARENA),
    )
    out = proc.stdout.strip()
    try:
        return json.loads(out[out.index("{") :])
    except Exception:  # noqa: BLE001
        return {
            "label": label,
            "valid": False,
            "problems": [f"scorer failed: {proc.stderr.strip()[:200] or out[:200]}"],
            "score": None,
        }


def main() -> None:
    rows = [score_path(RUNS / slug / "orders.csv", slug) for slug in MODELS]
    frame = pd.DataFrame(rows)
    frame["model"] = frame["label"].map(MODELS)
    cols = [
        "label",
        "model",
        "valid",
        "score",
        "holding_weeks3_8",
        "shortage_weeks3_8",
        "ordered_total",
        "problems",
    ]
    for col in cols:
        if col not in frame:
            frame[col] = None
    frame = frame[cols].sort_values("score", na_position="last")
    RESULTS.mkdir(parents=True, exist_ok=True)
    frame.to_csv(RESULTS / "round2_results.csv", index=False)

    base_rows = []
    for path in sorted((RESULTS / "round2_baselines").glob("*.csv")):
        payload = score_path(path, path.stem)
        payload["name"] = path.stem
        base_rows.append(payload)
    base = pd.DataFrame(base_rows).sort_values("score")

    lines = [
        "# VN2 arena — round 2 (ordering game)",
        "",
        "Scored with the provided simulator: EUR 1.00 per missed unit, "
        "EUR 0.20 per held unit-week, weeks 3-8, orders weeks 1-6. Lower is better.",
        "",
        "| rank | model | score | holding | shortage | units ordered | valid | notes |",
        "|---:|---|---:|---:|---:|---:|:--:|---|",
    ]
    for i, row in enumerate(frame.itertuples(index=False), start=1):
        score = f"{row.score:,.2f}" if pd.notna(row.score) else "—"
        hold = f"{row.holding_weeks3_8:,.2f}" if pd.notna(row.holding_weeks3_8) else "—"
        short = (
            f"{row.shortage_weeks3_8:,.2f}" if pd.notna(row.shortage_weeks3_8) else "—"
        )
        ordered = f"{row.ordered_total:,.0f}" if pd.notna(row.ordered_total) else "—"
        notes = "" if row.valid else "; ".join(row.problems or [])[:120]
        lines.append(
            f"| {i} | {row.model} | {score} | {hold} | {short} | {ordered} | "
            f"{'yes' if row.valid else 'NO'} | {notes} |"
        )
    lines += [
        "",
        "## Reference plans",
        "",
        "| plan | score | holding | shortage | ordered | note |",
        "|---|---:|---:|---:|---:|---|",
    ]
    notes = {
        "zero_orders": "order nothing at all",
        "deficit_static_13wmean": "reference rule, static plan, fair bar",
        "deficit_adaptive_feedback": "reference rule with true weekly feedback (headroom, unreachable by a static plan)",
    }
    for row in base.itertuples(index=False):
        score = f"{row.score:,.2f}" if pd.notna(row.score) else "—"
        lines.append(
            f"| {row.name} | {score} | {row.holding_weeks3_8:,.2f} | "
            f"{row.shortage_weeks3_8:,.2f} | {row.ordered_total:,.0f} | "
            f"{notes.get(row.name, '')} |"
        )
    (RESULTS / "round2_leaderboard.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print("\n".join(lines))


if __name__ == "__main__":
    main()
