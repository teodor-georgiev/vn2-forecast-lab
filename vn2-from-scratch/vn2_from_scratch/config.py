"""Frozen experiment constants for the from-scratch VN2 solution.

THIS FILE IS THE SINGLE SOURCE OF TRUTH shared by every module and every
worker. Do not change values here without re-running the verification gates.

Competition: VN2 Inventory Planning Challenge (DataSource.ai).
Reference:   https://www.datasource.ai/en/home/data-science-competitions-for-startups/vn2-inventory-planning-challenge/description
"""

from __future__ import annotations

import hashlib
from pathlib import Path

# --------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------
N_SERIES = 599
HISTORY_WEEKS = 157  # Week0 sales, 2021-04-12 .. 2024-04-08 (inclusive)
GAME_WEEKS = 8  # revealed game weeks 2024-04-15 .. 2024-06-03 (inclusive)
N_DECISIONS = 6  # rounds 1..6; one order per round for all 599 series

# --------------------------------------------------------------------------
# Economics (EUR)
# --------------------------------------------------------------------------
HOLDING_COST = 0.20  # per unit of ENDING inventory, per week; no cost in transit
SHORTAGE_COST = 1.00  # per unit of LOST sale; no backorders
SCORE_WEEKS_START = 3  # 1-based inclusive game week
SCORE_WEEKS_END = 8  # 1-based inclusive game week
# -> 0-indexed scoring slice over an 8-week weekly array is [2:8].

# --------------------------------------------------------------------------
# Lead time
# --------------------------------------------------------------------------
# Order placed at the end of game week X is received at the start of game
# week X+3, so it is sellable in game week X+3. Round d (1-based) therefore
# arrives in game week d+2 (1-based).
#
# 0-indexed arrival schedule over game weeks t=0..7:
#   arr[0] = In Transit W+1   (pipeline already in flight at game start)
#   arr[1] = In Transit W+2
#   arr[t] = orders[t-2]      for t = 2..7   (orders shape is (6, N_SERIES))
#
# Consequence: game weeks 1 and 2 (t=0,1) costs are INDEPENDENT of every
# order decision. That is a hard invariant used as a verification gate.

# --------------------------------------------------------------------------
# Cross-validation (chronological, non-overlapping)
# --------------------------------------------------------------------------
CV_WINDOW = 8  # scored evaluation window length in weeks
# 0-indexed starts into the 157-week history; 8 disjoint windows.
CV_ORIGINS = (63, 72, 81, 90, 99, 108, 117, 126)
CV_INITIAL_STATE_SEEDS = (0, 1, 2)  # synthetic start states for CV only
DEMAND_STRESS = (0.0, 0.10, 0.20)  # multiplicative demand shocks, CV only
ROBUST_WEIGHT_STD = 0.45  # robust = mean + w*std over folds
ROBUST_WEIGHT_WORST = 0.15  # + v*(worst - mean) over folds

# --------------------------------------------------------------------------
# Promotion gate (declared BEFORE any candidate is evaluated)
# --------------------------------------------------------------------------
# A candidate replaces the incumbent only if ALL hold, measured on CV only:
GATE_MIN_ROBUST_IMPROVEMENT = 0.0005  # absolute, robust normalized cost
GATE_MAX_HOLDOUT_DRIFT = 0.05  # provisional research tolerance (5%)

# --------------------------------------------------------------------------
# Randomness
# --------------------------------------------------------------------------
SEED = 20261022

# --------------------------------------------------------------------------
# Reference anchors (from the shipped VN2 reference replay). These are the
# reproduction targets the simulator MUST hit before any model work counts.
# --------------------------------------------------------------------------
ANCHOR_ORGANIZER_BASELINE_WEEKS_3_8 = 4334.00  # published organizer baseline
ANCHOR_ORGANIZER_WEEKLY_SCORED = (784.8, 915.4, 733.2, 603.2, 604.0, 693.4)
ANCHOR_ORGANIZER_HOLDING = 3618.80  # all 8 weeks
ANCHOR_ORGANIZER_SHORTAGE = 1629.00  # all 8 weeks
ANCHOR_POLICY_INDEPENDENT_WEEK1 = 380.6  # weekly cost, game week 1
ANCHOR_POLICY_INDEPENDENT_WEEK2 = 533.2  # weekly cost, game week 2
ANCHOR_PUBLISHED_WINNER = 3763.20  # published winning score

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "raw"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
REPORTS_DIR = PROJECT_ROOT / "reports"
SUBMISSIONS_DIR = PROJECT_ROOT / "submissions"
DATA_SHA256 = PROJECT_ROOT / "data" / "SHA256SUMS.txt"

# --------------------------------------------------------------------------
# File names exactly as downloaded from DataSource.ai
# --------------------------------------------------------------------------
F_INITIAL_STATE = "Week 0 - 2024-04-08 - Initial State.csv"
F_HISTORY_SALES = "Week 0 - 2024-04-08 - Sales.csv"
F_IN_STOCK = "Week 0 - In Stock.csv"
F_MASTER = "Week 0 - Master.csv"
F_SUBMISSION_TEMPLATE = "Week 0 - Submission Template.csv"
GAME_SALES_FILES = (
    "Week 1 - 2024-04-15 - Sales.csv",
    "Week 2 - 2024-04-22 - Sales.csv",
    "Week 3 - 2024-04-29 - Sales.csv",
    "Week 4 - 2024-05-06 - Sales.csv",
    "Week 5 - 2024-05-13 - Sales.csv",
    "Week 6 - 2024-05-20 - Sales.csv",
    "Week 7 - 2024-05-27 - Sales.csv",
    "Week 8 - 2024-06-03 - Sales.csv",
)


def sha256_file(path: Path) -> str:
    """Return the SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def verify_data_hashes(data_dir: Path | None = None) -> dict[str, bool]:
    """Re-verify every downloaded CSV against data/SHA256SUMS.txt.

    The manifest is UTF-16 LE with a BOM and holds ``sha256sum``-style lines
    (``<digest><one or more spaces>[ *]<filename>``). Returns {filename: ok}.
    Raises nothing; callers assert on the result.
    """
    data_dir = Path(data_dir or DATA_DIR)
    expected: dict[str, str] = {}
    if DATA_SHA256.exists():
        text = DATA_SHA256.read_text(encoding="utf-16").lstrip("\ufeff")
        for line in text.splitlines():
            parts = line.strip().split(None, 1)
            if len(parts) != 2:
                continue  # blank line or malformed entry
            digest, name = parts[0].lower(), parts[1].strip()
            if name.startswith("*"):
                name = name[1:]  # binary-mode marker from sha256sum -b
            if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                continue  # not a SHA-256 hex digest
            expected[name] = digest
    out: dict[str, bool] = {}
    for name, digest in expected.items():
        target = data_dir / name
        out[name] = target.is_file() and sha256_file(target) == digest
    return out
