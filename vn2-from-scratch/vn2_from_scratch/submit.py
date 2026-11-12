"""Submission construction and format validation.

The competition rejects a submission outright on any format error, so the
validator names the specific failure rather than returning a boolean.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import DATA_DIR, F_SUBMISSION_TEMPLATE, N_SERIES, N_DECISIONS

ORDER_COLUMN = "Order"


def template() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / F_SUBMISSION_TEMPLATE)[["Store", "Product"]]


def build_submission(orders: np.ndarray) -> pd.DataFrame:
    """Wrap a (6, 599) order matrix into the single-column submission format.

    The platform asks for one submission file per round; each contains the
    quantity for that round only. This returns the round-1 file by default via
    `round_index`, and all six are written by `write_rounds`.
    """
    orders = np.asarray(orders, dtype=np.float64)
    if orders.shape != (N_DECISIONS, N_SERIES):
        raise ValueError(
            f"orders must be ({N_DECISIONS}, {N_SERIES}), got {orders.shape}"
        )
    return _wrap(orders[0])


def _wrap(qty: np.ndarray) -> pd.DataFrame:
    qty = np.asarray(qty, dtype=np.float64)
    if not np.isfinite(qty).all():
        raise ValueError("order quantities contain NaN or infinite values")
    if (qty < 0).any():
        raise ValueError("order quantities must be non-negative")
    tpl = template()
    out = tpl.copy()
    out[ORDER_COLUMN] = np.round(qty).astype(np.int64)
    return out


def write_submission(df: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, sep=",")
    return path


def write_rounds(orders: np.ndarray, out_dir: str | Path) -> list[Path]:
    """Write one valid submission file per round."""
    out_dir = Path(out_dir)
    orders = np.asarray(orders, dtype=np.float64)
    paths = []
    for r in range(orders.shape[0]):
        paths.append(
            write_submission(
                _wrap(orders[r]), out_dir / f"submission_round_{r + 1}.csv"
            )
        )
    return paths


def validate_submission(df: pd.DataFrame, tpl: pd.DataFrame | None = None) -> None:
    """Raise ValueError naming the exact reason a submission would be rejected."""
    tpl = template() if tpl is None else tpl
    if list(df.columns) != ["Store", "Product", ORDER_COLUMN]:
        raise ValueError(
            f"columns must be ['Store','Product','{ORDER_COLUMN}'], got {list(df.columns)}"
        )
    if len(df) != N_SERIES:
        raise ValueError(f"expected {N_SERIES} rows, got {len(df)}")
    got = df[["Store", "Product"]].to_numpy()
    want = tpl[["Store", "Product"]].to_numpy()
    if not np.array_equal(got, want):
        mismatch = int((got != want).any(axis=1).sum())
        raise ValueError(
            f"(Store, Product) index differs from the template in {mismatch} rows; "
            "row order must match the submission template exactly"
        )
    qty = df[ORDER_COLUMN]
    if qty.isna().any():
        raise ValueError(f"{int(qty.isna().sum())} missing order quantities")
    if (qty < 0).any():
        raise ValueError(f"{int((qty < 0).sum())} negative order quantities")
    if not np.array_equal(qty.to_numpy(), np.round(qty.to_numpy())):
        raise ValueError("order quantities must be integers")
