"""Load and validate the VN2 competition CSVs.

Everything here is deterministic and hash-verified. `assert_contract` is the
gate that must pass before any modelling work is trusted.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import (
    DATA_DIR,
    F_HISTORY_SALES,
    F_IN_STOCK,
    F_INITIAL_STATE,
    F_MASTER,
    F_SUBMISSION_TEMPLATE,
    GAME_SALES_FILES,
    GAME_WEEKS,
    HISTORY_WEEKS,
    N_SERIES,
    verify_data_hashes,
)


@dataclass(frozen=True)
class Dataset:
    keys: np.ndarray  # (599, 2) int64 [store, product], submission order
    history_dates: list[str]
    history: np.ndarray  # (599, 157) observed weekly sales
    in_stock: np.ndarray  # (599, 157) bool availability aligned to history
    in_stock_full: np.ndarray  # (599, 165) bool availability through 2024-06-03
    game_dates: list[str]
    game_sales: np.ndarray  # (599, 8) EXPOSED labels, game weeks 1..8
    on_hand0: np.ndarray  # (599,) opening on-hand
    pipe_w1: np.ndarray  # (599,) arrives game week 1
    pipe_w2: np.ndarray  # (599,) arrives game week 2
    master: pd.DataFrame

    @property
    def n(self) -> int:
        return self.keys.shape[0]


def load_dataset(data_dir: Path | None = None) -> Dataset:
    d = Path(data_dir or DATA_DIR)

    # Gate 1: every input CSV must match data/SHA256SUMS.txt before it is read.
    mismatched = sorted(name for name, ok in verify_data_hashes(d).items() if not ok)
    if mismatched:
        raise ValueError(
            f"SHA-256 mismatch for {len(mismatched)} input CSV(s) in {d}: "
            + ", ".join(mismatched)
        )

    hist = pd.read_csv(d / F_HISTORY_SALES)
    stock = pd.read_csv(d / F_IN_STOCK)
    init = pd.read_csv(d / F_INITIAL_STATE)
    master = pd.read_csv(d / F_MASTER)
    template = pd.read_csv(d / F_SUBMISSION_TEMPLATE)

    keys = template[["Store", "Product"]].to_numpy(dtype=np.int64)

    history_dates = [c for c in hist.columns if c not in ("Store", "Product")]
    in_stock_dates = [c for c in stock.columns if c not in ("Store", "Product")]

    history = hist[history_dates].to_numpy(dtype=np.float64)
    in_stock_full = stock[in_stock_dates].to_numpy(dtype=bool)
    in_stock = in_stock_full[:, :HISTORY_WEEKS]

    # Game labels: each week-k file appends exactly one new date column.
    game_cols: list[str] = []
    game_vals: list[np.ndarray] = []
    for name in GAME_SALES_FILES:
        wk = pd.read_csv(d / name)
        newest = [c for c in wk.columns if c not in ("Store", "Product")][-1]
        game_cols.append(newest)
        game_vals.append(wk[newest].to_numpy(dtype=np.float64))
    game_sales = np.vstack(game_vals).T  # (599, 8)

    ordered = init.set_index(["Store", "Product"]).reindex(
        pd.MultiIndex.from_arrays([keys[:, 0], keys[:, 1]], names=["Store", "Product"])
    )
    on_hand0 = ordered["End Inventory"].to_numpy(dtype=np.float64)
    pipe_w1 = ordered["In Transit W+1"].to_numpy(dtype=np.float64)
    pipe_w2 = ordered["In Transit W+2"].to_numpy(dtype=np.float64)

    ds = Dataset(
        keys=keys,
        history_dates=history_dates,
        history=history,
        in_stock=in_stock,
        in_stock_full=in_stock_full,
        game_dates=game_cols,
        game_sales=game_sales,
        on_hand0=on_hand0,
        pipe_w1=pipe_w1,
        pipe_w2=pipe_w2,
        master=master,
    )
    assert_contract(ds)
    return ds


def assert_contract(ds: Dataset) -> None:
    """Raise AssertionError on any geometry or ordering mismatch."""
    k = ds.keys
    assert k.shape == (N_SERIES, 2), f"keys shape {k.shape}"
    assert ds.history.shape == (N_SERIES, HISTORY_WEEKS), ds.history.shape
    assert ds.in_stock.shape == (N_SERIES, HISTORY_WEEKS), ds.in_stock.shape
    assert ds.in_stock_full.shape == (N_SERIES, HISTORY_WEEKS + GAME_WEEKS), (
        ds.in_stock_full.shape
    )
    assert ds.game_sales.shape == (N_SERIES, GAME_WEEKS), ds.game_sales.shape
    for name, arr in (
        ("on_hand0", ds.on_hand0),
        ("pipe_w1", ds.pipe_w1),
        ("pipe_w2", ds.pipe_w2),
    ):
        assert arr.shape == (N_SERIES,), f"{name} shape {arr.shape}"
        assert np.isfinite(arr).all(), f"{name} has non-finite values"
        assert (arr >= 0).all(), f"{name} has negative values"

    assert len(ds.history_dates) == HISTORY_WEEKS
    assert len(ds.game_dates) == GAME_WEEKS
    assert ds.history_dates[-1] <= ds.game_dates[0], "history and game weeks overlap"
    assert np.isfinite(ds.history).all(), "history has non-finite values"
    assert (ds.history >= 0).all(), "history has negative sales"
    assert np.isfinite(ds.game_sales).all(), "game labels have non-finite values"

    # Key order must match the submission template exactly, row for row.
    tpl = pd.read_csv(DATA_DIR / F_SUBMISSION_TEMPLATE)[["Store", "Product"]].to_numpy(
        dtype=np.int64
    )
    assert k.shape == tpl.shape, "key count differs from submission template"
    assert (k == tpl).all(), "key ORDER differs from submission template"
    assert len(np.unique(k, axis=0)) == N_SERIES, "duplicate store x product keys"

    # The week-k sales files must be strict extensions of the Week 0 history.
    hist_idx = pd.MultiIndex.from_arrays([k[:, 0], k[:, 1]])
    for name in (GAME_SALES_FILES[0],):
        wk = (
            pd.read_csv(DATA_DIR / name)
            .set_index(["Store", "Product"])
            .reindex(hist_idx)
        )
        tail = [c for c in wk.columns][-1]
        prev = [c for c in wk.columns][-2]
        assert not wk[[tail, prev]].isna().any().any(), f"{name} has misaligned rows"


def align_history(ds: Dataset, upto: int) -> tuple[np.ndarray, np.ndarray]:
    """Causal slice of history and availability ending just before week ``upto``."""
    assert 0 <= upto <= HISTORY_WEEKS
    return ds.history[:, :upto], ds.in_stock[:, :upto]
