"""Submission format gate: anything the validator accepts must survive a round-trip."""

import numpy as np
import pandas as pd
import pytest

from vn2_from_scratch.submit import (
    build_submission,
    template,
    validate_submission,
    write_submission,
    write_rounds,
)


def test_round_trip(tmp_path):
    rng = np.random.default_rng(0)
    orders = rng.integers(0, 500, size=(6, 599)).astype(float)
    df = build_submission(orders)
    assert list(df.columns) == ["Store", "Product", "Order"]
    assert len(df) == 599
    path = write_submission(df, tmp_path / "s.csv")
    back = pd.read_csv(path)
    assert back.shape == (599, 3)
    assert np.array_equal(
        back[["Store", "Product"]].to_numpy(),
        template()[["Store", "Product"]].to_numpy(),
    )
    assert (back["Order"] >= 0).all()
    assert back["Order"].dtype.kind in "iu"


def test_zero_and_large_orders_validate():
    for qty in (np.zeros((6, 599)), np.full((6, 599), 1e6)):
        df = build_submission(qty)
        validate_submission(df)


def test_row_order_must_match_template():
    df = build_submission(np.zeros((6, 599)))
    shuffled = df.sample(frac=1.0, random_state=1).reset_index(drop=True)
    with pytest.raises(ValueError, match="index differs"):
        validate_submission(shuffled)


def test_missing_negative_and_non_integer_rejected():
    df = build_submission(np.zeros((6, 599)))
    bad = df.copy()
    bad.loc[3, "Order"] = np.nan
    with pytest.raises(ValueError, match="missing"):
        validate_submission(bad)

    bad = df.copy()
    bad.loc[3, "Order"] = -1
    with pytest.raises(ValueError, match="negative"):
        validate_submission(bad)

    bad = df.copy()
    bad["Order"] = bad["Order"].astype(float) + 0.5
    with pytest.raises(ValueError, match="integer"):
        validate_submission(bad)


def test_write_rounds_emits_six_valid_files(tmp_path):
    paths = write_rounds(np.arange(6 * 599).reshape(6, 599).astype(float), tmp_path)
    assert len(paths) == 6
    for p in paths:
        validate_submission(pd.read_csv(p))
