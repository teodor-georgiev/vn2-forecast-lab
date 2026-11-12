"""Gate 1: the downloaded CSVs must match data/SHA256SUMS.txt before any load."""

import shutil

from vn2_from_scratch.config import (
    DATA_DIR,
    DATA_SHA256,
    F_INITIAL_STATE,
    verify_data_hashes,
)
from vn2_from_scratch.data import load_dataset


def test_manifest_lists_every_input_and_all_match():
    ok = verify_data_hashes()
    assert len(ok) == 13, f"manifest parsed {len(ok)} entries"
    assert all(ok.values()), [name for name, good in ok.items() if not good]


def test_load_dataset_verifies_and_succeeds():
    ds = load_dataset()
    assert ds.n == 599


def test_corrupted_copy_is_reported_false(tmp_path):
    target = tmp_path / F_INITIAL_STATE
    shutil.copyfile(DATA_DIR / F_INITIAL_STATE, target)
    assert verify_data_hashes(tmp_path)[F_INITIAL_STATE] is True

    raw = bytearray(target.read_bytes())
    raw[len(raw) // 2] ^= 0xFF
    target.write_bytes(bytes(raw))

    assert verify_data_hashes(tmp_path)[F_INITIAL_STATE] is False
    assert DATA_SHA256.is_file(), "manifest itself must stay untouched"
