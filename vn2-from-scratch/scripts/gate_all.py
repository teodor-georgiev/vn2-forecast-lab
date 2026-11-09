"""All-gates runner: data hashes + anchor gate + organizer baseline gate.

Runs ``verify_data_hashes()`` (import) and ``scripts/check_anchors.py`` /
``scripts/verify_organizer_baseline.py`` (subprocess, each of which calls
``simulate()``). Prints a summary and returns nonzero if ANY check fails.
No model logic lives here.

Usage: python scripts/gate_all.py
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vn2_from_scratch.config import verify_data_hashes  # noqa: E402

_SCRIPTS = ("check_anchors.py", "verify_organizer_baseline.py")


def _run_gate(script_name: str) -> int:
    print(f"\n=== {script_name} ===", flush=True)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(PROJECT_ROOT)
    proc = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / script_name)],
        cwd=PROJECT_ROOT,
        env=env,
    )
    return proc.returncode


def main() -> int:
    failed: list[str] = []

    hashes = verify_data_hashes()
    bad = [name for name, ok in hashes.items() if not ok]
    print(f"data hashes: {len(hashes) - len(bad)}/{len(hashes)} match")
    for name in bad:
        print(f"  FAIL  {name}")
    if not hashes or bad:
        failed.append("data hashes")

    for script_name in _SCRIPTS:
        if _run_gate(script_name) != 0:
            failed.append(script_name)

    print(
        f"\nGATE {'FAILED' if failed else 'PASSED'}  ({len(failed)} failing check(s): {failed})"
    )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
