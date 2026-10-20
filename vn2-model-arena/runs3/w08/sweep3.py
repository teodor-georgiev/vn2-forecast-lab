"""Multi-cutoff evaluation of a few candidate configs."""

from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from sweep2 import evaluate, clear_env

CUTS = (60, 80, 100, 120, 130, 140)

if __name__ == "__main__":
    configs = [
        ("sp0.7 base fb", {}),
        ("fb clip .4-2.5", {"VN2_FB_LO": "0.4", "VN2_FB_HI": "2.5"}),
        ("fb clip .3-3", {"VN2_FB_LO": "0.3", "VN2_FB_HI": "3"}),
        ("no feedback", {"VN2_FBPOW": "0"}),
        ("fb pow 0.7", {"VN2_FBPOW": "0.7"}),
        ("fb shrink 2", {"VN2_SHRINK": "2"}),
    ]
    for label, env in configs:
        clear_env()
        os.environ.update(env)
        evaluate(label, cuts=CUTS)
