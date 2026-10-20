"""Config sweep (v2) for the w08 policy."""

from __future__ import annotations

import importlib
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import hb_fixed as hb
from work import Y, D

CUTS = (100, 130)


def run_one(cutoff, truth, policy_mod="policy"):
    os.environ["VN2_CUTOFF"] = str(cutoff)
    sys.modules.pop(policy_mod, None)
    mod = importlib.import_module(policy_mod)
    h = Y if truth == "raw" else D
    t0 = time.perf_counter()
    res = hb.backtest(mod.decide, h, cutoff)
    return res["score"], time.perf_counter() - t0


def evaluate(label, cuts=CUTS, truths=("raw",)):
    vals, tmax = [], 0.0
    for t in truths:
        for c in cuts:
            s, el = run_one(c, t)
            vals.append(s)
            tmax = max(tmax, el)
    print(
        f"{label:34s} "
        + " ".join(f"{v:8.0f}" for v in vals)
        + f" | mean={np.mean(vals):8.0f}  tmax={tmax:4.1f}s",
        flush=True,
    )
    return float(np.mean(vals))


def clear_env():
    for k in list(os.environ):
        if k.startswith("VN2_") and k != "VN2_CUTOFF":
            del os.environ[k]


if __name__ == "__main__":
    configs = [
        ("base(sp1,ls13,p52,lw13)", {}),
        ("season_pow=0", {"VN2_SEASON_POW": "0"}),
        ("season_pow=0.5", {"VN2_SEASON_POW": "0.5"}),
        ("season_pow=1.5", {"VN2_SEASON_POW": "1.5"}),
        (
            "ls26+lw8+th1",
            {"VN2_LEVEL_SHRINK": "26", "VN2_LEVEL_WEEKS": "8", "VN2_THETA": "1"},
        ),
        (
            "ls26+lw8+th1+sp0.7",
            {
                "VN2_LEVEL_SHRINK": "26",
                "VN2_LEVEL_WEEKS": "8",
                "VN2_THETA": "1",
                "VN2_SEASON_POW": "0.7",
            },
        ),
        (
            "ls39+lw8+th1",
            {"VN2_LEVEL_SHRINK": "39", "VN2_LEVEL_WEEKS": "8", "VN2_THETA": "1"},
        ),
        (
            "ls26+lw13+th1",
            {"VN2_LEVEL_SHRINK": "26", "VN2_LEVEL_WEEKS": "13", "VN2_THETA": "1"},
        ),
        (
            "ls26+lw8+th0",
            {"VN2_LEVEL_SHRINK": "26", "VN2_LEVEL_WEEKS": "8", "VN2_THETA": "0"},
        ),
        (
            "pool78+ls26+lw8",
            {"VN2_POOL_WEEKS": "78", "VN2_LEVEL_SHRINK": "26", "VN2_LEVEL_WEEKS": "8"},
        ),
    ]
    for label, env in configs:
        clear_env()
        os.environ.update(env)
        evaluate(label)
