"""Config sweep for the w08 policy (experiment script)."""

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


def score(cutoff, truth="raw", policy_mod="policy"):
    os.environ["VN2_CUTOFF"] = str(cutoff)
    sys.modules.pop(policy_mod, None)
    mod = importlib.import_module(policy_mod)
    h = Y if truth == "raw" else D
    t0 = time.perf_counter()
    res = hb.backtest(mod.decide, h, cutoff)
    el = time.perf_counter() - t0
    return res["score"], el, res


def evaluate(label, cuts=(100, 130), truths=("raw", "corr")):
    rows = []
    tmax = 0.0
    for truth in truths:
        for c in cuts:
            s, el, res = score(c, truth)
            rows.append(s)
            tmax = max(tmax, el)
    print(
        f"{label:38s} mean={np.mean(rows):8.1f}  "
        + "  ".join(
            f"{t}@{c}={s:7.0f}"
            for t, s in zip([f"{t}" for t in truths for c in cuts], rows)
        )
        + f"   tmax={tmax:4.1f}s"
    )
    return float(np.mean(rows))


if __name__ == "__main__":
    import itertools

    configs = []
    for pool in (26, 52, 104):
        configs.append((f"pool={pool}", {"VN2_POOL_WEEKS": str(pool)}))
    for shrink in (0.5, 2.0, 4.0):
        configs.append((f"shrink={shrink}", {"VN2_SHRINK": str(shrink)}))
    for hp in (0.0, 1.0):
        configs.append((f"histpow={hp}", {"VN2_HISTPOW": str(hp)}))
    for th in (0.0, 1.0):
        configs.append((f"theta={th}", {"VN2_THETA": str(th)}))
    configs.append(("base", {}))

    base_env = {k: v for k, v in os.environ.items() if k.startswith("VN2_")}
    for k in list(os.environ):
        if k.startswith("VN2_"):
            del os.environ[k]
    os.environ.update(base_env)

    for label, env in configs:
        for k in list(os.environ):
            if k.startswith("VN2_"):
                del os.environ[k]
        os.environ.update(env)
        evaluate(label)
