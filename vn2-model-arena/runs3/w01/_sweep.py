"""Sweep several historical cutoffs: my policy vs the arena reference rule."""

import importlib
import sys
import time

import numpy as np

sys.path.insert(0, r"C:\Users\Toni\Documents\Default Project\vn2-model-arena\runs3\w01")
import exp  # noqa: E402

S, INIT = exp.S, exp.INIT
CUTS = tuple(range(56, 150, 6))


def mine(cfg, cuts=CUTS, shuffle=False):
    pol = importlib.import_module("policy")
    importlib.reload(pol)
    out = {}
    for c in cuts:
        pol.fit(cutoff=c, cfg=cfg)
        init = INIT[exp.PERM] if shuffle else INIT
        out[c] = exp.replay(pol.decide, S, c, init)
    return out


def ref(cuts=CUTS):
    sys.path.insert(0, r"C:\Users\Toni\Documents\Default Project\vn2-model-arena\tools")
    import inventory_sim as inv

    out = {}
    for c in cuts:
        mu = S[:, :c].mean(1)
        out[c] = inv.order_plan_from_policy(
            S[:, c : c + 8].T,
            INIT,
            lambda w, st, h, mu=mu: np.array([mu, mu, mu]),
            lambda f, st: inv.reference_orders(f, st),
        )["score"]
    return out


if __name__ == "__main__":
    t0 = time.perf_counter()
    R = ref()
    print(
        "REF  ",
        {k: round(v) for k, v in R.items()},
        "mean",
        round(np.mean(list(R.values()))),
    )
    print(f"[{time.perf_counter() - t0:.0f}s]")
    for cfg in [
        {"tau_mode": "global", "tau_global": t}
        for t in (0.70, 0.80, 0.833, 0.85, 0.90, 0.95)
    ]:
        sc = mine(cfg)
        wins = sum(1 for k in R if sc[k] < R[k])
        print(
            f"MINE tau={cfg['tau_global']:.3f}",
            {k: round(v) for k, v in sc.items()},
            "mean",
            round(np.mean(list(sc.values()))),
            f"wins {wins}/{len(R)}",
        )
    print(f"total {time.perf_counter() - t0:.0f}s")
