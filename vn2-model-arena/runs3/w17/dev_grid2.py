"""Dev: grid search over honest truncated cutoffs."""

from __future__ import annotations
import time
import numpy as np
import hb_local as hb
from dev_bt import S, INIT, truncate, ref_decide_factory

CUTS = [60, 100, 130, 145]
BASE = {
    "hist_weeks_level": 13,
    "halflife": 8.0,
    "q": 0.88,
    "q_mode": "level",
    "q_beta": 1.0,
    "q_lo": 0.5,
    "q_hi": 0.99,
    "group_cols": ("DepartmentGroup", "StoreFormat"),
    "seasonal_power": 1.0,
    "seasonal_smooth": 3,
    "level_shrink_a": 0.5,
    "share_blend": 0.5,
    "kappa_shrink": 2.0,
    "m_k": 1.0,
    "m_clip": (0.2, 5.0),
    "proj_w": 1.0,
    "fill_censored": True,
}


def evaluate(cutoffs=CUTS, initial=None):
    import policy

    tot_h = tot_s = 0.0
    sc = []
    for h in cutoffs:
        truncate(h)
        r = hb.backtest(policy.decide, S, h, initial=initial)
        tot_h += sum(x["holding"] for x in r["weekly"][2:])
        tot_s += sum(x["shortage"] for x in r["weekly"][2:])
        sc.append(r["score"])
    return float(np.mean(sc)), tot_h, tot_s


def ref_mean(cutoffs=CUTS, initial=None):
    return float(
        np.mean(
            [
                hb.backtest(ref_decide_factory(h), S, h, initial=initial)["score"]
                for h in cutoffs
            ]
        )
    )


def sweep(cfgs, cutoffs=CUTS, base=BASE):
    import policy

    rows = []
    t0 = time.perf_counter()
    for label, cfg in cfgs:
        policy._CFG.clear()
        policy._CFG.update(base)
        policy._CFG.update(cfg)
        policy._PROFILE_CACHE.clear()
        z, zh, zs = evaluate(cutoffs, None)
        i, ih, is_ = evaluate(cutoffs, INIT)
        rows.append((0.5 * (z + i), label, z, i, zh, zs, ih, is_))
        print(
            f"{label:22s} both {0.5 * (z + i):8.0f} | zeros {z:8.0f} (h{zh:6.0f}/s{zs:6.0f})"
            f" | init {i:8.0f} (h{ih:6.0f}/s{is_:6.0f})"
        )
        with open("results.txt", "a") as fh:
            fh.write(
                f"{label:22s} both {0.5 * (z + i):8.0f} zeros {z:8.0f} init {i:8.0f}\n"
            )
    print(f"({time.perf_counter() - t0:.1f}s)")
    return rows


if __name__ == "__main__":
    print("ref13 zeros", round(ref_mean()), "init", round(ref_mean(initial=INIT)))
    cfgs = [("default", {})]
    for q in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95):
        cfgs.append((f"fixed q={q}", {"q_mode": "fixed", "q": q}))
    for b in (0.3, 0.6, 2.0, 4.0):
        cfgs.append((f"q_beta={b}", {"q_beta": b}))
    cfgs.append(("noseason", {"seasonal_power": 0.0}))
    cfgs.append(("season^0.5", {"seasonal_power": 0.5}))
    cfgs.append(("noseason+q0.8", {"seasonal_power": 0.0, "q_mode": "fixed", "q": 0.8}))
    for L in (6, 8, 20, 26):
        cfgs.append((f"L={L}", {"hist_weeks_level": L}))
    for hl in (3.0, 5.0, 13.0, 200.0):
        cfgs.append((f"hl={hl}", {"halflife": hl}))
    for pw in (0.8, 1.2, 1.5):
        cfgs.append((f"proj_w={pw}", {"proj_w": pw}))
    for mk in (0.2, 3.0):
        cfgs.append((f"m_k={mk}", {"m_k": mk}))
    cfgs.append(("no_cens_fix", {"fill_censored": False}))
    for gc in (
        ("Department",),
        ("ProductGroup",),
        ("DepartmentGroup",),
        ("DepartmentGroup", "StoreFormat", "Format"),
        ("Division",),
    ):
        cfgs.append((f"grp={'+'.join(gc)}", {"group_cols": gc}))
    cfgs.append(("no_share(own)", {"share_blend": 0.0}))
    cfgs.append(("heavy_family", {"share_blend": 3.0}))
    cfgs.append(("kappa_shrink=0", {"kappa_shrink": 0.0}))
    cfgs.append(("kappa_shrink=10", {"kappa_shrink": 10.0}))
    sweep(cfgs)
