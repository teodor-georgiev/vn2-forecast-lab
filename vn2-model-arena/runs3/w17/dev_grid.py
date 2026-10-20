"""Dev: grid search + cost breakdown over honest truncated cutoffs."""

from __future__ import annotations
import itertools, sys, time
import numpy as np
import hb_local as hb
from dev_bt import S, INIT, truncate, ref_decide_factory

CUTS = [60, 100, 130, 145]


def evaluate(policy, cutoffs=CUTS, initial=None, detail=False):
    tot_h = tot_s = 0.0
    out = {}
    for h in cutoffs:
        truncate(h)
        res = hb.backtest(policy.decide, S, h, initial=initial)
        hold = sum(r["holding"] for r in res["weekly"][2:])
        short = sum(r["shortage"] for r in res["weekly"][2:])
        tot_h += hold
        tot_s += short
        out[h] = res["score"]
    out["_hold"] = tot_h
    out["_short"] = tot_s
    out["_mean"] = float(np.mean([out[h] for h in cutoffs]))
    return out


def ref_scores(cutoffs=CUTS, initial=None):
    out = {}
    for h in cutoffs:
        out[h] = hb.backtest(ref_decide_factory(h), S, h, initial=initial)["score"]
    out["_mean"] = float(np.mean(list(out.values())))
    return out


def apply(policy, cfg):
    policy._CFG.update(cfg)
    policy._PROFILE_CACHE.clear()


if __name__ == "__main__":
    import policy

    t0 = time.perf_counter()
    print("ref13 (zeros):", {k: round(v) for k, v in ref_scores().items()})
    print("ref13 (INIT) :", {k: round(v) for k, v in ref_scores(initial=INIT).items()})
    base = np.mean([ref_scores()["_mean"], ref_scores(initial=INIT)["_mean"]])
    print("ref mean of both:", round(base))

    for label, cfg in [
        ("default", {}),
        ("q=0.5", {"q": 0.5}),
        ("q=0.65", {"q": 0.65}),
        ("q=0.75", {"q": 0.75}),
        ("q=0.95", {"q": 0.95}),
        ("q=0.99", {"q": 0.99}),
        ("noseason", {"seasonal_power": 0.0}),
        ("halflife=4", {"halflife": 4.0}),
        ("halflife=16", {"halflife": 16.0}),
        ("L=8", {"hist_weeks_level": 8}),
        ("L=26", {"hist_weeks_level": 26}),
        ("m_k=0.2", {"m_k": 0.2}),
        ("m_k=5", {"m_k": 5.0}),
        ("no_kappa_shrink", {"kappa_shrink": 0.0}),
        ("grp=Department", {"group_cols": ("Department",)}),
        ("grp=ProductGroup", {"group_cols": ("ProductGroup",)}),
        ("grp=DeptGroup", {"group_cols": ("DepartmentGroup",)}),
        ("DL=all", {"group_cols": ("DepartmentGroup", "StoreFormat", "Format")}),
    ]:
        policy._CFG.clear()
        policy._CFG.update(
            {
                "hist_weeks_level": 13,
                "halflife": 8.0,
                "q": 0.88,
                "group_cols": ("DepartmentGroup", "StoreFormat"),
                "seasonal_power": 1.0,
                "seasonal_smooth": 3,
                "level_shrink_a": 0.5,
                "share_blend": 0.5,
                "kappa_shrink": 2.0,
                "m_k": 1.0,
                "m_clip": (0.2, 5.0),
                "fill_censored": True,
            }
        )
        apply(policy, cfg)
        z = evaluate(policy, initial=None)
        i = evaluate(policy, initial=INIT)
        print(
            f"{label:16s} zeros {z['_mean']:8.0f} (h{z['_hold']:7.0f}/s{z['_short']:7.0f})"
            f"  init {i['_mean']:8.0f} (h{i['_hold']:7.0f}/s{i['_short']:7.0f})"
            f"  both {0.5 * (z['_mean'] + i['_mean']):8.0f}"
        )
    print(f"grid elapsed {time.perf_counter() - t0:.1f}s")
