"""w02: grid search over the policy knobs, scored on several historical cutoffs."""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import tune  # noqa: E402
import policy as pol  # noqa: E402

STARTS = (30, 40, 60, 80, 100, 115, 120, 130, 140, 145)


def evaluate(params, starts=STARTS, initial="proxy"):
    out = {}
    for s in starts:
        pol.HISTORY_END = s
        pol.GAME_DOY = tune.DOY[s : s + 8].tolist()
        st = tune.make_initial(s) if initial == "proxy" else None
        out[s] = tune.backtest(pol.decide, tune.SALES, s, initial=st)["score"]
    return out


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "q"
    base = dict(pol.PARAMS)
    base.update(dict(blend=0.9, q=0.6, alpha=2.0))
    if mode == "q":
        grid = [
            dict(q=q, W=w, start_off=o)
            for q, w, o in itertools.product(
                (0.45, 0.55, 0.65, 0.75, 0.85), (2, 3, 4), (0, 1)
            )
        ]
    elif mode == "alpha":
        grid = [
            dict(alpha=a, q=q, damp=d)
            for a, q, d in itertools.product(
                (0.0, 0.5, 1.0, 2.0), (0.6, 0.7, 0.8), (1.0,)
            )
        ]
    elif mode == "hl":
        grid = [
            dict(hl=h, q=q, shrink_k=k)
            for h, q, k in itertools.product(
                (8.0, 13.0, 26.0, 52.0), (0.6, 0.7, 0.8), (8.0, 20.0)
            )
        ]
    elif mode == "blend":
        grid = [
            dict(blend=bl, q=q, level_scale=ls)
            for bl, q, ls in itertools.product(
                (0.0, 0.3, 0.5, 0.7, 1.0), (0.55, 0.65, 0.75), (1.0, 1.1)
            )
        ]
    elif mode == "alpha2":
        grid = [
            dict(alpha=a, q=q, blend=bl)
            for a, q, bl in itertools.product(
                (0.0, 0.5, 1.0, 2.0), (0.55, 0.65, 0.75), (0.3, 0.5, 0.7)
            )
        ]
    elif mode == "final":
        grid = [
            dict(blend=bl, q=q, level_scale=ls, alpha=a)
            for bl, q, ls, a in itertools.product(
                (0.5, 0.7, 0.9), (0.6, 0.7, 0.8), (1.0, 1.1, 1.2), (1.0, 2.0)
            )
        ]
    elif mode == "tail":
        grid = [
            dict(q=q, q4=q4, q5=q5)
            for q, q4, q5 in itertools.product(
                (0.55, 0.65), (0.65, 0.75, 0.85), (0.75, 0.85, 0.95)
            )
        ]
    elif mode == "deficit":
        grid = [
            dict(mode="deficit", q1=q1, beta=b, blend=bl, alpha=a)
            for q1, b, bl, a in itertools.product(
                (0.6, 0.7, 0.8, 0.9), (0.0, 0.5, 1.0), (0.7, 0.9), (1.0, 2.0)
            )
        ]
    elif mode == "deficit2":
        grid = [
            dict(mode="deficit", q1=q1, q4=q4, q5=q5, beta=b, blend=bl, alpha=a)
            for q1, q4, q5, b, bl, a in itertools.product(
                (0.6, 0.7),
                (0.7, 0.8, 0.9),
                (0.8, 0.9, 0.95),
                (0.0,),
                (0.7, 0.9),
                (1.0, 2.0),
            )
        ]
    elif mode == "news":
        grid = [
            dict(mode="news", k_over=k, beta=b, blend=bl, alpha=a)
            for k, b, bl, a in itertools.product(
                (0.15, 0.3, 0.5, 0.8, 1.2), (0.0,), (0.7, 0.9), (1.0, 2.0)
            )
        ]
    elif mode == "final2":
        grid = [
            dict(mode="deficit", q1=q1, q4=q4, q5=q5, blend=bl, alpha=a)
            for q1, q4, q5, bl, a in itertools.product(
                (0.65, 0.7, 0.75),
                (0.7, 0.8, 0.9),
                (0.85, 0.9, 0.95),
                (0.85, 0.9),
                (1.0, 2.0),
            )
        ]
    elif mode == "hl2":
        grid = [
            dict(hl=h, raw_hl=r, q1=q1)
            for h, r, q1 in itertools.product(
                (6.0, 8.0, 13.0, 26.0), (6.0, 13.0), (0.65, 0.7, 0.75)
            )
        ]
    elif mode == "final3":
        grid = [
            dict(q1=q1, q4=0.8, q5=0.9, blend=bl, alpha=a, fb_c=c, hl=8.0, raw_hl=13.0)
            for q1, bl, a, c in itertools.product(
                (0.65, 0.7, 0.75),
                (0.85, 0.9, 1.0),
                (0.4, 0.55, 0.7, 0.85),
                (0.5, 1.0, 2.0),
            )
        ]
    else:
        grid = [dict()]

    rows = []
    for g in grid:
        p = dict(base)
        p.update(g)
        pol.PARAMS.clear()
        pol.PARAMS.update(p)
        pol._CACHE.clear()  # the prepare cache must follow the params
        sc = evaluate(p)
        tot = sum(sc.values())
        rows.append((tot, g, sc))
        print(
            f"{g}  "
            + "  ".join(f"{s}:{v:,.0f}" for s, v in sc.items())
            + f"   TOT {tot:,.0f}"
        )
    rows.sort(key=lambda r: r[0])
    print("\nBEST:", rows[0][1], f"{rows[0][0]:,.0f}")

