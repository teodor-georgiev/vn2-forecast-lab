"""Sweep model params; objective = mean score over historical cutoffs."""

import itertools
import sys
import numpy as np
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "tools"))
from exp import Model, score, load  # noqa: E402
import hb2  # noqa: E402
from inventory_sim import reference_orders  # noqa: E402

CUTS = [53, 79, 105, 130, 149]


def ref_rule(window=13, buffer=0.75):
    Y, A, D, init_s = load()

    def decide(w, state, revealed):
        d = D[:, : decide.base]
        if w:
            d = np.column_stack((d, np.asarray(revealed).T))
        f = d[:, -window:].mean(1)
        return reference_orders([f, f, f], state, buffer=buffer)

    decide.base = None
    return decide


def show(rows, top=10, label=""):
    print(f"--- {label}")
    for r in rows[:top]:
        print(
            f"mean {r[0]:7.0f} max {r[1]:7.0f} "
            + " ".join(f"{x:6.0f}" for x in r[2])
            + "  "
            + str(r[3])
        )


def grid(base_kw, grid_kw, cuts=CUTS, top=10, label=""):
    keys = list(grid_kw)
    rows = []
    for vals in itertools.product(*[grid_kw[k] for k in keys]):
        kw = dict(base_kw)
        kw.update(dict(zip(keys, vals)))
        m = Model(**kw)
        s = score(m, cuts)
        v = np.array([s[c] for c in cuts])
        rows.append((v.mean(), v.max(), v, dict(zip(keys, vals))))
    rows.sort(key=lambda r: r[0])
    show(rows, top, label)
    return rows


if __name__ == "__main__":
    Y, A, D, init_s = load()
    pol = ref_rule()
    ref = []
    for c in CUTS:
        pol.base = c
        ref.append(hb2.backtest(pol, D, start_week=c, initial=init_s)["score"])
    print(
        "reference(D,init):",
        " ".join(f"{x:6.0f}" for x in ref),
        "mean",
        round(np.mean(ref)),
    )

    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "lev"):
        grid(
            dict(q_A=0.8, q_B=0.8, q_C=0.8),
            dict(
                win_short=[6, 10, 13, 20],
                w_recent=[0.7, 0.85, 1.0],
                use_trend=[True, False],
                trend_shrink=[0.5, 1.0],
                seas_mode=["global", "blend"],
            ),
            top=10,
            label="level/trend/season",
        )
    if which in ("all", "q"):
        grid(
            dict(
                win_short=13,
                w_recent=0.85,
                use_trend=True,
                trend_shrink=0.5,
                seas_mode="global",
            ),
            dict(
                q_A=[0.6, 0.7, 0.75, 0.8, 0.9],
                q_B=[0.6, 0.7, 0.75, 0.8, 0.9],
                q_C=[0.6, 0.7, 0.75, 0.8, 0.9],
            ),
            top=10,
            label="quantiles",
        )
