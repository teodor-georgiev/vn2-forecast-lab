"""Fast multi-cutoff sweep: fit once per cutoff, then vary the per-series target."""

import importlib
import sys

import numpy as np

sys.path.insert(0, r"C:\Users\Toni\Documents\Default Project\vn2-model-arena\runs3\w01")
import exp  # noqa: E402

S, INIT = exp.S, exp.INIT
CUTS = tuple(range(56, 150, 6))
RES = []


def replay_with(pol, history, start, initial, q, mu, mode="exact"):
    """Same flow as the engine, but with an explicit per-series target vector q."""
    P = pol._PARAMS
    sa, sb = P["samp_a"], P["samp_b"]
    n = history.shape[0]
    demand = history[:, start : start + 8].T
    state = np.asarray(initial, dtype=float).copy()
    revealed, weekly = [], []
    for w in range(8):
        order = np.zeros(n)
        if w < 6:
            on_hand, arr1, arr2 = state.T
            X = on_hand + arr1
            qq = q.copy()
            if P["cfg"].get("adapt_k", 0) > 0 and w > 0:
                r = np.asarray(revealed, dtype=float).reshape(w, n)
                k = float(P["cfg"]["adapt_k"])
                mult = (r.sum(0) + k * mu * w) / (np.maximum(mu, 1e-9) * (w + k))
                qq = qq * np.clip(mult, 0.25, 4.0)
            left = pol._leftover_mean(X, arr2, sa, sb, mode)
            order = np.rint(np.maximum(0.0, np.nan_to_num(qq - left)))
        on_hand, arr1, arr2 = state.T
        supply = on_hand + arr1
        end = np.maximum(0.0, supply - demand[w])
        missed = np.maximum(0.0, demand[w] - supply)
        weekly.append(0.2 * end.sum() + 1.0 * missed.sum())
        state = np.column_stack((end, arr2, order))
        revealed.append(demand[w])
    return float(sum(weekly[2:]))


def evaluate(name, qfun, cuts=CUTS, adapt_k=0, mode="exact", cfg=None):
    pol = importlib.import_module("policy")
    importlib.reload(pol)
    c0 = dict(cfg or {})
    c0["adapt_k"] = adapt_k
    scores = {}
    for c in cuts:
        P = pol.fit(cutoff=c, cfg=c0)
        q = qfun(P)
        scores[c] = replay_with(pol, S, c, INIT, q, P["mu"], mode)
    mean = float(np.mean(list(scores.values())))
    RES.append((name, mean, scores))
    print(
        f"{name:34s} mean {mean:8.1f}  "
        + " ".join(f"{k}:{scores[k]:.0f}" for k in cuts),
        flush=True,
    )
    return scores


def ref_scores(cuts=CUTS):
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
    R = ref_scores()
    print(
        "REF                                mean %8.1f  " % np.mean(list(R.values()))
        + " ".join(f"{k}:{R[k]:.0f}" for k in CUTS),
        flush=True,
    )
    taus = np.array([0.30, 0.40, 0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95])
    for t in taus:
        evaluate(
            f"tau={t:.2f} global",
            lambda P, t=t: P["qmat"][:, int(np.argmin(np.abs(P["taus"] - t)))],
        )
    for t in (0.5, 0.6, 0.7):
        evaluate(
            f"tau={t:.2f} adapt_k=3",
            lambda P, t=t: P["qmat"][:, int(np.argmin(np.abs(P["taus"] - t)))],
            adapt_k=3,
        )
