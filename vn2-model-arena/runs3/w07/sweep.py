import numpy as np, sys
import dev
from pathlib import Path
import inventory_sim as sim

Y, AV, ST0 = dev.Y, dev.AV, dev.ST0
N = Y.shape[0]
CUTS = [60, 100, 130]
rng = np.random.default_rng(7)


def nb_paths(m, v, h, S=3000):
    m = np.maximum(m, 1e-6)
    v = np.maximum(v, 1.05 * m)
    r = np.clip(m * m / (v - m), 0.05, 1e6)
    lam = rng.gamma(
        shape=r[None, :, None], scale=(m / r)[None, :, None], size=(S, N, h)
    )
    return rng.poisson(lam).astype(float)


def make_policy(m, v, H, tau, adapt=0.0):
    """order up to tau-quantile of demand over the next H weeks minus pipeline."""
    paths = nb_paths(m, v, H, S=4000)
    target = np.quantile(paths.sum(axis=2), tau, axis=0)

    def decide(w, state, revealed):
        if w >= 6:
            return np.zeros(state.shape[0])
        mm = m
        if adapt > 0 and w > 0 and np.asarray(revealed).size:
            mr = np.asarray(revealed, dtype=float).mean(axis=0)
            mm = (adapt * m + w * mr) / (adapt + w)
        sc = np.where(m > 0, mm / m, 1.0)
        t = np.quantile(paths.sum(axis=2), tau, axis=0) * 1.0  # base target
        t = target * np.clip(sc, 0.0, None)
        ip = state[:, 0] + state[:, 1] + state[:, 2]
        return np.maximum(0.0, t - ip)

    return decide


def score(cut, dec):
    m, v = dev.policy.build_prior(Y, AV, cols=slice(0, cut))
    res = dev.hb.backtest(dec, Y, start_week=cut, initial=ST0)
    return res["score"]


if __name__ == "__main__":
    print("H tau  adapt   " + "  ".join(f"c{c}" for c in CUTS) + "   mean")
    best = None
    for H in [1, 2, 3, 4]:
        for tau in [0.50, 0.65, 0.75, 0.83, 0.90, 0.95]:
            for adapt in [0.0, 8.0]:
                ss = []
                for c in CUTS:
                    m, v = dev.policy.build_prior(Y, AV, cols=slice(0, c))
                    ss.append(score(c, make_policy(m, v, H, tau, adapt)))
                mean = float(np.mean(ss))
                print(
                    f"{H} {tau:.2f} {adapt:5.0f}   "
                    + "  ".join(f"{s:5.0f}" for s in ss)
                    + f"  {mean:6.0f}"
                )
                if best is None or mean < best[0]:
                    best = (mean, H, tau, adapt)
    print("BEST", best)
