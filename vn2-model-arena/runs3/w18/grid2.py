"""Grid: forecast blend (SEAS_W, FAMILY_W) x newsvendor fractile, on several cutoffs."""

import sys, time, itertools
import numpy as np

HERE = r"C:\Users\Toni\Documents\Default Project\vn2-model-arena\runs3\w18"
sys.path.insert(0, HERE)
import policy

sv = np.load(HERE + r"\sv.npy")
history = sv
CUTOFFS = (100, 115, 130, 145)
CACHE = {}


def components(c):
    if c not in CACHE:
        CACHE[c] = policy._components(c)
    return CACHE[c]


def run_game(
    F, dem, r, w_state_zero=True, fractile=0.8, paths=256, use_adapt=True, init=None
):
    n = dem.shape[1]
    state = np.zeros((n, 3)) if init is None else init.copy()
    revealed = []
    tot = 0.0
    for w in range(8):
        if w < 6:
            Fw = policy._adapt(F, w, revealed) if use_adapt else F
            rng = np.random.default_rng(1234 + w)
            order = policy._orders(
                Fw, r, state, w, rng, fractile=fractile, n_paths=paths
            )
        else:
            order = np.zeros(n)
        on_hand, arr1, arr2 = state.T
        supply = on_hand + arr1
        end = np.maximum(0.0, supply - dem[w])
        missed = np.maximum(0.0, dem[w] - supply)
        if w >= 2:
            tot += 0.20 * end.sum() + 1.00 * missed.sum()
        state = np.column_stack((end, arr2, order))
        revealed.append(dem[w])
    return tot


def wape(f, d):
    return np.abs(f - d).sum() / max(d.sum(), 1e-9)


if __name__ == "__main__":
    t0 = time.perf_counter()
    comps = {c: components(c) for c in CUTOFFS}
    for c in CUTOFFS:
        comps[c]["r"] = policy._dispersion(comps[c]["D"])
    print("components built %.1fs" % (time.perf_counter() - t0))

    # --- stage A: fractile with a few forecast blends -------------------
    print("\n== stage A: fractile x blend ==")
    blends = [
        (0.0, 0.0),
        (0.40, 0.0),
        (0.40, 0.25),
        (0.70, 0.25),
        (0.0, 0.25),
        (0.40, 0.50),
    ]
    rows = []
    for seas_w, family_w in blends:
        for fr in (0.60, 0.70, 0.75, 0.80, 0.85, 0.90):
            sc = []
            for c in CUTOFFS:
                comp = comps[c]
                F = policy._assemble(comp, seas_w, family_w)
                dem = history[:, c : c + 8].T
                sc.append(run_game(F, dem, comp["r"], fractile=fr))
            m = float(np.mean(sc))
            rows.append((m, seas_w, family_w, fr, sc))
            print(
                "  seas=%.2f fam=%.2f fr=%.2f  mean %8.1f  %s"
                % (seas_w, family_w, fr, m, " ".join("%6.0f" % x for x in sc))
            )
    rows.sort()
    print("  BEST: mean %.1f seas=%.2f fam=%.2f fr=%.2f" % rows[0][:4])
    print("total %.1fs" % (time.perf_counter() - t0))
