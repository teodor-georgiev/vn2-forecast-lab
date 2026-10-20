"""Grid 3: forecast design knobs (level window, share window, family pooling mode,
seasonal weight) x newsvendor fractile.  Self-contained, vectorised."""

import sys, time, itertools
import numpy as np

HERE = r"C:\Users\Toni\Documents\Default Project\vn2-model-arena\runs3\w18"
sys.path.insert(0, HERE)
import policy

sv = np.load(HERE + r"\sv.npy")
av = np.load(HERE + r"\av.npy")
history = sv
CUTOFFS = (100, 115, 130, 145)
NC = len(CUTOFFS)

raw = {}
for c in CUTOFFS:
    D = policy._correct(sv, av, c)
    fam = np.load(HERE + r"\fam.npy")
    raw[c] = (D, fam)


def family_agg(D, fam, n_fam):
    DF = np.zeros((n_fam, D.shape[1]))
    np.add.at(DF, fam, D)
    return DF


def shapes(D, DF, SEAS_K=8, SHRINK=4.0):
    n, H = D.shape
    n_fam = DF.shape[0]
    lih = policy._trailing_means(D, SEAS_K)
    lfh = policy._trailing_means(DF, SEAS_K)
    sh_i = D / np.maximum(lih, 1e-6)
    sh_f = DF / np.maximum(lfh, 1e-6)
    woy = np.arange(H) % 52
    idx_f = np.ones((n_fam, 52))
    idx_i = np.ones((n, 52))
    for w in range(52):
        m = woy == w
        if m.sum() == 0:
            continue
        idx_f[:, w] = sh_f[:, m].mean(axis=1)
        idx_i[:, w] = sh_i[:, m].mean(axis=1)
    idx_f = np.clip(1.0 + (idx_f - 1.0) * (n_fam / (n_fam + SHRINK)), 0.15, 3.0)
    vol = D.sum(axis=1)
    wgt = (vol / (vol + SHRINK * max(vol.mean(), 1e-6)))[:, None]
    idx_i = np.clip(wgt * idx_i + (1 - wgt) * idx_f[fam], 0.15, 3.0)
    return idx_i, idx_f


def tmean(x, K):
    k = min(K, x.shape[1])
    v = x[:, -k:]
    w = np.exp(-np.arange(k - 1, -1, -1) / (k / 2.0))
    return (v * w).sum(axis=1) / w.sum()


def forecast(
    c, K=13, W=26, seas_w=0.0, fam_mode="fix", fam_w=0.25, SHRINK=4.0, SEAS_K=8
):
    D, fam = raw[c]
    n, H = D.shape
    n_fam = int(fam.max()) + 1
    DF = family_agg(D, fam, n_fam)
    lvl_i = tmean(D, K)
    lvl_f = tmean(DF, K)
    idx_i, idx_f = shapes(D, DF, SEAS_K, SHRINK)
    sum_i = D[:, -W:].sum(axis=1)
    sum_f = DF[:, -W:].sum(axis=1)
    cnt = np.bincount(fam, minlength=n_fam).astype(float)
    prior_f = 0.15 * sum_f / np.maximum(cnt, 1)
    share = (sum_i + prior_f[fam]) / (sum_f[fam] + prior_f[fam])
    F = np.zeros((8, n))
    for g in range(8):
        w = (H + g) % 52
        own = (1 - seas_w) * lvl_i + seas_w * (lvl_i * idx_i[:, w])
        alloc = (lvl_f * idx_f[:, w])[fam] * share
        if fam_mode == "fix":
            wgt = np.full(n, fam_w)
        elif fam_mode == "vol":
            vol = D.sum(axis=1)
            wgt = vol / (vol + fam_w * max(vol.mean(), 1e-6))
        else:  # 'volinv': low-volume series lean on the family
            vol = D.sum(axis=1)
            wgt = 1.0 - (vol / (vol + fam_w * max(vol.mean(), 1e-6)))
        F[g] = (1 - wgt) * own + wgt * alloc
    return np.maximum(F, 0.0)


def run_game(F, dem, r, fractile=0.7, paths=256):
    n = dem.shape[1]
    state = np.zeros((n, 3))
    revealed = []
    tot = 0.0
    for w in range(8):
        if w < 6:
            Fw = policy._adapt(F, w, revealed)
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


if __name__ == "__main__":
    t0 = time.perf_counter()
    rs = {c: policy._dispersion(raw[c][0]) for c in CUTOFFS}
    dems = {c: history[:, c : c + 8].T for c in CUTOFFS}

    def score(**kw):
        fr = kw.pop("fractile", 0.7)
        sc = []
        for c in CUTOFFS:
            F = forecast(c, **kw)
            sc.append(run_game(F, dems[c], rs[c], fractile=fr))
        return float(np.mean(sc)), sc

    print(
        "== stage 1: level window K x share window W (seas=0, fam=fix 0.25, fr=0.70) =="
    )
    best = None
    for K in (8, 13, 26):
        for W in (13, 26, 52):
            m, sc = score(K=K, W=W)
            print(
                "  K=%2d W=%2d  mean %8.1f  %s"
                % (K, W, m, " ".join("%6.0f" % x for x in sc))
            )
            if best is None or m < best[0]:
                best = (m, K, W)
    print("  BEST K=%d W=%d (%.1f)" % (best[1], best[2], best[0]))
    K, W = best[1], best[2]

    print("\n== stage 2: family pooling mode x seas_w x fractile ==")
    rows = []
    for fam_mode, fam_w, seas_w in [
        ("fix", 0.15, 0.0),
        ("fix", 0.25, 0.0),
        ("fix", 0.40, 0.0),
        ("fix", 0.25, 0.4),
        ("fix", 0.25, 0.8),
        ("vol", 2.0, 0.0),
        ("vol", 8.0, 0.0),
        ("vol", 8.0, 0.4),
        ("volinv", 2.0, 0.0),
        ("volinv", 8.0, 0.0),
    ]:
        for fr in (0.65, 0.70, 0.75, 0.80):
            m, sc = score(
                K=K, W=W, fam_mode=fam_mode, fam_w=fam_w, seas_w=seas_w, fractile=fr
            )
            rows.append((m, fam_mode, fam_w, seas_w, fr, sc))
            print(
                "  %-6s w=%.2f seas=%.2f fr=%.2f  mean %8.1f  %s"
                % (fam_mode, fam_w, seas_w, fr, m, " ".join("%6.0f" % x for x in sc))
            )
    rows.sort()
    print("\n  BEST: %.1f  mode=%s fam_w=%.2f seas=%.2f fr=%.2f" % rows[0][:5])
    print("total %.1fs" % (time.perf_counter() - t0))
