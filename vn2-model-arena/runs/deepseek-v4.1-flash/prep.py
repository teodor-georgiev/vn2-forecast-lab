import numpy as np, pandas as pd

D = np.load("D.npy")
A = np.load("A.npy")
n, T = D.shape


def impute(D, A):
    """Fill unavailable weeks with seasonal / local estimates of latent demand."""
    X = D.copy()
    for t in np.where(~A.any(axis=0))[0]:
        pass
    # local median of available values in +-3 week window (same series)
    filled = X.copy()
    for t in range(T):
        bad = ~A[:, t]
        if not bad.any():
            continue
        lo, hi = max(0, t - 3), min(T, t + 4)
        w = A[:, lo:hi]
        vals = np.where(w, X[:, lo:hi], np.nan)
        med = np.nanmedian(vals, axis=1)
        # fall back to same week other years
        alt = []
        for k in (-104, -52, 52, 104):
            tt = t + k
            if 0 <= tt < T:
                alt.append(np.where(A[:, tt], X[:, tt], np.nan))
        if alt:
            alt = np.nanmedian(np.vstack(alt), axis=0) if len(alt) > 1 else alt[0]
            med = np.where(np.isnan(med), alt, med)
        med = np.where(np.isnan(med), np.nanmean(np.where(A, X, np.nan), axis=1), med)
        filled[bad, t] = np.maximum(0, med[bad])
    return filled


L = impute(D, A)
np.save("L.npy", L)


def forecast(L, A, cut, h=8):
    """forecasts for weeks cut+1 .. cut+h (0-based cut = last observed)."""
    hist = L[:, : cut + 1]
    av = A[:, : cut + 1]
    out = np.zeros((h, n))
    # recent level: mean of last 8 weeks, weight recent
    lvl8 = hist[:, -8:].mean(axis=1)
    lvl4 = hist[:, -4:].mean(axis=1)
    lvl13 = hist[:, -13:].mean(axis=1)
    # seasonal index from same calendar week in previous years, normalized by local mean
    for j in range(h):
        t = cut + 1 + j
        cand = []
        for lag in (52, 104):
            tt = t - lag
            if 0 <= tt <= cut and av[:, tt].mean() > 0.5:
                # ratio of that week to the 13-week mean around it
                lo, hi = max(0, tt - 6), min(cut + 1, tt + 7)
                loc = hist[:, lo:hi].mean(axis=1)
                cand.append(np.where(loc > 0, hist[:, tt] / np.maximum(loc, 1e-9), 1.0))
        if cand:
            sf = np.clip(np.mean(cand, axis=0), 0.3, 3.0)
        else:
            sf = np.ones(n)
        base = 0.5 * lvl8 + 0.3 * lvl4 + 0.2 * lvl13
        out[j] = base * sf
    # damp: shrink toward series mean for noisy small series
    mu = hist.mean(axis=1)
    out = 0.85 * out + 0.15 * mu[None, :]
    return np.maximum(out, 0)


if __name__ == "__main__":
    for cut in (85, 98, 111, 124, 137, 144):
        f = forecast(L, A, cut)
        y = L[:, cut + 1 : cut + 9].T
        mae = np.abs(f - y).mean()
        # scale-weighted
        wmae = np.abs(f - y).sum() / max(y.sum(), 1)
        print(cut, "MAE", round(mae, 3), "wMAE", round(wmae, 3))
