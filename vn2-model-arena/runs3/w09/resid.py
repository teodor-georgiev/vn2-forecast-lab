import sys, numpy as np, pandas as pd

sys.path.insert(0, r"C:\Users\Toni\Documents\Default Project\vn2-model-arena\runs3\w09")
import exp

# Use corrected history
cm = "seasonal_mean"
SH = exp.correct(exp.S, exp.A, cm)
spans = (4, 13, 52)
sw = 0.2


def ewma_level(hist):
    # hist (n,T)
    levels = []
    for sp in spans:
        alpha = 2.0 / (sp + 1)
        level = hist[:, 0].copy()
        for t in range(1, hist.shape[1]):
            level = alpha * hist[:, t] + (1 - alpha) * level
        levels.append(level)
    L = np.stack(levels, 0)
    return np.median(L, 0)


def forecast_at(hist):
    # hist (n,T) corrected history up to cutoff (inclusive? we'll pass history ending at cutoff)
    base = ewma_level(hist)
    T = hist.shape[1]
    fc = np.zeros((3, hist.shape[0]))
    for hi in range(3):
        h = hi + 1
        idx = T - 52 + (h - 1)
        if 0 <= idx < T:
            seas = hist[:, idx]
        else:
            seas = base
        fc[hi] = (1 - sw) * base + sw * seas
    return np.maximum(fc, 0)


# Rolling residuals
# cutoffs: last index t (0-based) such that t+3 < T
residuals = {1: [], 2: [], 3: []}  # each list of (n,) arrays
cutoffs = list(range(80, 150, 4))
for t in cutoffs:
    hist = SH[:, : t + 1]  # up to week t
    fc = forecast_at(hist)  # (3,n) forecast for weeks t+1,t+2,t+3
    for hi in range(3):
        actual = SH[:, t + 1 + hi]
        res = actual - fc[hi]
        residuals[hi + 1].append(res)
for h in [1, 2, 3]:
    R = np.stack(residuals[h], 0)  # (K,n)
    print(f"horizon {h}: residual mean {R.mean():.3f} std {R.std():.3f}")
# quantile per series per horizon
alpha = 0.8333
Q = {}
for h in [1, 2, 3]:
    R = np.stack(residuals[h], 0)
    Q[h] = np.quantile(R, alpha, axis=0)
print("q3 median", np.median(Q[3]), "min", Q[3].min(), "max", Q[3].max())
np.save("q_resid_83.npy", np.stack([Q[1], Q[2], Q[3]], 0))
# also try alpha 0.9
for alpha in [0.8, 0.9]:
    Q3 = np.quantile(np.stack(residuals[3], 0), alpha, axis=0)
    print(f"alpha={alpha} q3 median {np.median(Q3):.3f}")
print("done")
