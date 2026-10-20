import sys, numpy as np

sys.path.insert(0, r"C:\Users\Toni\Documents\Default Project\vn2-model-arena\runs3\w09")
import exp

# load q
Q = np.load("q_resid_83.npy")  # (3,n)
SH = exp.correct(exp.S, exp.A, "seasonal_mean")
spans = (4, 13, 52)
sw = 0.2
proj = 1.2


def decide(week_index, state, revealed):
    state = np.asarray(state, float)
    n = state.shape[0]
    revealed = np.asarray(revealed, float)
    if revealed.size == 0:
        rev = np.zeros((0, n))
    else:
        rev = revealed.reshape(-1, n)
    if n == exp.S.shape[0]:
        hist = SH.copy()
        if rev.shape[0] > 0:
            hist = np.column_stack([hist, rev.T])
    else:
        hist = rev.T if rev.shape[0] > 0 else np.zeros((n, 1))
    # forecast
    # ewma
    levels = []
    for sp in spans:
        alpha = 2.0 / (sp + 1)
        level = hist[:, 0].copy()
        for t in range(1, hist.shape[1]):
            level = alpha * hist[:, t] + (1 - alpha) * level
        levels.append(level)
    base = np.median(np.stack(levels, 0), 0)
    T = hist.shape[1]
    fc = np.zeros((3, n))
    for hi in range(3):
        h = hi + 1
        idx = T - 52 + (h - 1)
        seas = hist[:, idx] if 0 <= idx < T else base
        fc[hi] = (1 - sw) * base + sw * seas
    fc = np.maximum(fc, 0)
    a, b, c = fc[0], fc[1], fc[2]
    on_hand, arr1, arr2 = state.T
    projected = np.maximum(0.0, on_hand + arr1 + arr2 - proj * (a + b))
    # residual quantile for horizon3
    q3 = Q[2] if n == exp.S.shape[0] else np.zeros(n)
    deficit = c + q3 - projected
    # also maybe use buffer? no
    order = np.clip(deficit, 0, None)
    return order


# quick compare with buffer version
decide_buf = exp.make_decide("seasonal_mean", spans, True, sw, None, proj, 0.5, 1.0)
for st in [100, 130]:
    sc_r, _, _ = exp.adaptive_backtest(decide, exp.S, st, exp.INIT)
    sc_b, _, _ = exp.adaptive_backtest(decide_buf, exp.S, st, exp.INIT)
    print(f"start {st}: resid={sc_r:.1f} buf={sc_b:.1f}")
print("done")
