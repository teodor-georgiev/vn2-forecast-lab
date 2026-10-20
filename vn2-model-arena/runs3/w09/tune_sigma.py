import sys, numpy as np

sys.path.insert(0, r"C:\Users\Toni\Documents\Default Project\vn2-model-arena\runs3\w09")
import exp

SH = exp.correct(exp.S, exp.A, "seasonal_mean")
spans = (4, 13, 52)
sw = 0.2
proj = 1.2
# residual std per series for horizon3
residuals3 = []


def ewma_level(hist):
    levels = []
    for sp in spans:
        alpha = 2.0 / (sp + 1)
        level = hist[:, 0].copy()
        for t in range(1, hist.shape[1]):
            level = alpha * hist[:, t] + (1 - alpha) * level
        levels.append(level)
    return np.median(np.stack(levels, 0), 0)


def forecast_at(hist):
    base = ewma_level(hist)
    T = hist.shape[1]
    fc = np.zeros((3, hist.shape[0]))
    for hi in range(3):
        h = hi + 1
        idx = T - 52 + (h - 1)
        seas = hist[:, idx] if 0 <= idx < T else base
        fc[hi] = (1 - sw) * base + sw * seas
    return np.maximum(fc, 0)


for t in range(80, 150, 4):
    hist = SH[:, : t + 1]
    fc = forecast_at(hist)
    residuals3.append(SH[:, t + 3] - fc[2])
R3 = np.stack(residuals3, 0)
sigma = np.std(R3, axis=0)
# quantile of standard normal for 0.833 = 0.967
z = 0.967


def make_decide(mode):
    def decide(week_index, state, revealed):
        state = np.asarray(state, float)
        n = state.shape[0]
        revealed = np.asarray(revealed, float)
        rev = revealed.reshape(-1, n) if revealed.size else np.zeros((0, n))
        if n == exp.S.shape[0]:
            hist = SH.copy()
            if rev.shape[0] > 0:
                hist = np.column_stack([hist, rev.T])
        else:
            hist = rev.T if rev.shape[0] > 0 else np.zeros((n, 1))
        levels = []
        for sp in spans:
            a = 2.0 / (sp + 1)
            level = hist[:, 0].copy()
            for t in range(1, hist.shape[1]):
                level = a * hist[:, t] + (1 - a) * level
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
        if mode == "poisson":
            safety = 0.5 * np.sqrt(np.maximum(c, 0))
        elif mode == "normal":
            safety = (
                z * sigma if n == exp.S.shape[0] else 0.5 * np.sqrt(np.maximum(c, 0))
            )
        elif mode == "maxboth":
            s1 = 0.5 * np.sqrt(np.maximum(c, 0))
            s2 = z * sigma if n == exp.S.shape[0] else s1
            safety = np.maximum(s1, s2)
        return np.clip(c + safety - projected, 0, None)

    return decide


for mode in ["poisson", "normal", "maxboth"]:
    dec = make_decide(mode)
    scs = [exp.adaptive_backtest(dec, exp.S, st, exp.INIT)[0] for st in [100, 130]]
    print(f"{mode:8s} avg={np.mean(scs):.0f} {[round(x) for x in scs]}")
print("done")
