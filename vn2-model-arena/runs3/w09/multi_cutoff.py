import sys, numpy as np

sys.path.insert(0, r"C:\Users\Toni\Documents\Default Project\vn2-model-arena\runs3\w09")
import exp

SH = exp.correct(exp.S, exp.A, "seasonal_mean")
spans = (4, 13, 52)
sw = 0.2
proj = 1.2


def make_param(buf):
    return exp.make_decide("seasonal_mean", spans, True, sw, None, proj, buf, 1.0)


# residual q3 alpha 0.75
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
q3 = np.quantile(R3, 0.75, axis=0)


def make_resid():
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
        add = q3 if n == exp.S.shape[0] else 0
        return np.clip(c + add - projected, 0, None)

    return decide


dec_p = make_param(0.5)
dec_r = make_resid()
print("cutoff  param   resid")
for st in [80, 90, 100, 110, 120, 130, 140]:
    sp = exp.adaptive_backtest(dec_p, exp.S, st, exp.INIT)[0]
    sr = exp.adaptive_backtest(dec_r, exp.S, st, exp.INIT)[0]
    print(f"{st:6d} {sp:8.1f} {sr:8.1f}")
print("done")
