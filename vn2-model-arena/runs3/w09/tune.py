import sys, time, numpy as np, pandas as pd

sys.path.insert(0, r"C:\Users\Toni\Documents\Default Project\vn2-model-arena\runs3\w09")
# reuse exp functions
import exp

# Override correct method easily by calling exp.correct
ids = exp.ids if hasattr(exp, "ids") else None
# exp has S,A,INIT,SH,correct,ewma_forecast,make_decide,adaptive_backtest


def evaluate(
    corr_method,
    spans,
    use_seasonal,
    seasonal_w,
    projection,
    buffer,
    scale,
    alpha_resid=None,
    qadd=None,
):
    decide = exp.make_decide(
        corr_method, spans, use_seasonal, seasonal_w, None, projection, buffer, scale
    )
    # inject residual quantile addition via monkey? We'll handle by wrapping
    if qadd is not None:
        base_decide = decide

        def decide2(week_index, state, revealed):
            o = base_decide(week_index, state, revealed)
            # add qadd[week? horizon?] - need per horizon not available; skip for now
            return o

        decide = decide2
    scores = []
    for st in [100, 130]:
        sc, _, _ = exp.adaptive_backtest(decide, exp.S, st, exp.INIT)
        scores.append(sc)
    return scores


# Grid over corr methods and seasonal weights
for cm in ["none", "mean_all", "seasonal_mean", "highq"]:
    for sw in [0.0, 0.3, 0.5, 0.7]:
        sc = evaluate(cm, (4, 8, 13, 26, 52), True, sw, 1.0, 0.75, 1.0)
        line = f"cm={cm:15s} sw={sw:.1f} scores={[round(x) for x in sc]} avg={np.mean(sc):.0f}"
        print(line)
        with open("tune_results.txt", "a", encoding="ascii") as f:
            f.write(line + "\n")
print("done")
