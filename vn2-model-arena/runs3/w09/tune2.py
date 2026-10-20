import sys, numpy as np

sys.path.insert(0, r"C:\Users\Toni\Documents\Default Project\vn2-model-arena\runs3\w09")
import exp


def evaluate(spans, use_seasonal, sw, projection, buffer, scale, cm="seasonal_mean"):
    decide = exp.make_decide(
        cm, spans, use_seasonal, sw, None, projection, buffer, scale
    )
    scores = []
    for st in [100, 130]:
        sc, _, _ = exp.adaptive_backtest(decide, exp.S, st, exp.INIT)
        scores.append(sc)
    return scores


# Grid
best = None
results = []
for spans in [(4, 8, 13), (8, 13, 26), (4, 8, 13, 26), (13,), (4, 13, 52), (8, 26)]:
    for sw in [0.0, 0.2, 0.3]:
        for proj in [0.8, 1.0, 1.2]:
            for buf in [0.5, 0.75, 1.0]:
                for scale in [1.0]:
                    sc = evaluate(spans, True, sw, proj, buf, scale)
                    avg = np.mean(sc)
                    results.append((avg, spans, sw, proj, buf, scale, sc))
                    print(
                        f"spans={spans} sw={sw} proj={proj} buf={buf} avg={avg:.0f} {[round(x) for x in sc]}"
                    )
results.sort()
best = results[0]
print("BEST", best)
with open("tune2_results.txt", "w") as f:
    for r in sorted(results)[:15]:
        f.write(str(r) + "\n")
print("done")
