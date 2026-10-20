import sys, numpy as np

sys.path.insert(0, r"C:\Users\Toni\Documents\Default Project\vn2-model-arena\runs3\w09")
import exp


def evaluate(spans, sw, projection, buffer, cm="seasonal_mean"):
    decide = exp.make_decide(cm, spans, True, sw, None, projection, buffer, 1.0)
    scores = []
    for st in [100, 130]:
        sc, _, _ = exp.adaptive_backtest(decide, exp.S, st, exp.INIT)
        scores.append(sc)
    return scores


best = None
cnt = 0
for spans in [(4, 13, 52), (8, 13, 26), (4, 8, 13)]:
    for sw in [0.0, 0.2]:
        for proj in [1.0, 1.1, 1.2]:
            for buf in [0.5, 0.75, 1.0]:
                sc = evaluate(spans, sw, proj, buf)
                avg = float(np.mean(sc))
                if best is None or avg < best[0]:
                    best = (avg, spans, sw, proj, buf, sc)
                cnt += 1
                print(
                    f"{cnt} spans={spans} sw={sw} proj={proj} buf={buf} avg={avg:.0f} {[round(x) for x in sc]}",
                    flush=True,
                )
                with open("tune3_results.txt", "a") as f:
                    f.write(
                        f"{avg:.1f} spans={spans} sw={sw} proj={proj} buf={buf} sc={[round(x) for x in sc]}\n"
                    )
print("BEST", best)
print("done")
