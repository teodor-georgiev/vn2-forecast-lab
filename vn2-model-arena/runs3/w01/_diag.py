import sys, numpy as np

sys.path.insert(0, r"C:\Users\Toni\Documents\Default Project\vn2-model-arena\runs3\w01")
import exp, importlib

S, INIT = exp.S, exp.INIT


def detail(decide, history, start_week, initial, label):
    n = history.shape[0]
    demand = history[:, start_week : start_week + 8].T
    state = np.asarray(initial, dtype=float).copy()
    revealed, rows = [], []
    for w in range(8):
        order = np.zeros(n)
        if w < 6:
            rev = (
                np.array(revealed, dtype=float).reshape(w, n) if w else np.empty((0, n))
            )
            order = np.rint(
                np.maximum(
                    0.0, np.nan_to_num(np.asarray(decide(w, state.copy(), rev), float))
                )
            )
        on_hand, arr1, arr2 = state.T.copy()
        supply = on_hand + arr1
        end = np.maximum(0.0, supply - demand[w])
        missed = np.maximum(0.0, demand[w] - supply)
        rows.append(
            (
                w + 1,
                float(demand[w].sum()),
                int(order.sum()),
                float(end.sum()),
                float(missed.sum()),
            )
        )
        state = np.column_stack((end, arr2, order))
        revealed.append(demand[w])
    score = sum(0.2 * r[3] + 1.0 * r[4] for r in rows[2:])
    print(f"{label} c={start_week} score={score:8.1f}")
    print("  wk  demand   orders   endinv  missed   cost")
    for w, d, o, e, m in rows:
        print(f"  {w}  {d:7.0f}  {o:7.0f}  {e:7.0f}  {m:7.0f}  {0.2 * e + m:8.1f}")
    return score


for c in (100, 130):
    hist_mean = S[:, :c].mean()
    win_mean = S[:, c : c + 8].mean()
    print(
        f"== cutoff {c}: history mean/obs {hist_mean:.3f}, window mean {win_mean:.3f}, "
        f"window total {S[:, c : c + 8].sum():.0f}, init total {INIT.sum():.0f}"
    )
    pol = importlib.import_module("policy")
    importlib.reload(pol)
    pol.fit(cutoff=c, cfg={"tau_mode": "global", "tau_global": 0.85})
    detail(pol.decide, S, c, INIT, "  mine  ")
    sys.path.insert(0, r"C:\Users\Toni\Documents\Default Project\vn2-model-arena\tools")
    import inventory_sim as inv

    mu = S[:, :c].mean(1)
    detail(
        lambda w, s, r, mu=mu: inv.reference_orders(np.array([mu, mu, mu]), s),
        S,
        c,
        INIT,
        "  ref   ",
    )
