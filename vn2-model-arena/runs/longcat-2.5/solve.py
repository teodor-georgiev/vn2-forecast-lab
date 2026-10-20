import numpy as np, pandas as pd, sys, time

BASE = r"C:\Users\Toni\Documents\Default Project\vn2-model-arena"
DATA = BASE + r"\data"
TOOLS = BASE + r"\tools"
OUT = BASE + r"\runs\longcat-2.5"
sys.path.insert(0, TOOLS)
import inventory_sim as sim

HOLD = 0.20
SHORT = 1.00


def load():
    sales = pd.read_csv(DATA + r"\sales.csv", parse_dates=["week"])
    init = pd.read_csv(DATA + r"\initial_state.csv")
    m = sales.copy()
    m["t"] = (m["week"] - m["week"].min()).dt.days // 7
    series = sorted(m.series_id.unique())
    n = len(series)
    T = int(m.t.max()) + 1
    idx = {s: i for i, s in enumerate(series)}
    Y = np.zeros((T, n))
    for r in m.itertuples():
        Y[int(r.t), idx[r.series_id]] = r.y
    woy_hist = m.groupby("t")["week"].first().dt.isocalendar().week.values.astype(int)
    ih = init.set_index("series_id").loc[series]
    state = ih[["on_hand", "arrival_w1", "arrival_w2"]].values.astype(float)
    game_woy = np.array(
        [(m.week.max() + pd.Timedelta(weeks=k)).isocalendar().week for k in range(1, 9)]
    )
    return Y, series, state, woy_hist, game_woy


def forecast(Y, H, woy_hist, game_woy, W=26, trend=False, lam=0.7):
    """L: (n, 8) mean demand forecast for game weeks 1..8 from history Y[:H]."""
    seg = Y[:H]
    base = seg[-W:].mean(axis=0)
    if trend:
        w2 = min(26, H // 2)
        prior = seg[-2 * w2 : -w2].mean(axis=0)
        tf = np.where(prior > 1e-9, base / np.maximum(prior, 1e-9), 1.0)
        base = base * np.clip(tf, 0.75, 1.33)
    wback = min(104, H - 8)
    sub = seg[-wback:]
    overall = sub.mean()
    factors = {}
    for w in game_woy:
        mask = woy_hist[H - wback : H] == w
        factors[w] = (
            (sub[mask].mean() / overall)
            if (mask.sum() >= 1 and overall > 1e-9)
            else 1.0
        )
    L = np.zeros((Y.shape[1], 8))
    for j, w in enumerate(game_woy):
        L[:, j] = base * (1.0 + lam * (factors[w] - 1.0))
    return np.maximum(L, 0.0)


def build_scenarios(Y, H, L, S, rng, pool_w=40):
    """D: (n, S, 8) demand scenarios = L * iid empirical residual ratio."""
    pw = min(pool_w, H - 8)
    seg = Y[H - pw : H]  # (pw, n)
    lvl = np.maximum(seg.mean(axis=0), 0.05 * L[:, 0])
    pool = np.maximum(seg / lvl, 0.0).T  # (n, pw) residual ratios
    idx = rng.integers(0, pw, size=(N_, S, 8))
    r = pool[np.arange(N_)[:, None, None], idx]  # (n, S, 8)
    return np.maximum(L[:, None, :] * r, 0.0)


N_ = 599


def batch_cost(O, D, state):
    """O: (n, k, 6); D: (n, S, 8); state: (n, 3) -> (n, k) mean cost wk3-8."""
    n, k, _ = O.shape
    S = D.shape[1]
    oh = np.broadcast_to(state[:, 0, None, None], (n, k, S)).copy()
    a1 = np.broadcast_to(state[:, 1, None, None], (n, k, S)).copy()
    a2 = np.broadcast_to(state[:, 2, None, None], (n, k, S)).copy()
    tot = np.zeros((n, k))
    for w in range(8):
        order = O[:, :, w][:, :, None] if w < 6 else np.zeros((n, k, 1))
        supply = oh + a1
        dw = D[:, :, w][:, None, :]
        end = np.maximum(0.0, supply - dw)
        missed = np.maximum(0.0, dw - supply)
        if w >= 2:
            tot += missed.sum(axis=2) + HOLD * end.sum(axis=2)
        oh = end
        a1 = a2
        a2 = order
    return tot / S


def optimize(D, state, passes=3):
    n, S, _ = D.shape
    mean_d = D.mean(axis=1)  # (n, 8)
    # init: just-in-time on mean path
    o = np.zeros((n, 6))
    oh = state[:, 0].copy()
    a1 = state[:, 1].copy()
    a2 = state[:, 2].copy()
    for w in range(8):
        arr = o[:, w - 2] if w >= 2 else np.zeros(n)
        supply0 = oh + a1
        if 2 <= w <= 7:
            o[:, w - 2] = np.maximum(0.0, mean_d[:, w] - supply0)
            arr = o[:, w - 2]
        supply = supply0 + arr
        end = np.maximum(0.0, supply - mean_d[:, w])
        order_this = o[:, w] if w < 6 else np.zeros(n)
        oh, a1, a2 = end, a2, order_this
    best = batch_cost(o[:, None, :], D, state)[:, 0]
    for p in range(passes):
        improved = False
        for c in range(6):
            rem = D[:, :, c + 2 : 8].sum(axis=2)  # (n,S)
            cap = np.quantile(rem, 0.995, axis=1) * 1.1 + 3.0
            cur = o[:, c].copy()
            offs = np.array([-8, -4, -2, -1, -0.5, 0.5, 1, 2, 4, 8], dtype=float)
            cand_sets = []
            for i in range(n):
                cs = {0.0, round(float(cur[i]), 3)}
                for d in offs:
                    v = cur[i] + d
                    if 0 <= v <= cap[i]:
                        cs.add(round(float(v), 3))
                for q in (0.5, 0.7, 0.85, 0.95):
                    cs.add(
                        round(
                            float(np.quantile(rem[i, :, None] if False else rem[i], q)),
                            3,
                        )
                    )
                cand_sets.append(sorted(cs))
            kmax = max(len(c) for c in cand_sets)
            O = np.repeat(o[:, None, :], kmax, axis=1)
            for i in range(n):
                for j, v in enumerate(cand_sets[i]):
                    O[i, j, c] = v
            C = batch_cost(O, D, state)
            for i in range(n):
                j = int(np.argmin(C[i]))
                if C[i, j] < best[i] - 1e-9:
                    best[i] = C[i, j]
                    o[i, c] = cand_sets[i][j]
                    improved = True
        if not improved:
            break
    # integer polish
    oi = np.rint(o).clip(0, None)
    for c in range(6):
        Oi = np.repeat(oi[:, None, :], 3, axis=1)
        Oi[:, 0, c] = np.clip(oi[:, c] - 1, 0, None)
        Oi[:, 1, c] = oi[:, c]
        Oi[:, 2, c] = oi[:, c] + 1
        C = batch_cost(Oi, D, state)
        for i in range(n):
            j = int(np.argmin(C[i]))
            oi[i, c] = Oi[i, j, c]
    return oi, best


def reference_adaptive_score(Y_true, L, state):
    def ffn(w, st, hist):
        return [L[:, min(w + h, 7)] for h in range(3)]

    res = sim.order_plan_from_policy(Y_true, state, ffn)
    return res["score"], res["setup"]


def main():
    t0 = time.time()
    Y, series, state, woy_hist, game_woy = load()
    n = Y.shape[1]
    globals()["N_"] = n
    print(f"loaded T={Y.shape[0]} n={n} game_woy={game_woy.tolist()}", flush=True)
    mode = sys.argv[1] if len(sys.argv) > 1 else "backtest"

    if mode == "backtest":
        rng = np.random.default_rng(7)
        Hs = [Y.shape[0] - 8 - 26, Y.shape[0] - 8 - 52]
        configs = [
            ("W26 lam0.0", dict(W=26, trend=False, lam=0.0)),
            ("W26 lam0.15", dict(W=26, trend=False, lam=0.15)),
            ("W20 lam0.3", dict(W=20, trend=False, lam=0.3)),
            ("W39 lam0.3", dict(W=39, trend=False, lam=0.3)),
            ("W26 lam0.3 pw52", dict(W=26, trend=False, lam=0.3, pool_w=52)),
        ]
        S = 48
        for name, cfg in configs:
            pw = cfg.pop("pool_w", 40)
            scores = []
            for H in Hs:
                L = forecast(Y, H, woy_hist, game_woy, **cfg)
                D = build_scenarios(Y, H, L, S, rng, pool_w=pw)
                if name == "W26 lam0.0":
                    o2, exp2 = optimize(D, np.zeros((n, 3)), passes=5)
                    print(
                        f"  exp_cost H={H}: plan={expcost_sum if False else exp2.sum():.0f}",
                        flush=True,
                    )
                st0 = np.zeros((n, 3))
                o, expc = optimize(D, st0)
                if name == "W26 lam0.0":
                    print(
                        f"  H={H}: expected_cost(passes3)={expc.sum():.0f} "
                        f"expected_cost(passes5)={exp2.sum():.0f}",
                        flush=True,
                    )
                true = Y[H : H + 8]
                r = sim.simulate(true, st0, o.T)
                z = sim.simulate(true, st0, np.zeros((6, n)))
                scores.append((r["score"], z["score"], r["setup"]))
            tot = sum(s[0] for s in scores) / len(scores)
            totz = sum(s[1] for s in scores) / len(scores)
            print(
                f"{name}: my={tot:.1f} zero={totz:.1f} "
                f"detail={[(round(s[0], 1), round(s[1], 1), round(s[2], 1)) for s in scores]} "
                f"t={time.time() - t0:.0f}s",
                flush=True,
            )
        for name, cfg in configs[:2]:
            sc = []
            for H in Hs:
                L = forecast(Y, H, woy_hist, game_woy, **cfg)
                st0 = np.zeros((n, 3))
                s, _ = reference_adaptive_score(Y[H : H + 8], L, st0)
                sc.append(s)
            print(f"ref {name}: {sum(sc) / len(sc):.1f}", flush=True)
        print(f"elapsed {time.time() - t0:.0f}s", flush=True)

    elif mode == "final":
        cfg = dict(W=26, trend=False, lam=0.0)
        rng = np.random.default_rng(11)
        S = 128
        L = forecast(Y, Y.shape[0], woy_hist, game_woy, **cfg)
        D = build_scenarios(Y, Y.shape[0], L, S, rng)
        t1 = time.time()
        o, expcost = optimize(D, state)
        print(
            f"optimized in {time.time() - t1:.0f}s, expected cost={expcost.sum():.1f}",
            flush=True,
        )
        rows = [
            (s, w + 1, int(o[i, w])) for i, s in enumerate(series) for w in range(6)
        ]
        df = pd.DataFrame(rows, columns=["series_id", "week", "quantity"])
        df.to_csv(OUT + r"\orders.csv", index=False)
        print(
            f"wrote orders.csv rows={len(df)} total_qty={int(df.quantity.sum())} "
            f"exp_demand_wk38={D[:, :, 2:8].sum() / S:.0f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
