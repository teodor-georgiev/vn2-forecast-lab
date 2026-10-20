"""Shared data helpers + feature builder (scratch, not the deliverable)."""

import numpy as np, pandas as pd, os

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "..", "data")


def load_raw():
    s = pd.read_csv(os.path.join(DATA, "sales.csv"))
    a = pd.read_csv(os.path.join(DATA, "availability.csv"))
    s["week"] = pd.to_datetime(s["week"])
    a["week"] = pd.to_datetime(a["week"])
    weeks = np.sort(s["week"].unique())
    sids = np.sort(s["series_id"].unique())
    Y = (
        s.pivot(index="series_id", columns="week", values="y")
        .reindex(index=sids, columns=weeks)
        .to_numpy(float)
    )
    A = (
        a.pivot(index="series_id", columns="week", values="is_available")
        .reindex(index=sids, columns=weeks)
        .to_numpy()
    )
    A = np.nan_to_num(A.astype(float))
    init = (
        pd.read_csv(os.path.join(DATA, "initial_state.csv"))
        .set_index("series_id")
        .reindex(sids)[["on_hand", "arrival_w1", "arrival_w2"]]
        .to_numpy(float)
    )
    return sids, Y, A, init


def impute(Y, A):
    """Censoring-corrected demand estimate: available weeks = observed, sold-out weeks = series available-week mean."""
    D = Y.copy()
    for i in range(D.shape[0]):
        av = A[i] > 0.5
        if av.sum() == 0:
            continue
        m = D[i, av].mean()
        D[i, ~av] = np.maximum(D[i, ~av], m)
    return D


LAGS = [1, 2, 3, 4]
ROLLS = [4, 8, 13, 26]


def build_features(D, A, origins):
    """D,A: (n,T). origins: 1-D int array of last-known column indices. Returns X (len(origins), n, F)."""
    n, T = D.shape
    cs = np.concatenate([np.zeros((n, 1)), np.cumsum(D, axis=1)], axis=1)
    cz = np.concatenate([np.zeros((n, 1)), np.cumsum(D == 0, axis=1)], axis=1)
    csq = np.concatenate([np.zeros((n, 1)), np.cumsum(D * D, axis=1)], axis=1)
    ca = np.concatenate([np.zeros((n, 1)), np.cumsum(A, axis=1)], axis=1)
    feats = []
    for t in origins:
        f = []
        for L in LAGS:
            f.append(D[:, t - L + 1])
        for k in ROLLS:
            lo = max(0, t + 1 - k)
            m = (cs[:, t + 1] - cs[:, lo]) / (t + 1 - lo)
            f.append(m)
        # zero fraction last 13
        lo = max(0, t + 1 - 13)
        f.append((cz[:, t + 1] - cz[:, lo]) / (t + 1 - lo))
        # cv last 13
        lo = max(0, t + 1 - 13)
        kk = t + 1 - lo
        m = (cs[:, t + 1] - cs[:, lo]) / kk
        v = (csq[:, t + 1] - csq[:, lo]) / kk - m * m
        f.append(np.sqrt(np.maximum(v, 0)) / (m + 1e-6))
        # availability last 4
        lo = max(0, t + 1 - 4)
        f.append((ca[:, t + 1] - ca[:, lo]) / (t + 1 - lo))
        # max last 8
        lo = max(0, t + 1 - 8)
        f.append(D[:, lo : t + 1].max(axis=1))
        # expanding series mean / zero frac
        f.append(cs[:, t + 1] / (t + 1))
        f.append(cz[:, t + 1] / (t + 1))
        # calendar
        woy = 2 * np.pi * ((t + 1) % 52) / 52.0
        f.append(np.full(n, np.sin(woy)))
        f.append(np.full(n, np.cos(woy)))
        feats.append(np.stack(f, axis=1))
    return np.stack(feats, axis=0)  # (len(origins), n, F)


FEATNAMES = (
    [f"lag{L}" for L in LAGS]
    + [f"roll{k}" for k in ROLLS]
    + [
        "zerofrac13",
        "cv13",
        "avail4",
        "max8",
        "smean",
        "szerofrac",
        "woy_sin",
        "woy_cos",
    ]
)
