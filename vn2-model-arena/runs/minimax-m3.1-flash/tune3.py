import numpy as np, pandas as pd
from pathlib import Path
from core import load, plan_news, sigma_from, plan_ref
import inventory_sim as sim
from tune2 import fc_anchor, proxy_state, WINDOWS, FORECASTS

H, series, I = load()
base = []
for T in WINDOWS:
    mu = H[max(0, T - 13) : T].mean(0)
    I0 = proxy_state(mu)
    base.append(sim.simulate(H[T : T + 8], I0, plan_ref(H, T, I0, mu))["score"])
base = np.array(base)
print("baseline total %.0f" % base.sum())
for anchor_w in [0.5]:
    for z in [0.4, 0.5, 0.6, 0.7, 0.9, 1.1]:
        for sw in [26, 52]:
            sc = []
            for T in WINDOWS:
                fc = fc_anchor(H, T, alpha=0.5, recent=4, anchor_w=anchor_w)
                sig = np.repeat(sigma_from(H, T, H.shape[1], sw)[None, :], 8, axis=0)
                I0 = proxy_state(H[max(0, T - 13) : T].mean(0))
                sc.append(
                    sim.simulate(H[T : T + 8], I0, plan_news(fc, sig, I0, z=z))["score"]
                )
            sc = np.array(sc)
            r = sc / base
            print(
                "aw=%.2f z=%.1f sw=%2d total=%.0f med=%.3f worst=%.3f trim=%.0f"
                % (
                    anchor_w,
                    z,
                    sw,
                    sc.sum(),
                    np.median(r),
                    r.max(),
                    np.sort(sc)[:12].sum(),
                )
            )
