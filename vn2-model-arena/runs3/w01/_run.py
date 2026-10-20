"""Runner: python _run.py [key=value ...] [shuffle] [cuts=100,130]"""

import ast
import sys

import numpy as np

sys.path.insert(0, r"C:\Users\Toni\Documents\Default Project\vn2-model-arena\runs3\w01")
import exp  # noqa: E402


def coerce(v):
    try:
        return ast.literal_eval(v)
    except Exception:
        return v


cfg, flags, cuts = {}, [], (100, 130)
for a in sys.argv[1:]:
    if a == "shuffle":
        flags.append(a)
    elif a.startswith("cuts="):
        cuts = tuple(int(x) for x in a.split("=")[1].split(","))
    elif "=" in a:
        k, v = a.split("=", 1)
        cfg[k] = coerce(v)
    else:
        flags.append(a)
sc, el, fs = exp.run(cfg, cutoffs=cuts, shuffle="shuffle" in flags)
print("CFG", cfg, "| shuffle", "shuffle" in flags)
print(
    "  scores",
    {k: round(v, 1) for k, v in sc.items()},
    "mean",
    round(float(np.mean(list(sc.values()))), 1),
    "| fit_secs",
    fs,
    "| total",
    round(el, 2),
    "s",
)
