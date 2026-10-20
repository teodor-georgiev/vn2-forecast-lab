import numpy as np, pandas as pd

base = r"C:\Users\Toni\Documents\Default Project\vn2-model-arena\data"
s = pd.read_csv(base + r"\sales.csv")
a = pd.read_csv(base + r"\availability.csv")
i = pd.read_csv(base + r"\initial_state.csv")
ids_s = s["series_id"].drop_duplicates().tolist()
ids_a = a["series_id"].drop_duplicates().tolist()
ids_i = i["series_id"].tolist()
print("n series", len(ids_s), len(ids_a), len(ids_i))
print("same order sales==avail", ids_s == ids_a)
print("same order sales==init", ids_s == ids_i)
print("same set init==sales", set(ids_s) == set(ids_i))
v = i[["on_hand", "arrival_w1", "arrival_w2"]].to_numpy(float)
print("init value counts on_hand", np.unique(v[:, 0], return_counts=True))
print("init rows unique triples", len(set(map(tuple, v))), "of", len(v))
print("init nonzero w1", (v[:, 1] > 0).sum(), "nonzero w2", (v[:, 2] > 0).sum())
