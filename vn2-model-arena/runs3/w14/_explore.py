import pandas as pd, numpy as np, os

d = os.path.join(os.path.dirname(__file__), "..", "..", "data")
s = pd.read_csv(os.path.join(d, "sales.csv"))
a = pd.read_csv(os.path.join(d, "availability.csv"))
i = pd.read_csv(os.path.join(d, "initial_state.csv"))
p = pd.read_csv(os.path.join(d, "product_attributes.csv"))
print("sales", s.shape, s.columns.tolist())
print(s.head())
print("avail", a.shape, a.columns.tolist())
print(a.head())
print("sales nunique series", s.series_id.nunique(), "weeks", s.date.nunique())
print(s.groupby("series_id").size().describe())
print("avail values", a.iloc[:, 2].unique()[:10] if a.shape[1] > 2 else "?")
# pivot
if s.shape[1] == 3 and "sales" in s.columns.tolist() or s.shape[1] == 3:
    pass
print(s.columns.tolist())
