import numpy as np, pandas as pd, os

D = r"C:\Users\Toni\Documents\Default Project\vn2-model-arena\data"
s = pd.read_csv(
    os.path.join(D, "sales.csv"), header=None, names=["series_id", "date", "v"]
)
print("RAW HEAD:")
print(s.head(4).to_string())
print("unique date count:", s["date"].nunique())
print("dates sample:", sorted(s["date"].unique())[:5], sorted(s["date"].unique())[-5:])
bad = s[~s["v"].astype(str).str.match(r"^-?[0-9.]+$")]
print("non numeric rows:", bad.head(10).to_string())
# rows per series
cnt = s.groupby("series_id").size()
print("rows/series:", cnt.describe().to_dict())
# check the order of dates for one series
one = s[s["series_id"] == s["series_id"].iloc[0]]
print(one.head(10).to_string())
print(one.tail(5).to_string())
