import numpy as np, pandas as pd, os, time

D = r"C:\Users\Toni\Documents\Default Project\vn2-model-arena\data"
for f in ["sales.csv", "availability.csv"]:
    df = pd.read_csv(os.path.join(D, f), header=None, names=["series_id", "date", "v"])
    print(
        f,
        df.shape,
        df["v"].dtype,
        df["v"].head(3).tolist(),
        df["date"].min(),
        df["date"].max(),
        df["series_id"].nunique(),
    )
    print("  value counts:", df["v"].value_counts().head(5).to_dict())
