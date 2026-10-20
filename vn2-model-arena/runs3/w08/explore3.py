import numpy as np, pandas as pd, os

D = r"C:\Users\Toni\Documents\Default Project\vn2-model-arena\data"
for f in ["sales.csv", "availability.csv"]:
    df = pd.read_csv(os.path.join(D, f))
    print("===", f, df.columns.tolist(), df.shape)
    print("head:", df.head(3).to_dict("records"))
    u = sorted(df.iloc[:, 1].astype(str).unique())
    print("n unique col1:", len(u))
    print("first 3:", u[:3], "last 3:", u[-3:])
    nonnum = df[~df.iloc[:, 2].astype(str).str.match(r"^-?[0-9.]+$")]
    print("nonnumeric count:", len(nonnum))
    if len(nonnum):
        print(nonnum.iloc[:, 2].value_counts().head(10).to_dict())
        print(nonnum.head(5).to_string())
