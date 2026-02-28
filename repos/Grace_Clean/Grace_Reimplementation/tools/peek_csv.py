import sys, pandas as pd
p = sys.argv[1]
df = pd.read_csv(p)
print("path:", p)
print("rows:", len(df))
print("cols:", list(df.columns))
print(df.head(3).to_string(index=False))
