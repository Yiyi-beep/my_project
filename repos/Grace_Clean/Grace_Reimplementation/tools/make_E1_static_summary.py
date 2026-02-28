import pandas as pd, os

runs = r"results_E1_static_trained_20260225\E1_static_runs.csv"
outdir = r"results_E1_static_trained_20260225\plots"

df = pd.read_csv(runs)

summary = (df.groupby(["mode","flows","policy"])
             .agg(delivery_mean=("delivery_ratio","mean"),
                  delivery_std=("delivery_ratio","std"),
                  goodput_mean=("goodput_mbps","mean"),
                  goodput_std=("goodput_mbps","std"),
                  p95_mean=("p95_latency_ms","mean"),
                  p95_std=("p95_latency_ms","std"))
             .reset_index()
             .sort_values(["mode","flows","policy"]))

os.makedirs(outdir, exist_ok=True)
out_csv = os.path.join(outdir, "E1_static_summary_by_mode_flows_policy.csv")
summary.to_csv(out_csv, index=False)

print("Wrote:", out_csv)
print(summary.head(18).to_string(index=False))
