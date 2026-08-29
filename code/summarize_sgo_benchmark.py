import os
import pandas as pd, numpy as np, json
s = pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "sgo_benchmark_summary.csv"))
s = s.sort_values(["fid", "variant"], key=lambda c: c.map(lambda v: int(str(v)[1:]) if str(v).startswith("F") else 0) if c.name == "fid" else c)


def agree(our, pub, fmin):
    """Agreement judged on distance to the known optimum, which works when the
    published value is exactly 0."""
    a = abs(our - fmin); b = abs(pub - fmin)
    if b == 0 and a <= 1e-10:
        return "match (both at optimum)"
    if b == 0:
        return f"published at optimum, ours {a:.1e} away"
    r = a / b
    if 0.5 <= r <= 2.0:
        return f"match (within 2x, ratio {r:.2f})"
    if r < 0.5:
        return f"ours BETTER ({1/r:.1f}x closer)"
    return f"ours WORSE ({r:.1f}x further)"


rows = []
for _, r in s.iterrows():
    rows.append(dict(fid=r.fid, name=r["name"], variant=r.variant,
                     our_mean=r.our_mean, our_std=r.our_std,
                     pub_mean=r.published_mean, pub_std=r.published_std,
                     known_min=r.known_min,
                     verdict=agree(r.our_mean, r.published_mean, r.known_min)))
out = pd.DataFrame(rows)
pd.set_option("display.width", 220)
print(out.to_string(index=False))
print()
lit = out[out.variant == "def_le_off"]
flp = out[out.variant == "def_ge_off"]
for label, d in [("literal (as implemented, D1)", lit), ("flipped (D1-alt)", flp)]:
    n_match = int(d.verdict.str.startswith("match").sum())
    print(f"{label}: {n_match}/{len(d)} functions agree with the published SGO mean")
# head-to-head: which variant is closer to the published value, per function
merged = lit.merge(flp, on="fid", suffixes=("_lit", "_flp"))
merged["d_lit"] = (merged.our_mean_lit - merged.pub_mean_lit).abs()
merged["d_flp"] = (merged.our_mean_flp - merged.pub_mean_flp).abs()
print(f"literal closer to published on {(merged.d_lit <= merged.d_flp).sum()}/{len(merged)} functions")
