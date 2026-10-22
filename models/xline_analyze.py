"""
Analyse the leave-one-line-out run (xline_models.py output) for VCC_NEXT_MODEL_REPORT.md / _DECISION.md.

- Groups: WELL_POWERED lines (H1, RPE1, HepG2, Jurkat: >= 80 reference-significant genes per knockdown, >= 190 knockdowns)
  are the basis for decisions; EXPLORATORY lines (Orion, Jiang) are reported but never decide.
- Paired comparisons: per knockdown, candidate minus comparator on the SAME knockdowns of the same held-out line
  (pds_proxy, dir_sig), bootstrap over knockdowns (2,000 resamples) -> mean difference and 95% interval.
- Weight stability: source weights of C_ridge / B_learned_weights across the outer folds.
Writes VCC_NEXT_MODEL_RESULTS.csv (one row per model x held-out line x subset, all proxy metrics, n, coverage, CI,
config) and prints the tables the report quotes.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

RES = Path("C:/VCC/results/VCC_NEXT_MODEL_PSEUDOBULK.csv")
PERKD = RES.with_suffix(".perkd.npz")
WELL = ["H1", "RPE1", "HepG2", "Jurkat"]
rng = np.random.default_rng(0)

df = pd.read_csv(RES)
pk = np.load(PERKD, allow_pickle=False)
lines = sorted(df.line.unique(), key=lambda L: (L not in WELL, L))
models = list(dict.fromkeys(df.model))


def paired(line, a, b, key):
    x, y = pk[f"{line}__{a}__{key}"], pk[f"{line}__{b}__{key}"]
    ok = np.isfinite(x) & np.isfinite(y)
    d = (x - y)[ok]
    if d.size < 5:
        return np.nan, np.nan, np.nan, int(d.size), np.nan
    boot = [d[rng.integers(0, d.size, d.size)].mean() for _ in range(2000)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(d.mean()), float(lo), float(hi), int(d.size), float((d > 0).mean())


out = df.copy()
out.insert(1, "group", np.where(out.line.isin(WELL), "well_powered", "exploratory"))
out.insert(0, "code_commit", "see VCC_NEXT_MODEL_REPORT.md")
out.to_csv("C:/VCC/VCC_NEXT_MODEL_RESULTS.csv", index=False)

allrows = df[df.subset == "all"]
print("== Proxy metrics per held-out line (subset = all knockdowns) ==")
for metric in ("pds_proxy", "dir_sig", "nmae_sig", "r2"):
    t = allrows.pivot_table(index="model", columns="line", values=metric)[lines]
    t["well_mean"] = t[WELL].mean(1); t["explor_mean"] = t[[L for L in lines if L not in WELL]].mean(1)
    print(f"\n-- {metric} --"); print(t.sort_values("well_mean", ascending=(metric == "nmae_sig")).round(3).to_string())

print("\n== Paired differences on identical knockdowns (mean, 95% CI, n, share of knockdowns improved) ==")
for comp in ("zero", "b5_dir", "B_equal_fitted"):
    for cand in ("C_ridge", "B_learned_weights", "A_denoise_equal", "D_ridge_denoise", "B_equal_fitted", "b5_dir", "src_k562",
                 "src_zhu", "src_hct116"):
        if cand == comp:
            continue
        for key in ("pds", "dir"):
            row = [f"{cand} - {comp} [{key}]"]
            for L in WELL:
                m, lo, hi, n, share = paired(L, cand, comp, key)
                row.append(f"{L}: {m:+.4f} [{lo:+.4f},{hi:+.4f}] n={n} up={share:.0%}" if np.isfinite(m) else f"{L}: n/a")
            print(" | ".join(row))

print("\n== Learned source weights across outer folds ==")
for mname in ("C_ridge", "B_learned_weights", "D_ridge_denoise"):
    rows = allrows[allrows.model == mname]
    for _, r in rows.iterrows():
        c = json.loads(r.config)
        w = c.get("w") or c.get("weights")
        print(f"{mname:18s} held-out {r.line:12s} " + " ".join(f"{k}={v:+.4f}" for k, v in w.items())
              + "".join(f" {k}={c[k]}" for k in ("lambda", "shrink_z", "alpha") if k in c))

ov = df[df.subset == "vcc_target_overlap"]
if len(ov):
    print("\n== Challenge-target overlap subset (knockdowns in the line that are also VCC validation targets) ==")
    print(ov.pivot_table(index="model", columns="line", values="pds_proxy").round(3).to_string())
    print(ov.groupby("line").n_knockdowns.first().to_string())
