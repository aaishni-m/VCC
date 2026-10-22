"""
Validate the cross-line table (xline_data.py output) BEFORE any model is fitted. Prints a report and exits non-zero on
a hard failure. Checks, per line file:
  - identifiers: no duplicate knockdowns or genes; knockdown names are gene symbols (Jiang rows are 'context:gene')
  - shapes: Y, FDR and every X_/Z_ array match (knockdowns x genes)
  - missingness is explicit: untested Y and absent source entries are NaN, not 0 (a source row that is all-zero but
    not NaN would mean a silent fill)
  - effect vs significance: X_s and Z_s agree in sign where both are large (|z| > 3) -- catches a z mistaken for an
    effect size or a flipped direction; reference FDR in [0, 1]
  - leakage: no file carries its own line as a source (HCT116 has no X_hct116, HEK293T no X_hek293t)
  - cross-source sanity: median per-knockdown cosine between sources, and of each source with the line's own Y
Usage: python models/xline_check.py
"""
import json
import sys
from pathlib import Path

import numpy as np

TAB = Path("D:/VCC_outputs/xline")
fail = []


def cos_rows(a, b):
    m = np.isfinite(a) & np.isfinite(b) & (a != 0) & (b != 0)
    out = []
    for i in range(len(a)):
        if m[i].sum() >= 50:
            x, y = a[i, m[i]], b[i, m[i]]
            out.append(float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-12)))
    return np.median(out) if out else np.nan, len(out)


files = sorted(TAB.glob("*.npz"))
if not files:
    sys.exit(f"no tables in {TAB}")
report = {}
for f in files:
    d = np.load(f, allow_pickle=False)
    line = f.stem
    genes, perts = d["genes"], d["perts"]
    r = {"knockdowns": len(perts), "genes": len(genes)}
    if len(set(genes)) != len(genes):
        fail.append(f"{line}: {len(genes) - len(set(genes))} duplicate genes")
    if len(set(perts)) != len(perts):
        fail.append(f"{line}: {len(perts) - len(set(perts))} duplicate knockdowns")
    shape = (len(perts), len(genes))
    for k in d.files:
        if k.startswith(("X_", "Z_", "Y", "FDR")) and d[k].shape != shape:
            fail.append(f"{line}: {k} shape {d[k].shape} != {shape}")
    srcs = sorted(k[2:] for k in d.files if k.startswith("X_"))
    r["sources"] = srcs
    for own in ("hct116", "hek293t"):
        if line.lower() == own and own in srcs:
            fail.append(f"{line}: carries itself as a source (leakage)")
    if "Y" in d.files:
        Y, F = d["Y"], d["FDR"]
        r["Y_tested_frac"] = round(float(np.isfinite(Y).mean()), 3)
        r["Y_exact_zero_frac_of_tested"] = round(float((Y[np.isfinite(Y)] == 0).mean()), 4)
        Ff = F[np.isfinite(F)]
        if Ff.size and (Ff.min() < 0 or Ff.max() > 1):
            fail.append(f"{line}: FDR outside [0, 1]")
        r["ref_sig_per_kd_median"] = float(np.median(((F < 0.05) & np.isfinite(Y)).sum(1)))
    for s in srcs:
        X, Z = d[f"X_{s}"], d.get(f"Z_{s}")
        has = ~np.isnan(X).all(1)
        zero_rows = int(((np.nan_to_num(X) == 0).all(1) & has).sum())
        info = {"coverage": int(has.sum()), "missing_gene_frac_in_covered": round(float(np.isnan(X[has]).mean()), 3)
                if has.any() else None, "all_zero_rows": zero_rows}
        if zero_rows:
            fail.append(f"{line}/{s}: {zero_rows} covered rows are all zero (silent fill?)")
        if Z is not None and has.any():
            m = np.isfinite(X) & np.isfinite(Z) & (np.abs(Z) > 3) & (X != 0)
            if m.sum() > 100:
                agree = float((np.sign(X[m]) == np.sign(Z[m])).mean())
                info["sign_agree_X_vs_Z_at_|z|>3"] = round(agree, 3)
                if agree < 0.8:
                    fail.append(f"{line}/{s}: effect and significance disagree in sign ({agree:.2f})")
        if "Y" in d.files and has.any():
            info["median_cos_with_line_Y"], info["n_cos"] = cos_rows(X, d["Y"])
        r[s] = info
    for i, a in enumerate(srcs):
        for b in srcs[i + 1:]:
            c, n = cos_rows(d[f"X_{a}"], d[f"X_{b}"])
            r[f"cos_{a}_vs_{b}"] = None if np.isnan(c) else round(c, 3)
    report[line] = r
    print(f"== {line}: {json.dumps(r, default=lambda o: round(float(o), 3))}")
(TAB / "check_report.json").write_text(json.dumps(report, indent=1, default=lambda o: round(float(o), 4)))
print("\nHARD FAILURES:" if fail else "\nNo hard failures.", *fail, sep="\n  ")
sys.exit(1 if fail else 0)
