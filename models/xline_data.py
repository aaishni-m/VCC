"""
Cross-line training/validation table for the next model (VCC_NEXT_MODEL_PLAN.md §2). Runs locally.

For every TARGET line (a cell line with measured knockdown responses) and every knockdown p in its panel:
  Y[p, g]   = the line's measured log2 fold change (NaN where the gene was not tested), SIG = reference-significant
  X_s[p, g] = what SOURCE screen s says for the same knockdown and gene (NaN where s lacks the knockdown or the gene)
  Z_s[p, g] = s's own significance statistic for that entry (signed z where the source has one)

Target lines and responses: the PIE processed release (Verma et al. 2026, Arc; one DE pipeline for all):
  H1 (VCC-2025, arc_vcc_25), RPE1 / HepG2 / Jurkat (Replogle 2022 + Nadig 2025 essential), HCT116 / HEK293T (X-Atlas/Orion,
  genome-wide; subsampled), Jiang 2025 lines (cytokine-stimulated, pooled per base line).
Sources (the only features that exist for the challenge targets):
  k562   Replogle 2022 K562 genome-wide: log2((CPM_p + 0.5) / (CPM_ntc + 0.5)) from raw bulk; z from normalized bulk
  zhu    Zhu 2025 CD4+ T cells: `log_fc` and `zscore` layers, mean over culture conditions
  hct116, hek293t  Orion via PIE: lfc_true, and a signed z from its FDR (a lower bound on |z|; marked as such)
Leakage rule: a line is never a source for itself (HCT116's table has no hct116 feature, etc.); K562 is not a target.

Output: D:/VCC_outputs/xline/<line>.npz  (+ coverage.json)
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

PIE = Path("D:/VCC_outputs/pie-pre")
REP = Path("D:/VCC_outputs/replogle")
OUT = Path("D:/VCC_outputs/xline")
ZHU_URL = "https://genome-scale-tcell-perturb-seq.s3.amazonaws.com/marson2025_data/GWCD4i.DE_stats.h5ad"
ORION_SUBSAMPLE = 1500
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)


def pie_dir(name):
    d = PIE / name
    m = json.load(open(d / "meta.json"))
    ctx = np.load(d / "ctx_ids.npy"); pert = np.load(d / "pert_ids.npy")
    id_ctx = {v: k for k, v in m["context_to_id"].items()}; id_pert = {v: k for k, v in m["pert_to_id"].items()}
    rows = pd.DataFrame({"ctx": [id_ctx[i] for i in ctx], "pert": [id_pert[i] for i in pert]})
    arr = {k: np.load(d / f"{k}.npy", mmap_mode="r") for k in ("lfc_true", "fdr", "tested")}
    return m["genes"], rows, arr


def signed_z_from_fdr(lfc, fdr):
    """|z| implied by treating the BH-adjusted p as a p-value (conservative: adjusted p >= raw p)."""
    p = np.clip(np.nan_to_num(fdr, nan=1.0), 1e-300, 1.0)
    return np.sign(np.nan_to_num(lfc)) * norm.isf(p / 2)


# ---------------- target lines ----------------
targets = {}                                  # line -> (genes, perts, Y, FDR)
g, rows, a = pie_dir("arc_vcc_25")
idx = np.arange(len(rows))
targets["H1"] = (g, rows.pert.to_numpy()[idx], np.asarray(a["lfc_true"])[idx], np.asarray(a["fdr"])[idx])
g, rows, a = pie_dir("replogle_nadig_essential")
for line, ctx in (("RPE1", "rpe1"), ("HepG2", "hepg2"), ("Jurkat", "jurkat")):
    idx = np.where(rows.ctx == ctx)[0]
    targets[line] = (g, rows.pert.to_numpy()[idx], np.asarray(a["lfc_true"][idx]), np.asarray(a["fdr"][idx]))
og, orows, oa = pie_dir("x_atlas_orion")
g, rows, a = pie_dir("jiang")
for base in sorted({c.split("_")[0] for c in rows.ctx} - {"k562"}):
    idx = np.where(rows.ctx.str.startswith(base + "_"))[0]
    targets[f"Jiang_{base}"] = (g, (rows.ctx + ":" + rows.pert).to_numpy()[idx],
                                np.asarray(a["lfc_true"][idx]), np.asarray(a["fdr"][idx]))
log("PIE target lines loaded:", {k: len(v[1]) for k, v in targets.items()})

# ---------------- sources: Orion (also targets, subsampled) ----------------
orion = {}
for ctx in ("hct116", "hek293t"):
    idx = np.where(orows.ctx == ctx)[0]
    orion[ctx] = {"genes": og, "perts": orows.pert.to_numpy()[idx], "lfc": oa["lfc_true"], "fdr": oa["fdr"], "rows": idx}

# ---------------- sources: K562 genome-wide ----------------
import anndata as ad  # noqa: E402
import scipy.sparse as sp  # noqa: E402


def bulk(path):
    b = ad.read_h5ad(path)
    t = (b.obs["gene"].astype(str) if "gene" in b.obs else pd.Series([s.split("_")[1] for s in b.obs_names])).to_numpy()
    gn = (b.var["gene_name"] if "gene_name" in b.var else b.var_names).astype(str).to_numpy()
    X = np.asarray(b.X.todense() if sp.issparse(b.X) else b.X, dtype=np.float64)
    return t, gn, X


t, gn, X = bulk(REP / "K562_gwps_raw_bulk.h5ad")
cpm = X / np.maximum(X.sum(1, keepdims=True), 1) * 1e6
ntc = cpm[t == "non-targeting"].mean(0)
first = pd.Series(np.arange(len(t)), index=t).groupby(level=0).first()
first = first.drop("non-targeting", errors="ignore")
k562 = {"genes": gn, "perts": first.index.to_numpy(),
        "lfc": np.log2((cpm[first.to_numpy()] + 0.5) / (ntc + 0.5)).astype(np.float32)}
tz, gz, Xz = bulk(REP / "K562_gwps_normalized_bulk.h5ad")
Xz[~np.isfinite(Xz)] = 0
fz = pd.Series(np.arange(len(tz)), index=tz).groupby(level=0).first()
fz = fz.drop("non-targeting", errors="ignore")
log(f"K562 genome-wide: {len(k562['perts'])} knockdowns x {len(gn)} genes")

# ---------------- sources: Zhu T cells (read only the knockdowns we need) ----------------
need = set()
for line, (_, perts, _, _) in targets.items():
    need |= {p.split(":")[-1] for p in perts}
rng = np.random.default_rng(0)
for ctx, o in orion.items():
    keep = [i for i, p in enumerate(o["perts"]) if p in set(k562["perts"])]
    keep = np.sort(rng.choice(keep, size=min(ORION_SUBSAMPLE, len(keep)), replace=False))
    targets[ctx.upper() if ctx == "hct116" else "HEK293T"] = (o["genes"], o["perts"][keep],
                                                              np.asarray(o["lfc"][o["rows"][keep]]),
                                                              np.asarray(o["fdr"][o["rows"][keep]]))
    need |= set(o["perts"][keep])
vcc_targets = pd.read_csv("D:/VCC_outputs/b5v2in/controls/pert_counts.csv")["target_gene"].str.strip().tolist()
need |= set(vcc_targets)
if len(sys.argv) > 1:                                  # extra target lists (e.g. the final round's pert_counts.csv)
    for f in sys.argv[1:]:
        need |= set(pd.read_csv(f)["target_gene"].str.strip())

import fsspec  # noqa: E402
import h5py  # noqa: E402
try:
    from anndata.io import read_elem  # noqa: E402
except ImportError:
    from anndata.experimental import read_elem  # noqa: E402

fs = fsspec.filesystem("http", block_size=2 << 20)
with fs.open(ZHU_URL, "rb", cache_type="blockcache") as fh, h5py.File(fh, "r") as zf:
    zt = read_elem(zf["obs"])["target_contrast_gene_name"].astype(str).to_numpy()
    zg = read_elem(zf["var"])["gene_name"].astype(str).to_numpy()
    sel = np.where(np.isin(zt, list(need)))[0]
    log(f"Zhu: reading {len(sel)} rows for {len(set(zt[sel]))} knockdowns (2 layers)")
    lay = {}
    for name in ("log_fc", "zscore"):
        ds = zf["layers"][name]
        parts = []
        for i in range(0, len(sel), 200):
            parts.append(np.stack([ds[r] for r in sel[i:i + 200]]).astype(np.float32))
            if i % 2000 == 0:
                log(f"  {name}: {i}/{len(sel)} rows")
        lay[name] = np.nan_to_num(np.clip(np.concatenate(parts), -50, 50))
zhu = {"genes": zg}
for name, M in lay.items():
    per = pd.DataFrame(M).groupby(zt[sel]).mean()
    zhu["perts"] = per.index.to_numpy(); zhu[name] = per.to_numpy(np.float32)
log(f"Zhu: {len(zhu['perts'])} knockdowns x {len(zg)} genes")


# ---------------- assemble per target line ----------------
def feature(src_genes, src_perts, values, line_genes, line_perts):
    """(n_line_perts, n_line_genes) float32, NaN where the source lacks the knockdown or the gene."""
    gp = {x: i for i, x in enumerate(src_genes)}; pp = {x: i for i, x in enumerate(src_perts)}
    gi = np.array([gp.get(x, -1) for x in line_genes]); pi = np.array([pp.get(x.split(":")[-1], -1) for x in line_perts])
    out = np.full((len(line_perts), len(line_genes)), np.nan, np.float32)
    rows_ok, cols_ok = np.where(pi >= 0)[0], np.where(gi >= 0)[0]
    if len(rows_ok) and len(cols_ok):
        sub = values(pi[rows_ok]) if callable(values) else values[pi[rows_ok]]
        out[np.ix_(rows_ok, cols_ok)] = np.asarray(sub)[:, gi[cols_ok]]
    return out


OUT.mkdir(parents=True, exist_ok=True)
coverage = {}
kz_g = gz
for line, (genes, perts, Y, FDR) in targets.items():
    genes = list(genes)
    f = {
        "X_k562": feature(gn, k562["perts"], k562["lfc"], genes, perts),
        "Z_k562": feature(kz_g, fz.index.to_numpy(), lambda r: Xz[fz.to_numpy()[r]], genes, perts),
        "X_zhu": feature(zg, zhu["perts"], zhu["log_fc"], genes, perts),
        "Z_zhu": feature(zg, zhu["perts"], zhu["zscore"], genes, perts),
    }
    for ctx, o in orion.items():
        if ctx.upper() == line.upper():
            continue                                    # never a source for itself
        lf = lambda r, o=o: o["lfc"][o["rows"][r]]       # noqa: E731
        fd = lambda r, o=o: o["fdr"][o["rows"][r]]       # noqa: E731
        f[f"X_{ctx}"] = feature(o["genes"], o["perts"], lf, genes, perts)
        f[f"Z_{ctx}"] = signed_z_from_fdr(f[f"X_{ctx}"], feature(o["genes"], o["perts"], fd, genes, perts)).astype(np.float32)
    Y = np.asarray(Y, np.float32); FDR = np.asarray(FDR, np.float32)
    np.savez_compressed(OUT / f"{line}.npz", genes=np.array(genes, dtype=str), perts=np.array(perts, dtype=str),
                        Y=Y, FDR=FDR, **f)
    cov = {s[2:]: int((~np.isnan(v).all(1)).sum()) for s, v in f.items() if s.startswith("X_")}
    coverage[line] = {"knockdowns": len(perts), "genes": len(genes),
                      "reference_significant_per_kd_median": float(np.median(((FDR < 0.05) & np.isfinite(Y)).sum(1))),
                      "source_coverage": cov}
    log(line, coverage[line])

# the challenge targets themselves (features only, no responses): same sources
genes18 = pd.read_csv("D:/VCC_outputs/b5v2in/controls/gene_names.csv")["gene_name"].astype(str).tolist()
f = {"X_k562": feature(gn, k562["perts"], k562["lfc"], genes18, vcc_targets),
     "Z_k562": feature(kz_g, fz.index.to_numpy(), lambda r: Xz[fz.to_numpy()[r]], genes18, vcc_targets),
     "X_zhu": feature(zg, zhu["perts"], zhu["log_fc"], genes18, vcc_targets),
     "Z_zhu": feature(zg, zhu["perts"], zhu["zscore"], genes18, vcc_targets)}
for ctx, o in orion.items():
    f[f"X_{ctx}"] = feature(o["genes"], o["perts"], lambda r, o=o: o["lfc"][o["rows"][r]], genes18, vcc_targets)
    f[f"Z_{ctx}"] = signed_z_from_fdr(f[f"X_{ctx}"], feature(o["genes"], o["perts"], lambda r, o=o: o["fdr"][o["rows"][r]],
                                                            genes18, vcc_targets)).astype(np.float32)
np.savez_compressed(OUT / "VCC_val.npz", genes=np.array(genes18, dtype=str), perts=np.array(vcc_targets, dtype=str), **f)
coverage["VCC_val"] = {"knockdowns": len(vcc_targets),
                       "source_coverage": {s[2:]: int((~np.isnan(v).all(1)).sum()) for s, v in f.items() if s.startswith("X_")}}
json.dump(coverage, open(OUT / "coverage.json", "w"), indent=1)
log("saved", OUT, json.dumps(coverage["VCC_val"]))
