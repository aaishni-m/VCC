"""
Kaggle job: log2 fold-change profiles of our 300 target knockdowns in the two external sources, on the challenge's
18,533-gene axis -- the inputs of B5 v3 (models/gene_transfer_b5v3.py).

Same formulas as the local benchmark (kaggle/localbench.py), which scored dense c = 2 log-fold-change best:
  K562  -- Replogle 2022 GENOME-WIDE raw bulk (K562_gwps_raw_bulk_01.h5ad, figshare, CC BY 4.0, 0.37 GB; the bench
           used the essential screen, which covers few of our targets):
           log2((CPM_target + 0.5) / (CPM_non-targeting + 0.5))
  T     -- Zhu 2025 CD4+ T cells, GWCD4i.DE_stats.h5ad layer `log_fc` (public S3, MIT), mean over all of the target's
           rows (all culture conditions). Only our rows are read, over HTTP range requests.
Genes a source did not measure get 0 (no change); targets a source does not cover are NaN rows.

Output in /kaggle/working: lfc_profiles.npz {targets, genes, K562, T} + lfc_profiles.json (coverage).
"""
import json
import subprocess
import sys
import time
from importlib.metadata import version
from pathlib import Path

pins = [f"{p}=={version(p)}" for p in ("numpy", "scipy", "pandas")]
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "anndata", "requests", "fsspec", "aiohttp", *pins],
               check=True)

import anndata as ad  # noqa: E402
import fsspec  # noqa: E402
import h5py  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import requests  # noqa: E402
import scipy.sparse as sp  # noqa: E402
try:
    from anndata.io import read_elem  # noqa: E402
except ImportError:
    from anndata.experimental import read_elem  # noqa: E402

T0 = time.time()
TMP, OUT = Path("/kaggle/temp"), Path("/kaggle/working")
TMP.mkdir(parents=True, exist_ok=True)
ZHU = "https://genome-scale-tcell-perturb-seq.s3.amazonaws.com/marson2025_data/GWCD4i.DE_stats.h5ad"
K562_GWPS_RAW_BULK = 35774443


def log(*a):
    print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)


controls = next(Path("/kaggle/input").rglob("pert_counts.csv")).parent
targets = pd.read_csv(controls / "pert_counts.csv")["target_gene"].str.strip().tolist()
G = pd.read_csv(controls / "gene_names.csv", header=None)[0].astype(str).str.strip().to_numpy()
if G[0].lower() in ("gene", "gene_name", "genes", "0"):
    G = G[1:]
assert len(G) == 18533, f"gene_names.csv: expected 18533 genes, got {len(G)}"
gpos = {g: i for i, g in enumerate(G)}
log(f"{len(targets)} targets, {len(G)} genes")


def align(mat, src_genes):
    out = np.zeros((mat.shape[0], len(G)), np.float32)
    sel = [(j, gpos[g]) for j, g in enumerate(src_genes) if g in gpos]
    out[:, [p for _, p in sel]] = mat[:, [j for j, _ in sel]]
    return out


# ---- K562 genome-wide raw bulk ----
path = TMP / "k562_gwps_raw_bulk.h5ad"
with requests.get(f"https://ndownloader.figshare.com/files/{K562_GWPS_RAW_BULK}", stream=True, timeout=900) as r:
    r.raise_for_status()
    with open(path, "wb") as f:
        for chunk in r.iter_content(1 << 24):
            f.write(chunk)
k = ad.read_h5ad(path)
k_t = (k.obs["gene"].astype(str) if "gene" in k.obs else pd.Series([s.split("_")[1] for s in k.obs_names])).to_numpy()
k_genes = (k.var["gene_name"] if "gene_name" in k.var else k.var_names).astype(str).to_numpy()
X = np.asarray(k.X.todense() if sp.issparse(k.X) else k.X, dtype=np.float64)
cpm = X / np.maximum(X.sum(1, keepdims=True), 1) * 1e6
ntc_cpm = cpm[k_t == "non-targeting"].mean(0)
first = {}
for i, t in enumerate(k_t):
    first.setdefault(t, i)
K = np.full((len(targets), len(G)), np.nan, np.float32)
k_have = [t for t in targets if t in first]
K[[targets.index(t) for t in k_have]] = align(
    np.log2((cpm[[first[t] for t in k_have]] + 0.5) / (ntc_cpm + 0.5)), k_genes)
log(f"K562 gwps: {X.shape}; covers {len(k_have)}/300 targets")
del k, X, cpm

# ---- Zhu T cells: log_fc rows for our targets ----
fs = fsspec.filesystem("http", block_size=2 << 20)
with fs.open(ZHU, "rb", cache_type="blockcache") as fh, h5py.File(fh, "r") as zf:
    z_tgt = read_elem(zf["obs"])["target_contrast_gene_name"].astype(str).to_numpy()
    z_gene = read_elem(zf["var"])["gene_name"].astype(str).to_numpy()
    rows = np.where(np.isin(z_tgt, targets))[0]
    ds = zf["layers"]["log_fc"]
    M = np.stack([ds[r] for r in rows]).astype(np.float64)
M = np.nan_to_num(np.clip(M, -50, 50))
per = pd.DataFrame(M).groupby(z_tgt[rows]).mean()
T = np.full((len(targets), len(G)), np.nan, np.float32)
t_have = [t for t in targets if t in per.index]
T[[targets.index(t) for t in t_have]] = align(per.loc[t_have].to_numpy(), z_gene)
log(f"Zhu: {len(rows)} rows; covers {len(t_have)}/300 targets")

cov = {"K562": len(k_have), "T": len(t_have), "both": len(set(k_have) & set(t_have)),
       "neither": [t for t in targets if t not in first and t not in per.index],
       "genes_measured": {"K562": int(sum(g in gpos for g in k_genes)), "T": int(sum(g in gpos for g in z_gene))}}
np.savez_compressed(OUT / "lfc_profiles.npz", targets=np.array(targets, dtype=str), genes=G.astype(str), K562=K, T=T)
(OUT / "lfc_profiles.json").write_text(json.dumps(cov, indent=2))
log("saved lfc_profiles.npz", json.dumps(cov))
