"""
Kaggle job: pull external knockdown response profiles for our 300 targets and test whether they transfer
to vcc-perturb (the decisive test for B5).

Sources
  K562  -- Replogle 2022 genome-wide screen, K562_gwps_normalized_bulk_01.h5ad (figshare, CC BY 4.0, 0.37 GB)
  RPE1  -- Replogle 2022, rpe1_normalized_bulk_01.h5ad (0.10 GB)
  T_*   -- Zhu et al. 2025 CD4+ T cells, GWCD4i.DE_stats.h5ad (16.8 GB, public S3, MIT). Only the rows for
           our targets are read, over HTTP range requests -- the file is never downloaded whole.

Test (same held-out logic as B3/B4, over the vcc-perturb knockdowns each source also covers):
  B4 baseline : pred_i = LOO mean response (+ median self-knockdown)
  source S    : pred_i = LOO mean + a * D_i, where D_i = S's profile for gene i minus S's own screen-wide mean,
                and `a` is least-squares fitted on the OTHER genes only.
  Gate (from B3): discrimination margin over B4 >= 2 x SE, and MSE no worse than B4.

Outputs in /kaggle/working: external_results.json, external_profiles.npz (centered D rows for our 300
targets in vcc-perturb gene space, NaN where a source lacks the target, plus full-data fitted scales).
"""
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

code_dir = next(Path("/kaggle/input").rglob("config.py")).parent
sys.path.insert(0, str(code_dir))
import os  # noqa: E402

os.environ["VCC_PERTURB_CSV"] = str(next(Path("/kaggle/input").rglob("vcc-perturb.csv")))
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "anndata", "fsspec", "aiohttp", "requests"], check=True)

import anndata as ad  # noqa: E402
import fsspec  # noqa: E402
import h5py  # noqa: E402
import requests  # noqa: E402
try:
    from anndata.io import read_elem  # noqa: E402
except ImportError:
    from anndata.experimental import read_elem  # noqa: E402

from models.gene_transfer_b2 import discrimination_score, load_response_matrix  # noqa: E402

WORK = Path("/kaggle/working")
TMP = Path("/kaggle/temp")
TMP.mkdir(parents=True, exist_ok=True)
REPLOGLE = {"K562": 35773217, "RPE1": 35775512}
ZHU_URL = "https://genome-scale-tcell-perturb-seq.s3.amazonaws.com/marson2025_data/GWCD4i.DE_stats.h5ad"


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


targets300 = pd.read_csv(next(Path("/kaggle/input").rglob("pert_counts.csv")))["target_gene"].str.strip().tolist()
vp_targets, vp_genes, R = load_response_matrix()
gpos = {g: i for i, g in enumerate(vp_genes)}
wanted = set(targets300) | set(vp_targets)
log(f"vcc-perturb {R.shape}; need profiles for {len(wanted)} genes (300 targets + 150 vcc-perturb knockdowns)")


def to_vp_space(mat, gene_names):
    """(n, G_src) -> (n, len(vp_genes)); vcc-perturb genes the source didn't measure get 0 (no evidence)."""
    out = np.zeros((mat.shape[0], len(vp_genes)), dtype=np.float32)
    src = np.array([i for i, g in enumerate(gene_names) if g in gpos])
    out[:, [gpos[gene_names[i]] for i in src]] = mat[:, src]
    return out


def symbol_col(df, candidates):
    return next(c for c in candidates if c in df.columns)


# ---- Replogle bulk: full files are small, load whole ----
profiles = {}   # source -> (dict gene -> row in vp space, screen-wide mean in vp space)
for name, fid in REPLOGLE.items():
    path = TMP / f"{name}.h5ad"
    if not path.exists():
        log(f"downloading Replogle {name}")
        with requests.get(f"https://plus.figshare.com/ndownloader/files/{fid}", stream=True, timeout=600) as r:
            r.raise_for_status()
            with open(path, "wb") as f:
                for chunk in r.iter_content(1 << 23):
                    f.write(chunk)
    a = ad.read_h5ad(path)
    log(f"{name}: {a.shape}; obs cols {list(a.obs.columns)[:12]}; var cols {list(a.var.columns)[:6]}")
    obs_gene = (a.obs[symbol_col(a.obs, ["gene", "gene_name", "target_gene"])].astype(str) if
                any(c in a.obs.columns for c in ["gene", "gene_name", "target_gene"]) else
                pd.Series([s.split("_")[1] for s in a.obs_names], index=a.obs_names))
    var_gene = (a.var[symbol_col(a.var, ["gene_name", "gene_symbol"])].astype(str).tolist()
                if any(c in a.var.columns for c in ["gene_name", "gene_symbol"]) else a.var_names.tolist())
    X = np.asarray(a.X.todense() if hasattr(a.X, "todense") else a.X, dtype=np.float32)
    keep = ~obs_gene.str.lower().str.contains("non-targeting").to_numpy()
    screen_mean = to_vp_space(X[keep].mean(0, keepdims=True), var_gene)[0]
    rows = {}
    for gname in wanted:
        hit = np.where((obs_gene.to_numpy() == gname) & keep)[0]
        if len(hit):
            rows[gname] = to_vp_space(X[hit].mean(0, keepdims=True), var_gene)[0]
    profiles[name] = (rows, screen_mean)
    log(f"{name}: covers {sum(g in rows for g in targets300)}/300 targets, "
        f"{sum(g in rows for g in vp_targets)}/150 vcc-perturb knockdowns")
    del a, X

# ---- Zhu T cells: read only our rows over HTTP ----
log("opening Zhu DE_stats over HTTP")
fs = fsspec.filesystem("http", block_size=2 << 20)
with fs.open(ZHU_URL, "rb", cache_type="blockcache") as fh, h5py.File(fh, "r") as f:
    obs = read_elem(f["obs"])
    var_gene = read_elem(f["var"])["gene_name"].astype(str).tolist()
    log(f"Zhu obs {obs.shape}, var {len(var_gene)}; layers {list(f['layers'].keys())}")
    layer = f["layers"]["zscore"]
    gcol = obs["target_contrast_gene_name"].astype(str).to_numpy()
    cond = obs["culture_condition"].astype(str).to_numpy()
    sel = np.where(np.isin(gcol, list(wanted)))[0]
    log(f"reading {len(sel)} of {len(gcol)} rows (layer is {'dense' if isinstance(layer, h5py.Dataset) else 'sparse'})")

    def read_rows(rows):
        if isinstance(layer, h5py.Dataset):
            return np.stack([layer[r] for r in rows]).astype(np.float32)
        indptr, data, idx = layer["indptr"], layer["data"], layer["indices"]
        out = np.zeros((len(rows), len(var_gene)), dtype=np.float32)
        for k, r in enumerate(rows):
            s, e = indptr[r], indptr[r + 1]
            out[k, idx[s:e]] = data[s:e]
        return out

    Z = []
    for i in range(0, len(sel), 50):
        Z.append(read_rows(sel[i:i + 50]))
        if i == 0:
            log("first 50 rows read")
    Z = np.nan_to_num(np.clip(np.concatenate(Z), -50, 50))   # z = logFC/SE; clip extreme tails before fitting
    log("Zhu rows read")

for c in ["Rest", "Stim8hr", "Stim48hr"]:
    m = cond[sel] == c
    rows = {g: to_vp_space(Z[m][gcol[sel][m] == g][:1], var_gene)[0] for g in set(gcol[sel][m])}
    # centering reference: mean over the rows we read for this condition (~450 genes, not target-selected by effect)
    profiles[f"T_{c}"] = (rows, to_vp_space(Z[m].mean(0, keepdims=True), var_gene)[0])
t_rows = {g: np.mean([profiles[f"T_{c}"][0][g] for c in ["Rest", "Stim8hr", "Stim48hr"] if g in profiles[f"T_{c}"][0]], 0)
          for g in set().union(*(profiles[f"T_{c}"][0] for c in ["Rest", "Stim8hr", "Stim48hr"]))}
profiles["T_mean"] = (t_rows, np.mean([profiles[f"T_{c}"][1] for c in ["Rest", "Stim8hr", "Stim48hr"]], 0))
for c in ["T_Rest", "T_Stim8hr", "T_Stim48hr", "T_mean"]:
    rows = profiles[c][0]
    log(f"{c}: covers {sum(g in rows for g in targets300)}/300 targets, {sum(g in rows for g in vp_targets)}/150 vcc-perturb")

# ---- decisive test ----
n = len(vp_targets)
self_cols = np.array([gpos[g] for g in vp_targets])
self_z = R[np.arange(n), self_cols]
loo_mu = (R.sum(0) - R) / (n - 1)
b4 = loo_mu.copy()
for i in range(n):
    b4[i, self_cols[i]] = np.median(np.delete(self_z, i))
resid = R - loo_mu
approx_se = lambda k: float(np.sqrt(1 / 12) / np.sqrt(k))  # noqa: E731


def centered_rows(src):
    rows, mean = profiles[src]
    have = np.array([g in rows for g in vp_targets])
    D = np.zeros_like(R, dtype=np.float32)
    D[have] = np.stack([rows[g] for g, h in zip(vp_targets, have) if h]) - mean
    return D, have


results, fitted_scale = {}, {}
for src in profiles:
    D, have = centered_rows(src)
    idx = np.where(have)[0]
    if len(idx) < 20:
        results[src] = {"n": int(len(idx)), "skipped": "too few overlapping genes"}
        continue
    num = (D * resid).sum(1)
    den = (D * D).sum(1)
    pred = b4.copy()
    for i in idx:
        others = idx[idx != i]
        a = num[others].sum() / max(den[others].sum(), 1e-12)
        pred[i] = b4[i] + a * D[i]
        pred[i, self_cols[i]] = b4[i, self_cols[i]]
    fitted_scale[src] = float(num[idx].sum() / max(den[idx].sum(), 1e-12))
    Rt, Pb, Ps = R[idx], b4[idx], pred[idx]
    dev_cos = [float(np.dot(D[i], resid[i]) / (np.linalg.norm(D[i]) * np.linalg.norm(resid[i]) + 1e-12)) for i in idx]
    disc_b4, disc_s = discrimination_score(Pb, Rt), discrimination_score(Ps, Rt)
    mse_b4, mse_s = float(np.mean((Pb - Rt) ** 2)), float(np.mean((Ps - Rt) ** 2))
    results[src] = {
        "n": int(len(idx)), "fitted_scale": fitted_scale[src],
        "median_cos_source_dev_vs_vcc_dev": float(np.median(dev_cos)),
        "mse_b4": mse_b4, "mse_source": mse_s,
        "disc_b4": disc_b4, "disc_source": disc_s,
        "required_margin": 2 * approx_se(len(idx)),
        "passes_gate": bool(disc_s - disc_b4 >= 2 * approx_se(len(idx)) and mse_s <= mse_b4),
    }
    r = results[src]
    log(f"{src:10s} n={r['n']:3d} cos(dev)={r['median_cos_source_dev_vs_vcc_dev']:+.3f}  "
        f"MSE {mse_b4:.4f}->{mse_s:.4f}  disc {disc_b4:.3f}->{disc_s:.3f} (need +{r['required_margin']:.3f})  "
        f"{'PASS' if r['passes_gate'] else 'fail'}")

(WORK / "external_results.json").write_text(json.dumps(results, indent=2))
out = {"targets": np.array(targets300), "vp_genes": np.array(vp_genes)}
for src, (rows, mean) in profiles.items():
    out[f"D_{src}"] = np.stack([rows[g] - mean if g in rows else np.full(len(vp_genes), np.nan, np.float32)
                                for g in targets300]).astype(np.float32)
    out[f"scale_{src}"] = np.float32(fitted_scale.get(src, np.nan))
np.savez_compressed(WORK / "external_profiles.npz", **out)
log("saved external_results.json and external_profiles.npz")
