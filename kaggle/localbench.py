"""
Kaggle job: FAITHFUL local benchmark -- the official cell-eval2 `vcc2026` scorer on REAL held-out single cells.

Why: two rounds of proxies (H1 z-scores, a synthetic truth) misled model choices (B7 scored 0.051 vs B5 0.128 on the
real leaderboard). Leaders evaluate on real single-cell holdouts; this does the same with public data.

Pseudo-context: Replogle 2022 RPE1 (CC BY 4.0) real single cells -- a panel of knockdowns shared with our two
sources (Replogle K562 essential screen, Zhu 2025 CD4+ T cells). Control cells are split in two, as in the VCC:
'released' controls (predictions are generated from these) and 'held-out' controls (the scorer's reference).
Predictions are transferred from K562 + T cells only; RPE1's measured effects are used only by the scorer.

Scoring is cell-eval2's own pipeline with the vcc2026 preset: baseline arm (`baseline`), real bundle with replicate
anchors (`prep-real-bundle`), `run`, then `score --real-bundle` -> the same (u-b)/(r-b) scaling as the leaderboard.

Calibration check built in: the 'B5like' and 'B7like' variants should reproduce the REAL leaderboard pattern
(B5: fidelity ~0, pds higher; B7: fidelity ~-0.23, pds lower). If they do not, these numbers should not be trusted.
"""
import json
import shutil
import subprocess
import sys
import time
from importlib.metadata import version
from pathlib import Path

pins = [f"{p}=={version(p)}" for p in ("numpy", "scipy", "pandas")]
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "anndata", "requests", "fsspec", "aiohttp", "uv", *pins],
               check=True)
subprocess.run(["uv", "tool", "install", "--force", "cell-eval2", "--with", "pdex"], check=True)  # isolated scorer env + CPU DE engine
CE = str(Path.home() / ".local" / "bin" / "cell-eval2")

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

code_dir = Path("/kaggle/temp/vcc")
shutil.copytree(next(Path("/kaggle/input").rglob("config.py")).parent, code_dir, dirs_exist_ok=True)
sys.path.insert(0, str(code_dir))
from models.transform_candidates import t2_log1p_additive  # noqa: E402

T0 = time.time()
TMP, OUT = Path("/kaggle/temp/bench"), Path("/kaggle/working")
TMP.mkdir(parents=True, exist_ok=True)
N_PANEL, MIN_CELLS, N_PRED = 300, 40, 100
BETA = 1.5
rng = np.random.default_rng(0)


def log(*a):
    print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)


def fetch(fid, dest):
    if not dest.exists():
        with requests.get(f"https://ndownloader.figshare.com/files/{fid}", stream=True, timeout=900) as r:
            r.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in r.iter_content(1 << 24):
                    f.write(chunk)
    return dest


# ---------------- real RPE1 cells ----------------
rpe = fetch(35775606, TMP / "rpe1_single.h5ad")
k_raw = ad.read_h5ad(fetch(35773070, TMP / "k562_raw_bulk.h5ad"))
k_norm = ad.read_h5ad(fetch(35780870, TMP / "k562_norm_bulk.h5ad"))
a = ad.read_h5ad(rpe, backed="r")
gcol = "gene" if "gene" in a.obs else "gene_name"
labels = a.obs[gcol].astype(str).to_numpy()
genes = (a.var["gene_name"] if "gene_name" in a.var else a.var_names).astype(str).to_numpy()
a.file.close()
gpos = {}
for i, g in enumerate(genes):
    gpos.setdefault(g, i)
keep_gene = np.array(sorted(gpos.values()))                  # first occurrence of each symbol
G = genes[keep_gene]
log(f"RPE1: {len(labels)} cells, {len(G)} genes")


def bulk_targets(b):
    col = "gene" if "gene" in b.obs else None
    return (b.obs[col].astype(str) if col else pd.Series([s.split("_")[1] for s in b.obs_names])).to_numpy()


k_raw_t, k_norm_t = bulk_targets(k_raw), bulk_targets(k_norm)

# Zhu T-cell rows (read over HTTP, only what we need)
ZHU = "https://genome-scale-tcell-perturb-seq.s3.amazonaws.com/marson2025_data/GWCD4i.DE_stats.h5ad"
fs = fsspec.filesystem("http", block_size=2 << 20)
zfh = fs.open(ZHU, "rb", cache_type="blockcache")
zf = h5py.File(zfh, "r")
z_obs = read_elem(zf["obs"])
z_gene = read_elem(zf["var"])["gene_name"].astype(str).to_numpy()
z_tgt = z_obs["target_contrast_gene_name"].astype(str).to_numpy()

counts = pd.Series(labels).value_counts()
cands = [t for t, n in counts.items() if t != "non-targeting" and n >= MIN_CELLS and t in set(k_raw_t)
         and t in set(k_norm_t) and t in set(z_tgt) and t in gpos]
panel = sorted(rng.choice(cands, size=min(N_PANEL, len(cands)), replace=False).tolist())
log(f"panel: {len(panel)} knockdowns (of {len(cands)} eligible with >= {MIN_CELLS} cells in all sources)")

ntc = np.where(labels == "non-targeting")[0]
ntc = rng.permutation(ntc)
ntc_rel, ntc_held = np.sort(ntc[: len(ntc) // 2]), np.sort(ntc[len(ntc) // 2:])
pert_rows = np.sort(np.where(np.isin(labels, panel))[0])


def read_rows(rows):
    with h5py.File(rpe, "r") as f:
        x = f["X"]
        blocks = [x[rows[i:i + 2000]][:, keep_gene] for i in range(0, len(rows), 2000)]
    return sp.csr_matrix(np.concatenate(blocks).astype(np.float32))


X_rel = read_rows(ntc_rel)
real_rows = np.sort(np.concatenate([pert_rows, ntc_held]))
X_real = read_rows(real_rows)
real = ad.AnnData(X=X_real, obs=pd.DataFrame({"target_gene": labels[real_rows]}, index=[f"c{i}" for i in real_rows]),
                  var=pd.DataFrame(index=G))
real.write_h5ad(TMP / "real.h5ad")
X_ctrl_held = X_real[np.isin(real_rows, ntc_held)]
log(f"real reference: {real.shape} ({len(pert_rows)} perturbed + {len(ntc_held)} held-out controls); "
    f"released controls: {X_rel.shape[0]}")

# ---------------- pseudobulk reference for the z-space emitters (from released controls) ----------------
lib = np.asarray(X_rel.sum(1)).ravel()
pb = []
for _ in range(300):
    idx = rng.choice(X_rel.shape[0], 15, replace=False)
    s = np.asarray(X_rel[idx].sum(0)).ravel()
    pb.append(np.log1p(s / s.sum() * 1e4))
pb = np.array(pb)
ref = {"sigma_log1p_cp10k_floored": np.maximum(pb.std(0), 1e-3), "raw_mean_count": np.asarray(X_rel.mean(0)).ravel()}

# ---------------- source effects on the RPE1 gene axis ----------------
def align(mat, src_genes):
    out = np.zeros((mat.shape[0], len(G)), np.float64)
    pos = {g: i for i, g in enumerate(G)}
    sel = [(j, pos[g]) for j, g in enumerate(src_genes) if g in pos]
    out[:, [p for _, p in sel]] = mat[:, [j for j, _ in sel]]
    return out


def first_rows(tgts, wanted):
    d = {}
    for i, t in enumerate(tgts):
        d.setdefault(t, i)
    return [d[t] for t in wanted]


Xn = np.nan_to_num(np.asarray(k_norm.X.todense() if sp.issparse(k_norm.X) else k_norm.X, dtype=np.float64),
                   nan=0, posinf=0, neginf=0)
kn_genes = k_norm.var["gene_name"].astype(str).to_numpy()
K_z = align(Xn[first_rows(k_norm_t, panel)] - Xn[k_norm_t != "non-targeting"].mean(0), kn_genes)

Xr = np.asarray(k_raw.X.todense() if sp.issparse(k_raw.X) else k_raw.X, dtype=np.float64)
cpm = Xr / np.maximum(Xr.sum(1, keepdims=True), 1) * 1e6
ntc_cpm = cpm[k_raw_t == "non-targeting"].mean(0)
kr_genes = k_raw.var["gene_name"].astype(str).to_numpy()
K_lfc = align(np.log2((cpm[first_rows(k_raw_t, panel)] + 0.5) / (ntc_cpm + 0.5)), kr_genes)

zrows = np.where(np.isin(z_tgt, panel))[0]
lay = {}
for name in ("zscore", "log_fc"):
    ds = zf["layers"][name]
    lay[name] = np.stack([ds[r] for r in zrows]).astype(np.float64)
zf.close()
zfh.close()
T = {}
for name, M in lay.items():
    M = np.nan_to_num(np.clip(M, -50, 50))
    per = pd.DataFrame(M).groupby(z_tgt[zrows]).mean()
    T[name] = align(per.loc[panel].to_numpy(), z_gene)
T_z = T["zscore"] - T["zscore"].mean(0)
T_lfc = T["log_fc"]
log("source effects ready (K562 z/lfc, T-cell z/lfc)")

self_col = np.array([np.where(G == t)[0][0] for t in panel])


def unit(M):
    return M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-12)


def sparsify(M, k):
    if k is None:
        return M
    out = np.zeros_like(M)
    idx = np.argsort(-np.abs(M), axis=1)[:, :k]
    r = np.arange(len(M))[:, None]
    out[r, idx] = M[r, idx]
    return out


def topk_level(M, k=200):
    return float(np.median(np.sort(np.abs(M), axis=1)[:, -k:]))


# ---------------- emitters ----------------
def emit_t2(z):
    idx = rng.choice(X_rel.shape[0], N_PRED, replace=True)
    return t2_log1p_additive(X_rel[idx], z, ref, rng, beta=BETA)


def emit_fold(fold):
    """Edit only genes with fold != 1: binomial thinning (down) / Poisson addition of expected counts (up)."""
    idx = rng.choice(X_rel.shape[0], N_PRED, replace=True)
    out = np.asarray(X_rel[idx].todense(), dtype=np.int64)
    g = np.flatnonzero(fold != 1)
    f = np.clip(fold[g], 1 / 20, 20)
    share = ref["raw_mean_count"] / ref["raw_mean_count"].sum()
    tot = out.sum(1, keepdims=True)
    d, u = g[f < 1], g[f > 1]
    if d.size:
        out[:, d] = rng.binomial(out[:, d], f[f < 1][None, :])
    if u.size:
        out[:, u] += rng.poisson((f[f > 1] - 1)[None, :] * tot * share[u][None, :])
    return out


def z_to_fold(z):
    return np.exp(BETA * z * ref["sigma_log1p_cp10k_floored"])


def write_pred(name, profiles, kind):
    cells, labs = [], []
    for i, t in enumerate(panel):
        p = profiles[i].copy()
        p[self_col[i]] = 0 if kind != "t2" else p[self_col[i]]
        c = emit_t2(p) if kind == "t2" else emit_fold(z_to_fold(p) if kind == "zfold" else 2.0 ** p)
        cells.append(sp.csr_matrix(c.astype(np.float32)))
        labs += [t] * N_PRED
    # cell-eval2 requires the same perturbation set on both sides; the vcc2026 preset scores against the REAL
    # controls (control_source: real), so the held-out control cells are appended unchanged
    cells.append(X_ctrl_held)
    labs += ["non-targeting"] * X_ctrl_held.shape[0]
    X = sp.vstack(cells).tocsr()
    pred = ad.AnnData(X=X, obs=pd.DataFrame({"target_gene": labs}, index=[f"p{i}" for i in range(len(labs))]),
                      var=pd.DataFrame(index=G))
    path = TMP / f"pred_{name}.h5ad"
    pred.write_h5ad(path)
    return path


# ---------------- variants ----------------
m_shared = align(Xn[k_norm_t != "non-targeting"].mean(0, keepdims=True), kn_genes)[0]
dir_unit = unit(K_z) + unit(T_z)
VARIANTS = {}
# B5-like: dense, shared mean + unit-sum directions, T2. Measured on B5's real profiles: direction top-200 level 0.94,
# ||0.5 x mean|| / median ||direction|| = 0.576 -- reproduced here with the K562 screen-wide mean as the shared part
def b5_like(direction):
    d = direction / topk_level(direction) * 0.94
    m = m_shared / (np.linalg.norm(m_shared) + 1e-12) * np.median(np.linalg.norm(d, axis=1)) * 0.576
    return d + m


def b5v2_direction(zhu_soft, agree_p):
    """models/gene_transfer_b5v2.py's combined direction: Zhu z soft-thresholded, then sign-agreement weighting."""
    tz = np.sign(T_z) * np.maximum(np.abs(T_z) - zhu_soft, 0) if zhu_soft > 0 else T_z
    uk, ut = unit(K_z), unit(tz)
    d = uk + ut
    if agree_p > 0:
        a = np.abs(uk) + np.abs(ut)
        d = d * np.divide(np.abs(d), a, out=np.zeros_like(d), where=a > 0) ** agree_p
    return d


VARIANTS["B5like_T2_dense"] = (b5_like(dir_unit), "t2")
VARIANTS["B5v2like_T2_dense"] = (b5_like(b5v2_direction(1.0, 1.0)), "t2")
VARIANTS["B5like_zhusoft1"] = (b5_like(b5v2_direction(1.0, 0.0)), "t2")
VARIANTS["B5like_agree1"] = (b5_like(b5v2_direction(0.0, 1.0)), "t2")
# B7-like: magnitude-preserving (sources put on a common scale), no mean, top-400, z-fold emitter
kz_s, tz_s = np.median(np.linalg.norm(K_z, axis=1)), np.median(np.linalg.norm(T_z, axis=1))
mag = (K_z / kz_s + T_z / tz_s) / 2
mag = mag / topk_level(mag)                                  # median top-200 |z| = 1, as B7
VARIANTS["B7like_zfold_k400"] = (sparsify(mag, 400), "zfold")
VARIANTS["zfold_k4000"] = (sparsify(mag, 4000), "zfold")
VARIANTS["zfold_dense"] = (mag, "zfold")
# log-fold-change representation (sources' own log2 fold changes, averaged), applied directly as folds
lfc = (K_lfc + T_lfc) / 2
for k in (400, 4000, None):
    for c in (0.5, 1.0):
        VARIANTS[f"lfc_c{c}_k{k or 'all'}"] = (c * sparsify(lfc, k), "lfc")
# v3 round: dense log-fold-change above c=1 (round 1: pds rose 0.40 -> 0.60 from c=0.5 to 1 with no sign of saturating)
for c in (1.5, 2.0, 3.0):
    VARIANTS[f"lfc_c{c}_kall"] = (c * lfc, "lfc")

# ---------------- official scoring ----------------
CFG = ["--preset", "vcc2026", "--pert-col", "target_gene", "--set", "de.backend=pdex"]


def ce(*args):
    r = subprocess.run([CE, *args], capture_output=True, text=True)
    if r.returncode != 0:
        log("cell-eval2 failed:", " ".join(args[:3]), "\n", r.stdout[-3000:], "\n", r.stderr[-3000:])
        raise SystemExit(1)
    return r


# Reuse a previous run's bundle (attached as a kernel source) when it was built on this exact panel -- the panel,
# control split and real.h5ad are all fixed by the seed, so the bundle is identical; saves ~47 min.
prev = next((m.parent for m in Path("/kaggle/input").rglob("localbench_meta.json")
             if (m.parent / "bundle" / "anchor_meta.json").exists()), None)
if prev is not None and json.loads((prev / "localbench_meta.json").read_text())["panel"] == panel:
    shutil.copytree(prev / "bundle", TMP / "bundle", dirs_exist_ok=True)
    log(f"reusing real bundle from {prev} (same panel)")
else:
    ce("baseline", "-ar", str(TMP / "real.h5ad"), "-o", str(TMP / "baseline"), "--save-pred",
       str(TMP / "baseline_pred.h5ad"), *CFG)
    log("baseline arm built")
    ce("prep-real-bundle", "--real", str(TMP / "real.h5ad"), "--baseline", str(TMP / "baseline_pred.h5ad"),
       "-o", str(TMP / "bundle"), "--anchor-splits", "2", "--force", *CFG)
    log("real bundle (baseline + replicate anchors) built")
    shutil.copytree(TMP / "baseline", OUT / "baseline", dirs_exist_ok=True)
shutil.copytree(TMP / "bundle", OUT / "bundle", dirs_exist_ok=True)       # reusable: ~47 min to rebuild

# which variants this run scores (each ~12 min); B5like first so it reproduces round 1 exactly (same RNG state)
RUN = ["B5like_T2_dense", "lfc_c1.5_kall", "lfc_c2.0_kall", "lfc_c3.0_kall"]
rows = []
for name in RUN:
    prof, kind = VARIANTS[name]
    path = write_pred(name, prof, kind)
    out = TMP / f"run_{name}"
    ce("run", "-ap", str(path), "-ar", str(TMP / "real.h5ad"), "-o", str(out), *CFG)
    sc = TMP / f"scored_{name}.csv"
    ce("score", "--user-agg", str(out / "agg_results.csv"), "--real-bundle", str(TMP / "bundle"), "-o", str(sc))
    df = pd.read_csv(sc)
    shutil.copy(sc, OUT / sc.name)
    shutil.copy(out / "agg_results.csv", OUT / f"agg_{name}.csv")
    df.insert(0, "variant", name)
    rows.append(df)
    log(name)
    print(df.to_string(index=False), flush=True)
    pd.concat(rows).to_csv(OUT / "localbench_scores.csv", index=False)
    path.unlink()
(OUT / "localbench_meta.json").write_text(json.dumps({"panel": panel, "n_pred": N_PRED, "variants": RUN}))
log("done")
