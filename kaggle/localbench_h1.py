"""
Kaggle job: OFFICIAL-SCORER benchmark on real VCC-2025 H1 cells (VCC_NEXT_MODEL_PLAN.md §2, primary benchmark).

Why H1: same Arc 10x Flex chemistry and depth regime as the challenge contexts, and a regulator-heavy panel; the RPE1
benchmark (kaggle/localbench.py) used essential genes, 100 predicted cells and 2 anchor splits.

Data: gs://arc-institute-virtual-cell-atlas/virtual-cell-challenge/2025/train/adata_Training.h5ad (public, 15.5 GB).
Setup, mirroring the challenge:
  - H1 non-targeting cells split in half (seed 0): 'released' controls (predictions are emitted from these) and
    'held-out' controls (the scorer's reference, appended unchanged to every prediction file)
  - real perturbed groups capped at 400 cells (seed 0); 400 predicted cells per knockdown, drawn WITHOUT replacement
  - cell-eval2 vcc2026 preset: `baseline`, `prep-real-bundle --anchor-splits 5`, `run`, `score --real-bundle`
Predictions: an attached `h1_profiles.npz` {genes, perts, V_<variant>: (n_perts, n_genes) log2 fold changes}, made
locally by models/xline_models.py WITHOUT any H1 response (leave-one-line-out). Emitter fixed: T5 (edit only genes with
nonzero log2 fold; binomial thinning down, Poisson addition up; folds clipped to [1/20, 20]).
Outputs in /kaggle/working: h1_scores.csv (all variants, all metrics, scaled + raw), agg_<variant>.csv, bundle/ (reusable).
"""
import json
import shutil
import subprocess
import sys
import time
from importlib.metadata import version
from pathlib import Path

pins = [f"{p}=={version(p)}" for p in ("numpy", "scipy", "pandas")]
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "anndata", "requests", "uv", *pins], check=True)
subprocess.run(["uv", "tool", "install", "--force", "cell-eval2", "--with", "pdex"], check=True)
CE = str(Path.home() / ".local" / "bin" / "cell-eval2")

import anndata as ad  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import requests  # noqa: E402
import scipy.sparse as sp  # noqa: E402

T0 = time.time()
TMP, OUT = Path("/kaggle/temp/h1bench"), Path("/kaggle/working")
TMP.mkdir(parents=True, exist_ok=True)
URL = "https://storage.googleapis.com/arc-institute-virtual-cell-atlas/virtual-cell-challenge/2025/train/adata_Training.h5ad"
# MODE "full": official-like (400 real cells/knockdown). MODE "reduced": the size the RPE1 bench ran at, for Kaggle's
# 31 GB CPU box (runs 3 and 4 were OOM-killed in `cell-eval2 baseline` at 73k x 18k and 64k x 12k cells x genes).
MODE = "reduced"
N_PRED, ANCHOR_SPLITS = 400, 5
CAP_REAL = 400 if MODE == "full" else 250
CTRL_HELD_CAP = 10000 if MODE == "full" else 8000
MIN_CTRL_CPM = 0.0 if MODE == "full" else 1.0     # reduced: also drop genes below 1 CPM in the released controls
RUN_VARIANTS = ["B_equal_fitted", "b5_dir", "B5like_T2", "eq_plus_mean", "b5dir_plus_mean"]
T2_BETA, PB_N, PB_CELLS = 1.5, 300, 15     # B5's T2 emitter settings (models/transform_candidates.py, pseudobulk_reference)
rng = np.random.default_rng(0)


def log(*a):
    print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)


_v = subprocess.run([CE, "--version"], capture_output=True, text=True)
log("cell-eval2 version:", (_v.stdout + _v.stderr).strip()[:200])
h5 = TMP / "adata_Training.h5ad"
if not h5.exists():
    with requests.get(URL, stream=True, timeout=1800) as r:
        r.raise_for_status()
        with open(h5, "wb") as f:
            for chunk in r.iter_content(1 << 24):
                f.write(chunk)
log(f"H1 training file: {h5.stat().st_size / 1e9:.1f} GB")

a = ad.read_h5ad(h5, backed="r")
log("obs columns:", list(a.obs.columns), "| var columns:", list(a.var.columns)[:6], "| shape", a.shape)
lab = a.obs["target_gene"].astype(str).to_numpy()
genes = (a.var["gene_name"] if "gene_name" in a.var else a.var_names).astype(str).to_numpy()
a.file.close()
prof_npz = next(Path("/kaggle/input").rglob("h1_profiles.npz"))
P = np.load(prof_npz, allow_pickle=False)
p_genes, p_perts = P["genes"].astype(str), P["perts"].astype(str)
variants = {k[2:]: P[k] for k in P.files if k.startswith("V_")}
kinds = {k: "lfc" for k in variants}
abl = next(Path("/kaggle/input").rglob("h1_ablation_profiles.npz"), None)
if abl is not None:
    A2 = np.load(abl, allow_pickle=False)
    assert list(A2["genes"].astype(str)) == list(p_genes) and list(A2["perts"].astype(str)) == list(p_perts)
    kinds.update(json.loads(str(A2["kinds"])))
    variants.update({k[2:]: A2[k] for k in A2.files if k.startswith("V_")})
variants = {k: variants[k] for k in RUN_VARIANTS if k in variants}
log("variants to score:", {k: kinds[k] for k in variants})
# memory: score on the genes of H1's processed gene list (PIE release: expressed, filtered) -- 12,286 instead of 18,080
pset = set(p_genes)
pos = {}
for i, g in enumerate(genes):
    if g in pset:
        pos.setdefault(g, i)
keep_gene = np.array(sorted(pos.values())); G = genes[keep_gene]
log(f"scoring gene axis: {len(G)} of {len(genes)} genes (H1 processed gene list)")
assert len(G) > 10000, f"only {len(G)} H1 genes match the profile gene symbols -- gene identifier mismatch"
try:
    log("machine RAM:", [l for l in open("/proc/meminfo") if l.startswith("MemTotal")][0].strip(), "| mode", MODE)
except OSError:
    pass
panel = [p for p in p_perts if (lab == p).sum() >= 50]
log(f"panel: {len(panel)} of {len(p_perts)} profiled knockdowns have >= 50 real cells; variants: {list(variants)}")

ntc = rng.permutation(np.where(lab == "non-targeting")[0])
ntc_rel, ntc_held = np.sort(ntc[: len(ntc) // 2]), np.sort(ntc[len(ntc) // 2:][:CTRL_HELD_CAP])
real_pert = []
for p in panel:
    idx = np.where(lab == p)[0]
    real_pert.append(np.sort(rng.choice(idx, size=min(CAP_REAL, len(idx)), replace=False)))
real_pert = np.concatenate(real_pert)


def read_rows(rows):
    """Sorted row indices -> CSR (cells x kept genes), read in 5,000-row batches from the backed file."""
    b = ad.read_h5ad(h5, backed="r")
    parts = []
    for i in range(0, len(rows), 5000):
        x = b.X[rows[i:i + 5000]]
        parts.append(sp.csr_matrix(x)[:, keep_gene])
    b.file.close()
    return sp.vstack(parts).tocsr()


X_rel = read_rows(ntc_rel).astype(np.float32)
real_rows = np.sort(np.concatenate([real_pert, ntc_held]))
X_real = read_rows(real_rows).astype(np.float32)
assert np.all(X_real.data == np.round(X_real.data)), "H1 X is not raw integer counts"
if MIN_CTRL_CPM > 0:                                    # reduced mode: drop genes barely expressed in released controls
    _lib = np.maximum(np.asarray(X_rel.sum(1)).ravel(), 1)
    _cpm = np.asarray((sp.diags(1e6 / _lib) @ X_rel).mean(0)).ravel()
    _k = _cpm > MIN_CTRL_CPM
    X_rel, X_real, G = X_rel[:, _k].tocsr(), X_real[:, _k].tocsr(), G[_k]
    log(f"reduced mode: kept {int(_k.sum())} genes with control mean CPM > {MIN_CTRL_CPM}")
real = ad.AnnData(X=X_real, obs=pd.DataFrame({"target_gene": lab[real_rows]}, index=[f"c{i}" for i in real_rows]),
                  var=pd.DataFrame(index=G))
real.write_h5ad(TMP / "real.h5ad")
X_ctrl_held = X_real[np.isin(real_rows, ntc_held)]
lib = np.asarray(X_rel.sum(1)).ravel()
log(f"real reference {real.shape}; released controls {X_rel.shape[0]} (median UMI {np.median(lib):.0f})")
raw_mean = np.asarray(X_rel.mean(0)).ravel(); share = raw_mean / raw_mean.sum()
pb = np.zeros((PB_N, X_rel.shape[1]))
for i in range(PB_N):                                   # sigma for T2: SD of log1p(CP10K) over 15-cell control pseudobulks
    sidx = rng.choice(X_rel.shape[0], size=PB_CELLS, replace=False)
    sm = np.asarray(X_rel[sidx].sum(0)).ravel(); pb[i] = np.log1p(sm / max(sm.sum(), 1) * 1e4)
sigma = np.maximum(pb.std(0), 1e-3)
gpos = {g: i for i, g in enumerate(G)}
cols = np.array([gpos.get(g, -1) for g in p_genes])


def emit(log2fold):
    idx = rng.choice(X_rel.shape[0], N_PRED, replace=False)
    out = np.asarray(X_rel[idx].todense(), dtype=np.int64)
    g = np.flatnonzero(log2fold != 0)
    f = np.clip(2.0 ** log2fold[g], 1 / 20, 20)
    tot = out.sum(1, keepdims=True)
    d, u = g[f < 1], g[f > 1]
    if d.size:
        out[:, d] = rng.binomial(out[:, d], f[f < 1][None, :])
    if u.size:
        out[:, u] += rng.poisson((f[f > 1] - 1)[None, :] * tot * share[u][None, :])
    return out


def emit_t2(z):
    """B5's T2 (models/transform_candidates.t2_log1p_additive): shift every gene's log1p(CP10K) by beta*z*sigma, then
    redraw each cell's own total from a multinomial over all genes."""
    idx = rng.choice(X_rel.shape[0], N_PRED, replace=False)
    dense = np.asarray(X_rel[idx].todense(), dtype=np.float64)
    tot = dense.sum(1, keepdims=True); tot[tot == 0] = 1
    tgt = np.expm1(np.clip(np.log1p(dense / tot * 1e4) + T2_BETA * z[None, :] * sigma[None, :], 0, None))
    tgt = tgt / np.maximum(tgt.sum(1, keepdims=True), 1e-12)
    totals = dense.sum(1).astype(np.int64)
    return np.stack([rng.multinomial(totals[i], tgt[i]) for i in range(N_PRED)])


def write_pred(name, V):
    cells, labs = [], []
    pidx = {p: i for i, p in enumerate(p_perts)}
    for t in panel:
        prof = np.zeros(len(G))
        ok = cols >= 0
        prof[cols[ok]] = np.nan_to_num(V[pidx[t]][ok])
        if t in gpos and kinds.get(name) != "t2_z":
            prof[gpos[t]] = 0.0                       # own gene: excluded by the scorer, left untouched
        if kinds.get(name) == "t2_z":
            cells.append(sp.csr_matrix(emit_t2(prof).astype(np.float32)))   # B5's own target-gene value kept
        else:
            cells.append(sp.csr_matrix(emit(prof).astype(np.float32)))
        labs += [t] * N_PRED
    cells.append(X_ctrl_held); labs += ["non-targeting"] * X_ctrl_held.shape[0]
    pred = ad.AnnData(X=sp.vstack(cells).tocsr(), obs=pd.DataFrame({"target_gene": labs}, index=[f"p{i}" for i in range(len(labs))]),
                      var=pd.DataFrame(index=G))
    path = TMP / f"pred_{name}.h5ad"; pred.write_h5ad(path)
    return path


CFG = ["--preset", "vcc2026", "--pert-col", "target_gene", "--set", "de.backend=pdex"]


def ce(*args):
    r = subprocess.run([CE, *args], capture_output=True, text=True)
    tag = args[0]
    (OUT / f"cell-eval2_{tag}.stdout.txt").write_text(r.stdout)
    (OUT / f"cell-eval2_{tag}.stderr.txt").write_text(r.stderr)
    if r.returncode != 0:
        lines = [x for x in (r.stdout + "\n" + r.stderr).splitlines() if x.strip()]
        errs = [x for x in lines if any(k in x for k in ("Error", "error", "Traceback", "refus", "invalid", "must", "expected"))]
        log(f"cell-eval2 failed: {' '.join(map(str, args[:3]))} (exit {r.returncode})")
        print("--- error-like lines:", *errs[-40:], "--- first 40 lines:", *lines[:40], "--- last 40 lines:", *lines[-40:],
              sep="\n", flush=True)
        raise SystemExit(1)
    return r


prev = next((m.parent for m in Path("/kaggle/input").rglob("h1bench_meta.json")
             if (m.parent / "bundle").exists()), None)
if prev is not None and json.loads((prev / "h1bench_meta.json").read_text())["panel"] == panel:
    shutil.copytree(prev / "bundle", TMP / "bundle", dirs_exist_ok=True); log("reusing bundle from", prev)
else:
    ce("baseline", "-ar", str(TMP / "real.h5ad"), "-o", str(TMP / "baseline"), "--save-pred", str(TMP / "baseline_pred.h5ad"), *CFG)
    log("baseline arm built")
    ce("prep-real-bundle", "--real", str(TMP / "real.h5ad"), "--baseline", str(TMP / "baseline_pred.h5ad"),
       "-o", str(TMP / "bundle"), "--anchor-splits", str(ANCHOR_SPLITS), "--force", *CFG)
    log("real bundle built")
shutil.copytree(TMP / "bundle", OUT / "bundle", dirs_exist_ok=True)
(OUT / "h1bench_meta.json").write_text(json.dumps({"panel": panel, "n_pred": N_PRED, "cap_real": CAP_REAL,
                                                   "anchor_splits": ANCHOR_SPLITS, "ctrl_held_cap": CTRL_HELD_CAP,
                                                   "n_genes_scored": int(len(G)), "variants": kinds}))
rows = []
for name, V in variants.items():
    path = write_pred(name, V)
    out = TMP / f"run_{name}"
    ce("run", "-ap", str(path), "-ar", str(TMP / "real.h5ad"), "-o", str(out), *CFG)
    sc = TMP / f"scored_{name}.csv"
    ce("score", "--user-agg", str(out / "agg_results.csv"), "--real-bundle", str(TMP / "bundle"), "-o", str(sc))
    df = pd.read_csv(sc); df.insert(0, "variant", name); rows.append(df)
    shutil.copy(out / "agg_results.csv", OUT / f"agg_{name}.csv")
    for extra in out.glob("*.csv"):
        if extra.name != "agg_results.csv":
            shutil.copy(extra, OUT / f"{name}__{extra.name}")
    pd.concat(rows).to_csv(OUT / "h1_scores.csv", index=False)
    log(name); print(df.to_string(index=False), flush=True)
    path.unlink()
log("done")
