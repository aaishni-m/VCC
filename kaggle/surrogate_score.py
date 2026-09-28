"""
Kaggle job: estimate how a candidate submission would score, without submitting, via a CALIBRATED SURROGATE.

The official reference (real perturbed cells for contexts A/B/C) is held out, so this builds synthetic "truth"
perturbations from real context-A control cells and scores predictions against them with re-implementations of
the six vcc2026 metrics (cell-eval2 docs/vcc2026_metrics), including the baseline (mean perturbation response)
and split-half replicate anchors and the (u-b)/(r-b) scaling with the spec's clamps.

Truth model per perturbation i (z-space, turned into counts with the T4 transform on held-apart control cells):
    T_i = a * m  +  A * s_i
    m   = vcc-perturb mean knockdown response (shared component)
    s_i = the top-K_true genes of  rho * u_i + sqrt(1 - rho^2) * w_i,  RMS-normalized
          u_i = our external (K562 + T-cell) direction for gene i, w_i = random direction
(a, A, rho, K_true) are chosen so the surrogate reproduces B5's REAL per-metric leaderboard scores and B0's real
overall (-0.3026); B6 and variants are then scored under the best-fitting settings. This is an estimate, not a
score: it is only as good as that calibration, which the output reports.

Simplifications (stated, not hidden): context A only; a subset of perturbations; a subsample of control cells
as the DE reference group; one replicate split; the MSE numerator's rho-cap taken as 1.
"""
import itertools
import json
import os
import subprocess
import sys
import time
from importlib.metadata import version
from pathlib import Path

pins = [f"{p}=={version(p)}" for p in ("numpy", "scipy", "pandas")]
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "anndata", *pins], check=True)

import anndata as ad  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from multiprocessing import Pool  # noqa: E402
from scipy.stats import mannwhitneyu  # noqa: E402

import shutil  # noqa: E402

code_dir = Path("/kaggle/temp/vcc")                      # /kaggle/input is read-only; models/ gets written to
shutil.copytree(next(Path("/kaggle/input").rglob("config.py")).parent, code_dir, dirs_exist_ok=True)
controls = next(Path("/kaggle/input").rglob("context_A.h5ad")).parent
os.environ["VCC_CONTROLS_DIR"] = str(controls)
os.environ["VCC_PERTURB_CSV"] = str(next(Path("/kaggle/input").rglob("vcc-perturb.csv")))
subprocess.run([sys.executable, "models/gene_transfer_b5.py"], check=True, cwd=code_dir)
sys.path.insert(0, str(code_dir))

import config as cfg  # noqa: E402
from models.gene_transfer_b2 import load_response_matrix  # noqa: E402
from models.gene_transfer_b6 import sparsify  # noqa: E402
from models.generate_b2_submission import BETA_REAL, align_z_matrix_to_context  # noqa: E402
from models.pseudobulk_reference import load_or_build  # noqa: E402
from models.transform_candidates import t2_log1p_additive, t4_sparse_thinning  # noqa: E402

T0 = time.time()
N_PERTS = 40
N_CTRL_DE = 2500
CELLS = 400
TS_PDS = 5e4
rng = np.random.default_rng(7)


def log(*a):
    print(f"[{time.time() - T0:7.0f}s]", *a, flush=True)


# ---------------- data ----------------
A = ad.read_h5ad(cfg.CONTEXT_FILES["A"])
X = A.X.tocsr().astype(np.float32)
genes = A.var_names.to_numpy()
G = len(genes)
perm = rng.permutation(X.shape[0])
pool_gen, pool_ctrl = perm[: X.shape[0] // 2], perm[X.shape[0] // 2:]   # generation cells vs held-out "measured control"
ref = load_or_build("A")

vp_targets, vp_genes, R = load_response_matrix()
tg = pd.read_csv(cfg.PERT_COUNTS_CSV)[cfg.PERT_COL].str.strip().tolist()
meta5 = json.loads((code_dir / "models" / "b5_meta.json").read_text())
Z5 = align_z_matrix_to_context(np.load(code_dir / "models" / "b5_profiles.npy").astype(np.float64), vp_genes, genes)
m_vec = align_z_matrix_to_context(R.mean(0, keepdims=True), vp_genes, genes)[0]
npz = np.load(next(Path("/kaggle/input").rglob("external_profiles.npz")))
from models.gene_transfer_b5 import combined_direction  # noqa: E402
gpos = {g: i for i, g in enumerate(vp_genes)}
U = align_z_matrix_to_context(combined_direction(npz, "D_", np.array([gpos.get(g, -1) for g in tg])), vp_genes, genes)

cands = [i for i, s in enumerate(meta5["source"]) if s == "mean+external"]
P_IDX = sorted(rng.choice(cands, size=N_PERTS, replace=False).tolist())
target_pos = np.array([np.where(genes == g)[0][0] for g in tg if g in set(genes)])
self_pos = {i: (np.where(genes == tg[i])[0][0] if tg[i] in set(genes) else -1) for i in P_IDX}
log(f"context A: {X.shape}, perturbations scored: {N_PERTS}, DE control cells: {N_CTRL_DE}")

# prediction z-profiles (context gene space)
PRED_Z = {"B5": Z5, "B6_k100": sparsify(Z5, 100), "B6_k200": sparsify(Z5, 200), "B6_k400": sparsify(Z5, 400)}


def gen_cells(z, transform, pool, n=CELLS):
    idx = rng.choice(pool, size=n, replace=True)
    if transform is None:
        return X[idx].toarray().astype(np.float32)
    return transform(X[idx], z, ref, rng, beta=BETA_REAL).astype(np.float32)


# ---------------- metric machinery ----------------
def profile(cells):
    P = cells.sum(0, dtype=np.float64)
    return np.log1p(TS_PDS * P / P.sum()), P


def jackknife_C(cells):
    """Delete-one jackknife correction of the spec's expression-error section (dense, n x G)."""
    P = cells.sum(0, dtype=np.float64)
    S = P.sum()
    ell = cells.sum(1, dtype=np.float64)
    n = cells.shape[0]
    tot = 0.0
    for s in range(0, G, 4000):
        v = np.log1p(TS_PDS * (P[s:s + 4000] - cells[:, s:s + 4000]) / (S - ell)[:, None])
        tot += ((v - v.mean(0)) ** 2).sum()
    return (n - 1) / n * tot


ctrl_all = X[pool_ctrl]
ctrl_sum = np.asarray(ctrl_all.sum(0)).ravel().astype(np.float64)
b_ctrl = np.log1p(TS_PDS * ctrl_sum / ctrl_sum.sum())
C_ctrl = 0.0   # the ~9,200-cell control group's jackknife term is ~1/20 of a 400-cell group's; omitted

ctrl_de_idx = rng.choice(pool_ctrl, N_CTRL_DE, replace=False)
ctrl_de = X[ctrl_de_idx].toarray().astype(np.float32)
ctrl_cpm = ctrl_de / ctrl_de.sum(1, keepdims=True) * 1e6
TESTED = np.where(ctrl_cpm.mean(0) > 5)[0]
CTRL_T = ctrl_cpm[:, TESTED]
CTRL_MEAN = CTRL_T.mean(0)
log(f"tested genes (>5 CPM in control): {len(TESTED)}")


def de_table(cells, ctrl_t=None, ctrl_mean=None):
    ctrl_t = CTRL_T if ctrl_t is None else ctrl_t
    ctrl_mean = CTRL_MEAN if ctrl_mean is None else ctrl_mean
    cpm = (cells / np.maximum(cells.sum(1, keepdims=True), 1) * 1e6)[:, TESTED]
    p = mannwhitneyu(cpm, ctrl_t, axis=0, method="asymptotic").pvalue
    p = np.nan_to_num(p, nan=1.0)
    o = np.argsort(p)
    m = len(p)
    adj = np.minimum.accumulate((p[o] * m / np.arange(1, m + 1))[::-1])[::-1]
    padj = np.empty(m)
    padj[o] = np.minimum(adj, 1)
    lfc = np.log2((cpm.mean(0) + 1e-9) / (ctrl_mean + 1e-9))
    return p, padj, lfc


def _de_worker(args):
    return de_table(*args)


def de_many(cell_list, ctrl_t=None, ctrl_mean=None):
    with Pool(4) as pool:
        return pool.map(_de_worker, [(c, ctrl_t, ctrl_mean) for c in cell_list])


def pds(pred_prof, ref_prof, ctrl_pred, ctrl_ref):
    keep = np.setdiff1d(np.arange(G), target_pos)
    dp = np.stack([p - ctrl_pred for p in pred_prof])[:, keep]
    dq = np.stack([q - ctrl_ref for q in ref_prof])[:, keep]
    nrm_p, nrm_q = np.linalg.norm(dp, axis=1), np.linalg.norm(dq, axis=1)
    cos = (dp @ dq.T) / np.maximum(np.outer(nrm_p, nrm_q), 1e-300)
    d = 1 - cos
    d[nrm_p == 0, :] = 1
    d[:, nrm_q == 0] = 1
    n = len(pred_prof)
    out = []
    for i in range(n):
        k = (d[i] < d[i, i]).sum() + 0.5 * ((d[i] == d[i, i]).sum() - 1)
        out.append(1 - k / (n - 1))
    return float(np.mean(out))


_JK = {}


def jk(cells):
    key = id(cells)
    if key not in _JK:
        _JK[key] = (cells, jackknife_C(cells))   # keep a reference so the id stays unique
    return _JK[key][1]


def mse_metric(pred_cells, ref_cells, pred_prof, ref_prof, ref_ctrl_prof):
    num = den = 0.0
    for i in range(len(ref_cells)):
        keep = np.ones(G, bool)
        if self_pos[P_IDX[i]] >= 0:
            keep[self_pos[P_IDX[i]]] = False
        C_hat, C = jk(pred_cells[i]), jackknife_C(ref_cells[i])
        Gp = keep.sum()
        num += (((pred_prof[i] - ref_prof[i])[keep] ** 2).sum() - min(C_hat, C) - C) / Gp
        den += (((ref_prof[i] - ref_ctrl_prof)[keep] ** 2).sum() - C - C_ctrl) / Gp
    return num / den


def de_metrics(pred_de, ref_de):
    fid, reach, jac, nmae = [], [], [], []
    tested_genes = genes[TESTED]
    for i, ((pp, pa, pl), (rp, ra, rl)) in enumerate(zip(pred_de, ref_de)):
        excl = tested_genes == tg[P_IDX[i]]
        rsig = (ra < 0.05) & ~excl
        psig = (pa < 0.05) & ~excl
        adjud = np.isfinite(rl) & (rl != 0)
        n_real = rsig.sum()
        calls = psig & adjud
        k = (np.sign(pl[calls]) == np.sign(rl[calls])).sum()
        if n_real or calls.sum():
            fid.append(k / max(calls.sum(), n_real) if n_real else k / max(calls.sum(), 1))
        union = (rsig | psig).sum()
        jac.append(1.0 if union == 0 else (rsig & psig).sum() / union)
        if n_real:
            pool = np.where(rsig)[0]
            order = pool[np.lexsort((-np.abs(pl[pool]), pp[pool], pa[pool], ~psig[pool]))]
            order = order[adjud[order]]
            ok = np.sign(pl[order]) == np.sign(rl[order])
            purity = np.cumsum(ok) / np.arange(1, len(ok) + 1)
            good = np.where(purity >= 0.9)[0]
            reach.append((good.max() + 1 if len(good) else 0) / n_real)
            gate = rsig & np.isfinite(rl)
            if gate.sum() >= 10 and np.abs(rl[gate]).sum() > 0:
                nmae.append(np.abs(np.nan_to_num(pl[gate]) - rl[gate]).sum() / np.abs(rl[gate]).sum())
    return {"fid": np.mean(fid), "reach": np.mean(reach), "jac": np.mean(jac), "nmae": np.mean(nmae) if nmae else 1.0}


def raw_scores(pred_cells, pred_de, ref_cells, ref_prof, ref_de, ctrl_pred_prof=None, ctrl_ref_prof=None):
    ctrl_pred_prof = b_ctrl if ctrl_pred_prof is None else ctrl_pred_prof
    ctrl_ref_prof = b_ctrl if ctrl_ref_prof is None else ctrl_ref_prof
    pred_prof = [profile(c)[0] for c in pred_cells]
    out = {"pds": pds(pred_prof, ref_prof, ctrl_pred_prof, ctrl_ref_prof),
           "mse": mse_metric(pred_cells, ref_cells, pred_prof, ref_prof, ctrl_ref_prof)}
    out.update(de_metrics(pred_de, ref_de))
    return out


LOWER_BETTER = {"mse", "nmae"}


def scale(u, b, r, k):
    s = (u - b) / (r - b) if r != b else 0.0
    if k == "mse":
        s = min(max(s, 0.0), 1.0)
    if k == "nmae":
        s = max(s, -6.0)
    return s


# ---------------- predictions (fixed across truth settings) ----------------
log("generating predictions and their DE tables ...")
PRED_CELLS = {"null": [gen_cells(None, None, pool_gen) for _ in P_IDX]}
PRED_CELLS["B5"] = [gen_cells(PRED_Z["B5"][i], t2_log1p_additive, pool_gen) for i in P_IDX]
for k in ("B6_k100", "B6_k200", "B6_k400"):
    PRED_CELLS[k] = [gen_cells(PRED_Z[k][i], t4_sparse_thinning, pool_gen) for i in P_IDX]
PRED_DE = {name: de_many(cells) for name, cells in PRED_CELLS.items()}
for name, des in PRED_DE.items():
    log(f"  {name:8s} median calls/perturbation: {np.median([(d[1] < 0.05).sum() for d in des]):.0f}")

REAL_B5 = {"pds": 0.4867, "mse": 0.0, "jac": -0.0096, "nmae": 0.0838, "fid": -0.0032, "reach": 0.2122}
REAL_B5_OVERALL, REAL_B0_OVERALL = 0.1283, -0.3026


def truth_z(i, a, A_amp, rho, k_true, seed):
    r = np.random.default_rng(seed + i)
    u = U[i].copy()
    u[target_pos] = 0
    u /= np.linalg.norm(u) + 1e-12
    w = r.standard_normal(G)
    w[target_pos] = 0
    w /= np.linalg.norm(w)
    d = rho * u + np.sqrt(1 - rho ** 2) * w
    top = np.argsort(-np.abs(d))[:k_true]
    s = np.zeros(G)
    s[top] = d[top] / np.sqrt(np.mean(d[top] ** 2))
    z = a * m_vec + A_amp * s
    if self_pos[i] >= 0:
        z[self_pos[i]] = -3.14                      # the knockdown itself (excluded from every metric anyway)
    return z


def evaluate(a, A_amp, rho, k_true):
    ref_cells = [gen_cells(truth_z(i, a, A_amp, rho, k_true, 11), t4_sparse_thinning, pool_gen) for i in P_IDX]
    ref_prof = [profile(c)[0] for c in ref_cells]
    ref_de = de_many(ref_cells)
    # baseline: the context's mean perturbation response, identical for every perturbation
    pooled = np.concatenate(ref_cells)
    base_cells = [pooled[rng.choice(len(pooled), CELLS, replace=False)] for _ in P_IDX]
    base = raw_scores(base_cells, de_many(base_cells), ref_cells, ref_prof, ref_de)
    # replicate: split-half, each half against its own control half
    half = [rng.permutation(CELLS) for _ in P_IDX]
    h1 = [c[h[: CELLS // 2]] for c, h in zip(ref_cells, half)]
    h2 = [c[h[CELLS // 2:]] for c, h in zip(ref_cells, half)]
    c1, c2 = CTRL_T[: N_CTRL_DE // 2], CTRL_T[N_CTRL_DE // 2:]
    cp = []
    for c in (X[pool_ctrl[: len(pool_ctrl) // 2]], X[pool_ctrl[len(pool_ctrl) // 2:]]):
        tot = np.asarray(c.sum(0)).ravel().astype(np.float64)
        cp.append(np.log1p(TS_PDS * tot / tot.sum()))
    rep = raw_scores(h1, de_many(h1, c1, c1.mean(0)), h2, [profile(c)[0] for c in h2],
                     de_many(h2, c2, c2.mean(0)), cp[0], cp[1])
    results = {}
    for name in PRED_CELLS:
        u = raw_scores(PRED_CELLS[name], PRED_DE[name], ref_cells, ref_prof, ref_de)
        sc = {k: scale(u[k], base[k], rep[k], k) for k in u}
        sc["overall"] = float(np.mean(list(sc.values())))
        results[name] = sc
    fit = sum((results["B5"][k] - v) ** 2 for k, v in REAL_B5.items()) + (results["null"]["overall"] - REAL_B0_OVERALL) ** 2
    return results, fit, base, rep


GRID = list(itertools.product([0.5, 1.0], [1.5, 3.0], [0.15, 0.3], [150, 500]))
rows = []
for a, A_amp, rho, k_true in GRID:
    res, fit, base, rep = evaluate(a, A_amp, rho, k_true)
    rows.append({"a": a, "A": A_amp, "rho": rho, "k_true": k_true, "fit_err": fit,
                 **{f"{n}_{k}": v for n, sc in res.items() for k, v in sc.items()}})
    log(f"a={a} A={A_amp} rho={rho} K={k_true}: fit_err={fit:.4f} | B5 sim overall {res['B5']['overall']:+.3f} "
        f"(real {REAL_B5_OVERALL:+.3f}), null {res['null']['overall']:+.3f} (real {REAL_B0_OVERALL:+.3f}) | "
        f"B6_k100 {res['B6_k100']['overall']:+.3f}  B6_k200 {res['B6_k200']['overall']:+.3f}  "
        f"B6_k400 {res['B6_k400']['overall']:+.3f}")
    pd.DataFrame(rows).to_csv("/kaggle/working/surrogate_grid.csv", index=False)

df = pd.DataFrame(rows).sort_values("fit_err")
log("best-fitting truth settings (per-metric scaled scores):")
metric_cols = ["pds", "mse", "jac", "nmae", "fid", "reach", "overall"]
for _, r in df.head(3).iterrows():
    log(f"  a={r.a} A={r.A} rho={r.rho} K={int(r.k_true)} fit_err={r.fit_err:.4f}")
    for name in PRED_CELLS:
        log(f"     {name:8s} " + "  ".join(f"{k}={r[f'{name}_{k}']:+.3f}" for k in metric_cols))
log(f"real B5 per metric: {REAL_B5} overall {REAL_B5_OVERALL}")
log("done")
