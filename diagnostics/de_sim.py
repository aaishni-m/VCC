"""Predicted-side DE diagnostic: how many genes does a submission's synthetic cells get called significant?
Mirrors the official vcc2026 DE procedure (CPM 1e6 per cell, >5 CPM in control, two-sided Wilcoxon normal
approx with tie correction, BH per perturbation, alpha 0.05, target gene excluded)."""
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
import numpy as np, pandas as pd, anndata as ad, scipy.sparse as sp
from scipy.stats import norm
import config as cfg
from models.gene_transfer_b2 import load_response_matrix
from models.generate_b2_submission import load_saved_profiles, align_z_matrix_to_context, BETA_REAL
from models.pseudobulk_reference import load_or_build
from models.transform_candidates import t2_log1p_additive, t4_sparse_thinning

N_PERTS = int(sys.argv[1]) if len(sys.argv) > 1 else 8
rng = np.random.default_rng(1)
a = ad.read_h5ad(cfg.CONTEXT_FILES["A"]); X = a.X.tocsr().astype(np.float32); genes = a.var_names.to_numpy()
tot = np.asarray(X.sum(1)).ravel(); cpm_ctrl = sp.diags(1e6 / tot) @ X
keep = np.where(np.asarray(cpm_ctrl.mean(0)).ravel() > 5)[0]
C = cpm_ctrl[:, keep].toarray()                                  # (18400, G_keep) dense
Cs = np.sort(C, axis=0)
print(f"genes passing 5-CPM control filter: {len(keep)}", flush=True)

_, gene_cols, _ = load_response_matrix()
tg = pd.read_csv(cfg.PERT_COUNTS_CSV)[cfg.PERT_COL].tolist()
ref = load_or_build("A")
Z5, _ = load_saved_profiles("b5", tg); Z5 = align_z_matrix_to_context(Z5, gene_cols, genes)
Z6, _ = load_saved_profiles("b6", tg); Z6 = align_z_matrix_to_context(Z6, gene_cols, genes)
meta_real = set(np.where(pd.read_json(cfg.PROJECT_ROOT / "models" / "b5_meta.json", typ="series")["is_real"])[0])
pick = [i for i in rng.permutation(len(tg)) if i not in meta_real][:N_PERTS]


def wilcoxon_bh(P):
    """P: (400, G_keep) CPM of the perturbation's cells. Returns adj p-values and lfc (log2)."""
    n1, n2 = P.shape[0], C.shape[0]
    pv = np.ones(P.shape[1]); lfc = np.zeros(P.shape[1])
    for j in range(P.shape[1]):
        x = P[:, j]; c = Cs[:, j]
        U = (np.searchsorted(c, x, "left") + 0.5 * (np.searchsorted(c, x, "right") - np.searchsorted(c, x, "left"))).sum()
        _, cnt = np.unique(np.concatenate([x, c]), return_counts=True)
        N = n1 + n2
        sigma = np.sqrt(n1 * n2 / 12 * ((N + 1) - (cnt ** 3 - cnt).sum() / (N * (N - 1))))
        z = (U - n1 * n2 / 2) / sigma if sigma > 0 else 0.0
        pv[j] = 2 * norm.sf(abs(z))
        lfc[j] = np.log2((x.mean() + 1e-9) / (c.mean() + 1e-9))
    o = np.argsort(pv); m = len(pv)
    adj = np.minimum.accumulate((pv[o] * m / np.arange(1, m + 1))[::-1])[::-1]
    out = np.empty(m); out[o] = np.minimum(adj, 1)
    return out, lfc


VARIANTS = sys.argv[2].split(",") if len(sys.argv) > 2 else ["null", "B5"]
rows = []
for variant in VARIANTS:
    for i in pick:
        idx = rng.choice(X.shape[0], cfg.CELLS_PER_PERT, replace=True)
        if variant in ("b6", "t4null"):
            syn = t4_sparse_thinning(X[idx], Z6[i] if variant == "b6" else np.zeros(len(genes)), ref, rng, beta=BETA_REAL).astype(np.float32)
        elif variant == "raw":                          # untouched resampled control cells (checks the test itself)
            syn = X[idx].toarray()
        else:
            z = np.zeros(len(genes)) if variant == "null" else Z5[i]
            syn = t2_log1p_additive(X[idx], z, ref, rng, beta=BETA_REAL).astype(np.float32)
        P = (syn / syn.sum(1, keepdims=True) * 1e6)[:, keep]
        adj, lfc = wilcoxon_bh(P)
        sig = adj < 0.05
        tpos = np.where(genes[keep] == tg[i])[0]
        sig[tpos] = False
        zz = (Z6 if variant == "b6" else Z5)[i][keep]
        if variant == "b6":
            chosen = zz != 0
            print(f"   b6 {tg[i]}: shifted {chosen.sum()} tested genes; called {sig.sum()}, of which {(sig & chosen).sum()} are shifted genes; "
                  f"called sign matches predicted sign: {np.mean(np.sign(lfc[sig & chosen]) == np.sign(zz[sig & chosen])):.2f}", flush=True)
        rows.append(dict(variant=variant, gene=tg[i], n_sig=int(sig.sum()),
                         n_up=int((sig & (lfc > 0)).sum()),
                         median_abs_z_of_called=float(np.median(np.abs(zz[sig]))) if sig.any() else np.nan))
        print(rows[-1], flush=True)
df = pd.DataFrame(rows)
print(df.groupby("variant").n_sig.describe())
