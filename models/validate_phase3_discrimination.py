"""
Validation check for Phase 3 co-expression predictions (phase3_coexpr.py) -- kept as a separate
script rather than folded into phase3_coexpr.py itself.

phase3_coexpr.py's own LOO validation only reports RAW cosine similarity between predicted and
actual response vectors. That number alone is not trustworthy: gene_transfer_b2.py's own
discrimination_score() docstring flags exactly this trap -- most of a knockdown response vector's
variance is a shared, gene-independent stress/arrest signature, so almost any reasonable predictor
looks deceptively close to the truth in raw cosine terms, even one carrying zero gene-specific
information. This is the same reason the four transfer methods tried before B2 settled were
rejected: not because their raw cosine was bad, but because they failed this harder, rank-based
check.

This script re-runs phase3_coexpr.py's exact co-expression KNN LOO loop (same hyperparameters,
same gene-order source: adata.var_names, NOT gene_names.csv -- see phase3_coexpr.py's header
comment on why gene_names.csv's own header row makes `header=None` unsafe there), and scores it
two ways side-by-side against two baselines:

  1. raw cosine-to-truth      -- what phase3_coexpr.py already reports
  2. discrimination score     -- gene_transfer_b2.py's discrimination_score(): does gene G's OWN
                                  predicted vector rank as the closest match to G's true vector
                                  among all 150 known genes' true vectors, or is it merely
                                  "reasonably close" in absolute terms? 0.5 = chance, 1.0 = perfect.
                                  This is what the official pds_cosine metric actually measures
                                  (report S1.3), not raw closeness.

Baselines compared:
  - "coexpr"      -- phase3_coexpr.py's actual KNN-over-NTC-correlation method
  - "global_mean" -- LOO-honest global mean profile (mean of the OTHER 149 genes' real rows) --
                     the same non-gene-specific fallback B2 already uses for the 287 genes without
                     vcc-perturb overlap, so it's the bar coexpr actually needs to clear to be
                     worth wiring into generate_b2_submission.py
  - "zero"        -- B0-equivalent (predict nothing), included for scale reference only

Usage:
    python models/validate_phase3_discrimination.py
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import anndata as ad

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gene_transfer_b2 import discrimination_score, cosine_row  # reuse the exact, already-trusted scorer

# ── Paths (same defaults as phase3_coexpr.py) ───────────────────────────────
CONTROLS_DIR = Path(os.environ.get("VCC_CONTROLS_DIR", "/mnt/c/Users/User/Downloads/controls"))
VCC_PERTURB  = Path(os.environ.get("VCC_PERTURB_CSV", "/mnt/c/VCC/vcc-perturb.csv"))

# ── Same hyperparameters as phase3_coexpr.py ────────────────────────────────
K_NEIGHBORS  = 5
WEIGHT_EXP   = 2
MIN_CORR     = 0.05
NTC_LABEL    = "non-targeting"

print("=== Phase 3 coexpr validation: raw cosine vs. rank-based discrimination ===\n")

# ── Load NTC cells + genome gene order (adata.var_names, matching phase3_coexpr.py's fix) ──
print("Loading NTC baseline cells ...")
ntc_list   = []
gene_names = None
for ctx in ["A", "B", "C"]:
    adata = ad.read_h5ad(CONTROLS_DIR / f"context_{ctx}.h5ad")
    if gene_names is None:
        gene_names = adata.var_names.to_numpy()
    else:
        assert np.array_equal(adata.var_names.to_numpy(), gene_names), (
            f"context {ctx} has a different gene axis than context A"
        )
    tg_col = next((c for c in ["target_gene", "perturbation", "condition", "gene", "pert"]
                    if c in adata.obs.columns), None)
    ntc = adata[adata.obs[tg_col] == NTC_LABEL] if tg_col else adata
    X = ntc.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    ntc_list.append(X.astype(np.float32))
    print(f"  context {ctx}: {ntc.n_obs} NTC cells  (col={tg_col})")

gene_names = [str(g).strip() for g in gene_names]
gene_idx   = {g: i for i, g in enumerate(gene_names)}
N_GENES    = len(gene_names)
print(f"Genome size: {N_GENES} genes")

ntc_expr = np.concatenate(ntc_list, axis=0)
print(f"Total NTC cells: {ntc_expr.shape[0]}\n")

ntc_c = ntc_expr - ntc_expr.mean(axis=0, keepdims=True)

# ── Load vcc-perturb response matrix (same __index_level_0__ fix as phase3_coexpr.py) ──────
print("Loading vcc-perturb response matrix ...")
vp_raw    = pd.read_csv(VCC_PERTURB)
resp_cols = [c for c in vp_raw.columns if c != "__index_level_0__"]
vp        = vp_raw.groupby(vp_raw["__index_level_0__"].astype(str).str.strip())[resp_cols].mean(numeric_only=True)
known_genes = list(vp.index)
resp_genes  = list(vp.columns)
R_raw       = vp.values.astype(np.float32)

R = np.zeros((len(known_genes), N_GENES), dtype=np.float32)
for j, rg in enumerate(resp_genes):
    if rg in gene_idx:
        R[:, gene_idx[rg]] = R_raw[:, j]
print(f"Known perturbations: {len(known_genes)} genes")

known_in_genome = [(g, i, gene_idx[g]) for i, g in enumerate(known_genes) if g in gene_idx]
K_known    = len(known_in_genome)
known_ridx = [ridx for _, ridx, _ in known_in_genome]
print(f"Known genes found in genome: {K_known}/{len(known_genes)}\n")

X_known   = np.stack([ntc_c[:, gidx] for _, _, gidx in known_in_genome], axis=0)
norms_k   = np.linalg.norm(X_known, axis=1, keepdims=True) + 1e-10
X_known_n = X_known / norms_k

# ── LOO: coexpr KNN predictions (identical logic to phase3_coexpr.py) vs. LOO-honest mean ──
print("Running LOO for coexpr KNN vs. LOO-honest global-mean baseline ...")
coexpr_preds = np.zeros((K_known, N_GENES), dtype=np.float32)
mean_preds   = np.zeros((K_known, N_GENES), dtype=np.float32)

for loo_i, (g, ridx, gidx) in enumerate(known_in_genome):
    others     = [j for j in range(K_known) if j != loo_i]
    other_ridx = [known_ridx[j] for j in others]

    vec   = ntc_c[:, gidx]
    vec_n = vec / (np.linalg.norm(vec) + 1e-10)
    corrs = X_known_n @ vec_n
    corrs_masked = corrs.copy()
    corrs_masked[loo_i] = -999

    top_idx   = np.argsort(corrs_masked)[::-1][:K_NEIGHBORS]
    top_corrs = corrs_masked[top_idx]
    valid     = top_corrs > MIN_CORR

    if valid.sum() == 0:
        coexpr_preds[loo_i] = R[other_ridx].mean(axis=0)
    else:
        top_idx   = top_idx[valid]
        top_corrs = top_corrs[valid]
        weights   = top_corrs ** WEIGHT_EXP
        weights  /= weights.sum()
        ridx_sel  = [known_ridx[ki] for ki in top_idx]
        coexpr_preds[loo_i] = (weights[:, None] * R[ridx_sel]).sum(axis=0)

    # LOO-honest global mean baseline: same non-gene-specific fallback B2 uses for the 287 unknowns
    mean_preds[loo_i] = R[other_ridx].mean(axis=0)

truth      = R[known_ridx]
zero_preds = np.zeros_like(truth)

results = {
    "coexpr":      (coexpr_preds, "Phase 3 co-expression KNN"),
    "global_mean": (mean_preds,   "LOO-honest global mean (B2's 287-gene fallback)"),
    "zero":        (zero_preds,   "Zero / B0-equivalent"),
}

print(f"\n{'Method':<52} {'raw_cosine':>10} {'discrimination':>15}")
print("-" * 80)
scores = {}
for key, (preds, label) in results.items():
    raw_cos = float(cosine_row(preds, truth).mean())
    disc    = discrimination_score(preds, truth)
    scores[key] = disc
    print(f"{label:<52} {raw_cos:>10.4f} {disc:>15.3f}")

print("\nchance discrimination = 0.500  |  1.000 = perfect gene-specific separation")

print("\nVerdict:")
coexpr_disc, mean_disc = scores["coexpr"], scores["global_mean"]
margin = coexpr_disc - mean_disc
if margin > 0.02:
    print(f"  coexpr discrimination ({coexpr_disc:.3f}) clears the global-mean baseline "
          f"({mean_disc:.3f}) by a real margin (+{margin:.3f}) -- there IS gene-specific signal "
          f"here, worth wiring into generate_b2_submission.py for the 287-gene fallback.")
elif margin > 0:
    print(f"  coexpr discrimination ({coexpr_disc:.3f}) edges out the global-mean baseline "
          f"({mean_disc:.3f}) only marginally (+{margin:.3f}) -- same trap as the four "
          f"earlier-rejected transfer methods. Treat with caution before using it for real.")
else:
    print(f"  coexpr discrimination ({coexpr_disc:.3f}) does NOT beat the global-mean baseline "
          f"({mean_disc:.3f}) -- the high raw cosine was mostly the shared common-mode response "
          f"shape, not real gene-specific signal. Same conclusion as the four methods already "
          f"rejected in the B2 report: stick with the flat mean-profile fallback for the 287.")
