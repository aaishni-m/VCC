"""
Phase 3 — Co-expression regression for 287 unknown VCC target genes.

For each unknown VCC target gene:
  1. Compute Pearson correlation with each of the ~150 vcc-perturb known genes
     using NTC (non-targeting control) cells from context_A/B/C baselines
  2. Take top-k most correlated known genes
  3. Weighted-average their z-score profiles → predicted perturbation response
  4. Save predicted profiles for all 300 VCC targets

Also runs LOO validation on known genes to confirm signal > B0.

Outputs (written relative to /mnt/c/VCC/):
  models/phase3_coexpr_profiles.npy   — shape (300, 18533), float32 z-scores
  models/phase3_coexpr_meta.json      — gene lists, validation scores, settings
"""
import os, sys, json
from pathlib import Path

import numpy as np
import pandas as pd
import anndata as ad

# ── Paths ──────────────────────────────────────────────────────────────────
CONTROLS_DIR = Path(os.environ.get("VCC_CONTROLS_DIR",
                    "/mnt/c/Users/User/Downloads/controls"))
VCC_PERTURB  = Path(os.environ.get("VCC_PERTURB_CSV",
                    "/mnt/c/VCC/vcc-perturb.csv"))
OUT_DIR      = Path("models")
OUT_DIR.mkdir(exist_ok=True)
OUT_PROFILES = OUT_DIR / "phase3_coexpr_profiles.npy"
OUT_META     = OUT_DIR / "phase3_coexpr_meta.json"

# ── Hyperparameters ─────────────────────────────────────────────────────────
K_NEIGHBORS  = 5      # top-k vcc-perturb genes to blend
WEIGHT_EXP   = 2      # weight = corr^WEIGHT_EXP  (penalises weak matches)
MIN_CORR     = 0.05   # discard any neighbour below this Pearson r
BETA_PHASE3  = 0.6    # conservative beta for co-expr predictions
NTC_LABEL    = "non-targeting"   # value in target_gene column for control cells

# ── Load reference data ────────────────────────────────────────────────────
print("=== Phase 3: Co-expression regression ===\n")

# 300 VCC target genes (ordered as in submission)
pert_df    = pd.read_csv(CONTROLS_DIR / "pert_counts.csv")
vcc_targets = list(pert_df["target_gene"].str.strip())
print(f"VCC target genes:  {len(vcc_targets)}")

# 18533 genome genes (ordered)
gene_names_df = pd.read_csv(CONTROLS_DIR / "gene_names.csv", header=None)
gene_names    = list(gene_names_df.iloc[:, 0].str.strip())
gene_idx      = {g: i for i, g in enumerate(gene_names)}
N_GENES       = len(gene_names)
print(f"Genome size:       {N_GENES} genes")

# vcc-perturb response matrix
print("\nLoading vcc-perturb response matrix ...")
vp = pd.read_csv(VCC_PERTURB, index_col=0)
if vp.index.duplicated().any():
    vp = vp.groupby(vp.index).mean()
known_genes = list(vp.index)
resp_genes  = list(vp.columns)
R_raw       = vp.values.astype(np.float32)   # (n_known, n_resp)
print(f"Known perturbations: {len(known_genes)} genes  ×  {len(resp_genes)} response genes")

# Map vcc-perturb z-scores into the 18533-gene genome space
R = np.zeros((len(known_genes), N_GENES), dtype=np.float32)
mapped = sum(1 for rg in resp_genes if rg in gene_idx)
for j, rg in enumerate(resp_genes):
    if rg in gene_idx:
        R[:, gene_idx[rg]] = R_raw[:, j]
print(f"Mapped {mapped}/{len(resp_genes)} response genes into the genome space")

known_set = {g: i for i, g in enumerate(known_genes)}

# ── Load NTC baseline cells ────────────────────────────────────────────────
print("\nLoading NTC baseline cells ...")
ntc_list = []
for ctx in ["A", "B", "C"]:
    path  = CONTROLS_DIR / f"context_{ctx}.h5ad"
    adata = ad.read_h5ad(path)
    # Find the target_gene / perturbation column
    tg_col = None
    for col in ["target_gene", "perturbation", "condition", "gene", "pert"]:
        if col in adata.obs.columns:
            tg_col = col; break
    if tg_col is None:
        print(f"  WARNING: context {ctx} has no recognised perturbation column; "
              f"using ALL cells as NTC")
        ntc = adata
    else:
        ntc = adata[adata.obs[tg_col] == NTC_LABEL]
    X = ntc.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    ntc_list.append(X.astype(np.float32))
    print(f"  context {ctx}: {ntc.n_obs} NTC cells  (col={tg_col})")

ntc_expr = np.concatenate(ntc_list, axis=0)   # (total_ntc, 18533)
print(f"Total NTC cells: {ntc_expr.shape[0]}")

# Mean-centre each gene (required for Pearson correlation via dot-product)
ntc_c   = ntc_expr - ntc_expr.mean(axis=0, keepdims=True)
norms_c = np.linalg.norm(ntc_c, axis=0, keepdims=True)   # (1, N_GENES)

# ── Find which known vcc-perturb genes are in the genome ──────────────────
known_in_genome = [(g, i, gene_idx[g])
                   for i, g in enumerate(known_genes) if g in gene_idx]
print(f"\nKnown vcc-perturb genes found in genome: {len(known_in_genome)}/{len(known_genes)}")
if len(known_in_genome) < 10:
    print("  WARNING: very few known genes found in genome — co-expression signal may be weak")

# Expression matrix for known genes: shape (K_known, N_cells)
# Normalised for fast cosine-based Pearson computation
K_known     = len(known_in_genome)
known_ridx  = [ridx for _, ridx, _ in known_in_genome]   # index into R

X_known     = np.stack(
    [ntc_c[:, gidx] for _, _, gidx in known_in_genome], axis=0
)   # (K_known, N_cells)
norms_k     = np.linalg.norm(X_known, axis=1, keepdims=True) + 1e-10
X_known_n   = X_known / norms_k    # (K_known, N_cells), unit-normed rows

def coexpr_predict(gene_expr_vec):
    """Given a centred expression vector for a target gene, return predicted
    z-score profile (N_GENES,) using KNN over vcc-perturb known genes."""
    norm_v  = np.linalg.norm(gene_expr_vec) + 1e-10
    vec_n   = gene_expr_vec / norm_v           # unit vector (N_cells,)
    corrs   = X_known_n @ vec_n               # (K_known,) Pearson r values

    top_idx   = np.argsort(corrs)[::-1][:K_NEIGHBORS]
    top_corrs = corrs[top_idx]
    valid     = top_corrs > MIN_CORR

    if valid.sum() == 0:
        # No sufficiently correlated known gene — fall back to global mean
        return R[known_ridx].mean(axis=0)

    top_idx   = top_idx[valid]
    top_corrs = top_corrs[valid]
    weights   = top_corrs ** WEIGHT_EXP
    weights  /= weights.sum()
    ridx_sel  = [known_ridx[ki] for ki in top_idx]
    return (weights[:, None] * R[ridx_sel]).sum(axis=0)

# ── LOO validation on known genes ─────────────────────────────────────────
print("\nRunning LOO validation on known genes ...")
loo_cosines = []
for loo_i, (g, ridx, gidx) in enumerate(known_in_genome):
    # Temporarily mask self from X_known_n
    vec          = ntc_c[:, gidx]
    norm_v       = np.linalg.norm(vec) + 1e-10
    vec_n        = vec / norm_v

    corrs        = X_known_n @ vec_n
    corrs_masked = corrs.copy()
    corrs_masked[loo_i] = -999   # exclude self

    top_idx   = np.argsort(corrs_masked)[::-1][:K_NEIGHBORS]
    top_corrs = corrs_masked[top_idx]
    valid     = top_corrs > MIN_CORR

    if valid.sum() == 0:
        pred = R[known_ridx].mean(axis=0)
    else:
        top_idx   = top_idx[valid]
        top_corrs = top_corrs[valid]
        weights   = top_corrs ** WEIGHT_EXP
        weights  /= weights.sum()
        ridx_sel  = [known_ridx[ki] for ki in top_idx]
        pred      = (weights[:, None] * R[ridx_sel]).sum(axis=0)

    actual  = R[ridx]
    cos_sim = float(np.dot(pred, actual) /
                    (np.linalg.norm(pred) * np.linalg.norm(actual) + 1e-10))
    loo_cosines.append(cos_sim)

loo_mean = float(np.mean(loo_cosines))
loo_std  = float(np.std(loo_cosines))
print(f"LOO cosine similarity:  mean={loo_mean:.4f}  std={loo_std:.4f}  "
      f"min={min(loo_cosines):.4f}  max={max(loo_cosines):.4f}")

if loo_mean > 0.05:
    signal = "GOOD — co-expression approach beats B0"
elif loo_mean > 0.01:
    signal = "MARGINAL — apply with very low beta (0.2-0.3)"
else:
    signal = "WEAK — may not be worth applying"
print(f"Signal quality: {signal}")

# ── Build predicted profiles for all 300 VCC targets ──────────────────────
print("\nBuilding predicted profiles for all 300 VCC targets ...")
predicted_profiles = np.zeros((len(vcc_targets), N_GENES), dtype=np.float32)
per_gene_beta      = np.zeros(len(vcc_targets), dtype=np.float32)
gene_source        = []

real_count, coexpr_count, b0_count = 0, 0, 0

for ti, gene in enumerate(vcc_targets):
    if gene in known_set:
        predicted_profiles[ti] = R[known_set[gene]]
        per_gene_beta[ti]      = 1.5    # BETA_REAL — real measured profile
        gene_source.append("real")
        real_count += 1
    elif gene in gene_idx:
        gidx = gene_idx[gene]
        predicted_profiles[ti] = coexpr_predict(ntc_c[:, gidx])
        per_gene_beta[ti]      = BETA_PHASE3
        gene_source.append("coexpr")
        coexpr_count += 1
    else:
        # Gene not in genome → pure B0
        per_gene_beta[ti] = 0.0
        gene_source.append("b0")
        b0_count += 1

print(f"  Real (vcc-perturb overlap):    {real_count}")
print(f"  Co-expression predicted:       {coexpr_count}")
print(f"  B0 fallback (not in genome):   {b0_count}")

# ── Save ───────────────────────────────────────────────────────────────────
np.save(OUT_PROFILES, predicted_profiles)
np.save(OUT_DIR / "phase3_coexpr_betas.npy", per_gene_beta)

meta = {
    "vcc_targets":        vcc_targets,
    "gene_source":        gene_source,    # "real" | "coexpr" | "b0"
    "real_genes":         [g for g, s in zip(vcc_targets, gene_source) if s == "real"],
    "coexpr_genes":       [g for g, s in zip(vcc_targets, gene_source) if s == "coexpr"],
    "b0_genes":           [g for g, s in zip(vcc_targets, gene_source) if s == "b0"],
    "loo_cosine_mean":    loo_mean,
    "loo_cosine_std":     loo_std,
    "loo_cosines":        loo_cosines,
    "k_neighbors":        K_NEIGHBORS,
    "weight_exp":         WEIGHT_EXP,
    "min_corr":           MIN_CORR,
    "beta_real":          1.5,
    "beta_phase3":        BETA_PHASE3,
    "n_ntc_cells":        int(ntc_expr.shape[0]),
    "n_known_in_genome":  K_known,
}
with open(OUT_META, "w") as f:
    json.dump(meta, f, indent=2)

print(f"\nSaved: {OUT_PROFILES}  shape={predicted_profiles.shape}")
print(f"Saved: {OUT_DIR / 'phase3_coexpr_betas.npy'}")
print(f"Saved: {OUT_META}")
print(f"\nLOO mean cosine: {loo_mean:.4f}  →  {signal}")
print(f"Recommended BETA_PHASE3: {BETA_PHASE3}")
print("\nDone. Next: update generate_b2_submission.py to load these profiles.")
