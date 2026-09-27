"""
B3 -- next iteration of B2, built only from locally-validated pieces.

Context: B2's actual generated submission (see generate_b2_submission.py, BETA_MEAN=0.0) applies a
REAL z-profile only for the 13/300 genes vcc-perturb directly measured; the other 287/300 get
beta=0 -- pure B0 (no shift at all). Applying ANY nonzero beta with the flat population-mean
z-profile was tested and REJECTED (raw MSE ~3.6 vs null's ~1.0) -- worse than doing nothing,
because that shape is wrong for any individual held-out gene even though it IS real aggregate
structure in the data.

Phase 3 (phase3_coexpr.py, models/validate_phase3_discrimination.py) found the co-expression KNN
profile has a real, if weak, discrimination edge over both the flat mean and chance
(discrimination 0.517 vs mean's 0.492 vs chance's 0.500). What was never checked: whether coexpr,
at SOME beta, actually beats null on raw MSE the way the flat mean failed to. That's what this
script settles, by extending the exact test that already rejected the flat mean -- a beta sweep on
raw MSE-to-truth, run in the same leave-one-out harness over the 150 known genes -- gated by a
discrimination margin sized to the actual sampling noise at n=150, not a hand-picked threshold.

NOT included here despite being planned: a separate per-context "responsiveness scaling" of beta by
each context's own baseline gene variability. Re-reading transform_candidates.py (t2_log1p_additive)
shows this already happens: shift = beta * z_profile * sigma_g(context), where sigma_g already comes
from THAT context's own pseudobulk reference (pseudobulk_reference.py). A flat per-gene beta already
produces a context-varying shift for free. A second scaling layer on top would double-count that, not
add anything new. Real per-context EFFECT specificity (beyond what sigma_g already provides) would
need real per-context ground truth to validate against, which -- per generate_b2_submission.py's own
stated limitation -- does not exist locally (vcc-perturb is single-cell-line H1 only).

What this script does:
  1. Rebuilds the real 13/300 z-profiles (identical to B2, beta=1.5, untouched).
  2. Rebuilds the Phase 3 co-expression KNN profile for the 287 unknown genes (identical logic to
     phase3_coexpr.py; gene order from adata.var_names, never gene_names.csv -- see that script's
     header comment for why).
  3. Runs a LOO beta sweep over the 150 known genes, comparing three z-profile-source candidates
     (flat global mean / co-expression KNN / zero) at beta in {0, 0.15, 0.3, 0.6, 1.0, 1.5}, scored
     by raw MSE-to-truth (the same criterion that rejected the flat mean) alongside raw-cosine and
     discrimination_score (scale-invariant, so computed once per candidate, not per beta).
  4. Picks a nonzero beta for the 287 genes ONLY if coexpr clears BOTH: a discrimination margin over
     the mean baseline of at least 2x the approximate sampling noise at n=150 known genes (~0.024,
     so a ~0.047 margin required -- not the softer 0.02 used in casual discussion earlier), AND a
     raw MSE at its best beta that actually beats null. Otherwise this script keeps BETA_MEAN-style
     behavior (beta=0) for the 287, same as B2 -- and says so plainly rather than forcing a "win".
  5. Saves models/b3_profiles.npy (300, N_GENES), models/b3_betas.npy (300,), and models/b3_meta.json
     in the same artifact shape phase3_coexpr.py already produces, ready for a generate_b3_submission.py
     to consume -- regardless of which way the verdict goes, so the artifact always reflects the
     honestly-validated choice, not a guess.

Usage:
    python models/gene_transfer_b3.py
"""
import os, sys, json
from pathlib import Path

import numpy as np
import pandas as pd
import anndata as ad

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gene_transfer_b2 import discrimination_score, cosine_row  # reuse the exact, already-trusted scorers

# ── Paths ──────────────────────────────────────────────────────────────────
CONTROLS_DIR = Path(os.environ.get("VCC_CONTROLS_DIR", "/mnt/c/Users/User/Downloads/controls"))
VCC_PERTURB  = Path(os.environ.get("VCC_PERTURB_CSV", "/mnt/c/VCC/vcc-perturb.csv"))
OUT_DIR      = Path("models")
OUT_DIR.mkdir(exist_ok=True)
OUT_PROFILES = OUT_DIR / "b3_profiles.npy"
OUT_BETAS    = OUT_DIR / "b3_betas.npy"
OUT_META     = OUT_DIR / "b3_meta.json"

# ── Hyperparameters (identical to phase3_coexpr.py for the coexpr side) ────
K_NEIGHBORS = 5
WEIGHT_EXP  = 2
MIN_CORR    = 0.05
NTC_LABEL   = "non-targeting"
BETA_REAL   = 1.5                                # unchanged from B2 -- real 13/300 genes
BETA_SWEEP  = [0.0, 0.15, 0.3, 0.6, 1.0, 1.5]     # candidate betas for the 287 unknown genes

print("=== B3: real passthrough + validated coexpr blend (no flat-mean fallback) ===\n")

# ── VCC target genes ─────────────────────────────────────────────────────
pert_df     = pd.read_csv(CONTROLS_DIR / "pert_counts.csv")
vcc_targets = list(pert_df["target_gene"].str.strip())
print(f"VCC target genes: {len(vcc_targets)}")

# ── Genome gene order + NTC baseline cells (adata.var_names, NOT gene_names.csv) ───────────
print("\nLoading NTC baseline cells ...")
ntc_list, gene_names = [], None
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
print(f"\nGenome size: {N_GENES} genes")

ntc_expr = np.concatenate(ntc_list, axis=0)
ntc_c    = ntc_expr - ntc_expr.mean(axis=0, keepdims=True)
print(f"Total NTC cells: {ntc_expr.shape[0]}")

# ── Real vcc-perturb response matrix, mapped into genome space ─────────────
print("\nLoading vcc-perturb response matrix ...")
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


def coexpr_predict_loo(loo_i, gidx):
    """Same KNN logic as phase3_coexpr.py, with `loo_i` excluded from its own neighbor pool."""
    vec   = ntc_c[:, gidx]
    vec_n = vec / (np.linalg.norm(vec) + 1e-10)
    corrs = X_known_n @ vec_n
    corrs_masked = corrs.copy()
    corrs_masked[loo_i] = -999
    top_idx   = np.argsort(corrs_masked)[::-1][:K_NEIGHBORS]
    top_corrs = corrs_masked[top_idx]
    valid     = top_corrs > MIN_CORR
    other_ridx = [known_ridx[j] for j in range(K_known) if j != loo_i]
    if valid.sum() == 0:
        return R[other_ridx].mean(axis=0)
    top_idx, top_corrs = top_idx[valid], top_corrs[valid]
    weights = top_corrs ** WEIGHT_EXP
    weights /= weights.sum()
    ridx_sel = [known_ridx[ki] for ki in top_idx]
    return (weights[:, None] * R[ridx_sel]).sum(axis=0)


def coexpr_predict_full(gidx):
    """Same as above but for a truly unknown gene -- no self-exclusion needed (it was never in the
    known bank to begin with)."""
    vec   = ntc_c[:, gidx]
    vec_n = vec / (np.linalg.norm(vec) + 1e-10)
    corrs = X_known_n @ vec_n
    top_idx   = np.argsort(corrs)[::-1][:K_NEIGHBORS]
    top_corrs = corrs[top_idx]
    valid     = top_corrs > MIN_CORR
    if valid.sum() == 0:
        return R[known_ridx].mean(axis=0)
    top_idx, top_corrs = top_idx[valid], top_corrs[valid]
    weights = top_corrs ** WEIGHT_EXP
    weights /= weights.sum()
    ridx_sel = [known_ridx[ki] for ki in top_idx]
    return (weights[:, None] * R[ridx_sel]).sum(axis=0)


# ── LOO: build unscaled candidate predictions for all 150 known genes ──────
print("Running LOO to build candidate predictions (mean / coexpr / zero) ...")
mean_preds   = np.zeros((K_known, N_GENES), dtype=np.float32)
coexpr_preds = np.zeros((K_known, N_GENES), dtype=np.float32)

for loo_i, (g, ridx, gidx) in enumerate(known_in_genome):
    other_ridx = [known_ridx[j] for j in range(K_known) if j != loo_i]
    mean_preds[loo_i]   = R[other_ridx].mean(axis=0)
    coexpr_preds[loo_i] = coexpr_predict_loo(loo_i, gidx)

truth      = R[known_ridx]
zero_preds = np.zeros_like(truth)

# ── Direction-quality metrics (scale-invariant, one number per candidate) ──
print(f"\n{'Candidate':<16} {'raw_cosine':>10} {'discrimination':>15}")
print("-" * 44)
direction_scores = {}
for key, preds in [("mean", mean_preds), ("coexpr", coexpr_preds), ("zero", zero_preds)]:
    raw_cos = float(cosine_row(preds, truth).mean())
    disc    = discrimination_score(preds, truth)
    direction_scores[key] = disc
    print(f"{key:<16} {raw_cos:>10.4f} {disc:>15.3f}")

approx_se = float(np.sqrt(1 / 12) / np.sqrt(K_known))
required_margin = 2.0 * approx_se
print(f"\napprox. sampling noise on discrimination score at n={K_known}: SE~{approx_se:.4f}")
print(f"required margin over mean baseline to count as real signal (2x SE): {required_margin:.4f}")

# ── Beta sweep: raw MSE to truth (the criterion that already rejected flat-mean@nonzero-beta) ──
print(f"\n{'Candidate':<10} " + " ".join(f"beta={b:<5.2f}" for b in BETA_SWEEP))
print("-" * (10 + 11 * len(BETA_SWEEP)))
mse_table = {}
for key, preds in [("mean", mean_preds), ("coexpr", coexpr_preds)]:
    row = [float(np.mean((beta * preds - truth) ** 2)) for beta in BETA_SWEEP]
    mse_table[key] = row
    print(f"{key:<10} " + " ".join(f"{v:<10.4f}" for v in row))

null_mse = mse_table["mean"][BETA_SWEEP.index(0.0)]  # beta=0 -> identical null MSE for any candidate
print(f"\nnull (beta=0) MSE (reference floor): {null_mse:.4f}")


def best_beta(key):
    row = mse_table[key]
    bi  = int(np.argmin(row))
    return BETA_SWEEP[bi], row[bi]


mean_best_beta, mean_best_mse     = best_beta("mean")
coexpr_best_beta, coexpr_best_mse = best_beta("coexpr")
print(f"\nBest beta for mean:   {mean_best_beta}  (MSE={mean_best_mse:.4f}, "
      f"{'beats' if mean_best_mse < null_mse else 'loses to'} null)")
print(f"Best beta for coexpr: {coexpr_best_beta}  (MSE={coexpr_best_mse:.4f}, "
      f"{'beats' if coexpr_best_mse < null_mse else 'loses to'} null)")

# ── Verdict: does coexpr earn a nonzero beta for the 287 unknown genes? ────
mean_disc, coexpr_disc = direction_scores["mean"], direction_scores["coexpr"]
disc_margin              = coexpr_disc - mean_disc
coexpr_clears_direction  = disc_margin >= required_margin
coexpr_clears_mse        = coexpr_best_mse < null_mse

print("\nVerdict:")
if coexpr_clears_direction and coexpr_clears_mse:
    final_beta_287 = coexpr_best_beta
    source_287 = "coexpr"
    print(f"  coexpr clears BOTH bars -- discrimination margin {disc_margin:.4f} >= required "
          f"{required_margin:.4f}, AND MSE beats null at beta={coexpr_best_beta} "
          f"({coexpr_best_mse:.4f} < {null_mse:.4f}). Using coexpr at beta={final_beta_287} "
          f"for the 287 unknown genes -- a real change from B2's beta=0.")
else:
    final_beta_287 = 0.0
    source_287 = "coexpr"  # profile still saved for the record; beta=0 makes it inert either way
    reasons = []
    if not coexpr_clears_direction:
        reasons.append(f"discrimination margin {disc_margin:.4f} < required {required_margin:.4f} "
                        f"(not distinguishable from noise at n={K_known})")
    if not coexpr_clears_mse:
        reasons.append(f"best coexpr MSE ({coexpr_best_mse:.4f}) does not beat null ({null_mse:.4f})")
    print(f"  coexpr does NOT clear the bar -- {'; '.join(reasons)}. Keeping beta=0 for the 287 "
          f"unknown genes, same conservative choice B2 already made. No change from B2's real standing.")

# ── Build final (300, N_GENES) z-profile + per-gene beta, same artifact shape as phase3_coexpr.py ──
print("\nBuilding final B3 profiles for all 300 VCC targets ...")
known_set = {g: i for i, g in enumerate(known_genes)}
predicted_profiles = np.zeros((len(vcc_targets), N_GENES), dtype=np.float32)
per_gene_beta      = np.zeros(len(vcc_targets), dtype=np.float32)
gene_source        = []
real_count, coexpr_count, b0_count = 0, 0, 0

for ti, gene in enumerate(vcc_targets):
    if gene in known_set:
        predicted_profiles[ti] = R[known_set[gene]]
        per_gene_beta[ti]      = BETA_REAL
        gene_source.append("real")
        real_count += 1
    elif gene in gene_idx:
        gidx = gene_idx[gene]
        predicted_profiles[ti] = coexpr_predict_full(gidx)
        per_gene_beta[ti]      = final_beta_287
        gene_source.append(source_287)
        coexpr_count += 1
    else:
        per_gene_beta[ti] = 0.0
        gene_source.append("b0")
        b0_count += 1

print(f"  Real (vcc-perturb overlap):    {real_count}  (beta={BETA_REAL})")
print(f"  Co-expression ({source_287}):  {coexpr_count}  (beta={final_beta_287})")
print(f"  B0 fallback (not in genome):   {b0_count}")

np.save(OUT_PROFILES, predicted_profiles)
np.save(OUT_BETAS, per_gene_beta)

meta = {
    "vcc_targets": vcc_targets,
    "gene_source": gene_source,
    "beta_real": BETA_REAL,
    "beta_287": final_beta_287,
    "coexpr_discrimination": coexpr_disc,
    "mean_discrimination": mean_disc,
    "discrimination_margin": disc_margin,
    "required_margin_2x_se": required_margin,
    "approx_se_at_n": approx_se,
    "coexpr_best_beta_by_mse": coexpr_best_beta,
    "coexpr_best_mse": coexpr_best_mse,
    "mean_best_beta_by_mse": mean_best_beta,
    "mean_best_mse": mean_best_mse,
    "null_mse": null_mse,
    "n_known_in_genome": K_known,
    "beta_sweep": BETA_SWEEP,
    "mse_table": mse_table,
}
with open(OUT_META, "w") as f:
    json.dump(meta, f, indent=2)

print(f"\nSaved: {OUT_PROFILES}  shape={predicted_profiles.shape}")
print(f"Saved: {OUT_BETAS}")
print(f"Saved: {OUT_META}")
print(f"\nFinal call: 287 unknown genes get beta={final_beta_287} "
      f"({'coexpr signal wired in' if final_beta_287 > 0 else 'B0-equivalent, unchanged from B2'}).")
