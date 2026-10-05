"""
Second attempt at a gene-effect transfer signal, after the perturbation-response-column embedding
(gene_transfer_b2.py) validated at chance once corrected to a genuinely zero-shot-honest embedding
(see REPORT_phase0-5.md's correction note).

Different, independently-available signal: gene-gene CO-EXPRESSION computed from the REAL 2026
control single cells (not from vcc-perturb at all). Hypothesis being tested: genes that are
co-regulated at baseline (correlated across real control cells -- same complex, same pathway, shared
upstream regulator) tend to show correlated behavior when perturbed. This is a different, classic
assumption (e.g. WGCNA-style module inference) from the "similar perturbation-response profile"
hypothesis that failed -- worth testing on its own rather than assumed to fail too.

Concretely: for each of the 150 known vcc-perturb genes ("landmarks"), compute its real Pearson
correlation (in log1p(CP10K) space) to every one of the 18,533 genes across a sample of real 2026
context-A control cells. This gives EVERY gene (including all 300 2026 targets, unlike the 97.5%-only
overlap of the response-column embedding) a 150-dim "coexpression-with-known-genes" embedding,
directly comparable in shape and role to the failed embedding, so it plugs into the exact same
corrected (column-based, self-row-excluded) leave-one-out harness for an apples-to-apples comparison.

Usage:
    python models/coexpression_transfer.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import anndata as ad

import config as cfg
from models.gene_transfer_b2 import load_response_matrix, knn_predict, discrimination_score, cosine_row

N_CELLS_FOR_COEXPR = 18400  # all of context A's released controls -- a 6,000-cell subsample left
                            # ~18.5% of the full gene axis with exactly zero detected counts (pure
                            # subsampling artifact for lowly-expressed genes), which produces
                            # degenerate all-zero, norm-0 embedding rows -> NaN cosine similarity.
                            # Using the full pool cuts that to 4/150 known targets (real biology --
                            # a handful of genes really are essentially undetected in this context),
                            # which are excluded explicitly below rather than silently propagating NaN.


def build_coexpression_embedding(context="A", n_cells=N_CELLS_FOR_COEXPR, seed=0):
    """Returns (gene_names: (18533,), embed_matrix: (18533, n_landmarks)) -- landmark genes are
    whichever of vcc-perturb's 150 known targets are present in this context's gene axis (should be
    all 150, gene axis overlap is 97.5% and these are a specific 150-gene subset, checked below)."""
    rng = np.random.default_rng(seed)
    adata = ad.read_h5ad(cfg.CONTEXT_FILES[context])
    gene_names = adata.var_names.to_numpy()
    idx = rng.choice(adata.n_obs, size=n_cells, replace=False)
    X = adata.X[idx].tocsr()
    dense = np.asarray(X.todense(), dtype=np.float32)
    totals = dense.sum(axis=1, keepdims=True)
    totals[totals == 0] = 1
    log1p_cp10k = np.log1p(dense / totals * 1e4)
    del dense, X, adata

    mu = log1p_cp10k.mean(axis=0)
    sigma = log1p_cp10k.std(axis=0)
    sigma_safe = np.where(sigma < 1e-6, 1e-6, sigma)
    standardized = (log1p_cp10k - mu) / sigma_safe   # (n_cells, n_genes)
    return gene_names, standardized


def correlation_embedding(standardized, gene_pos, landmark_genes):
    """(n_genes, n_landmarks) Pearson correlation matrix via standardized dot product."""
    n_cells = standardized.shape[0]
    landmark_idx = [gene_pos[g] for g in landmark_genes]
    landmark_block = standardized[:, landmark_idx]              # (n_cells, n_landmarks)
    corr = (standardized.T @ landmark_block) / n_cells           # (n_genes, n_landmarks)
    return np.clip(corr, -1, 1)


def leave_one_out_coexpr(targets, gene_pos, coexpr_embed, R, k, center=False):
    """
    Same corrected, column-based-equivalent LOO structure as gene_transfer_b2.py's fixed
    validation, but the EMBEDDING now comes from real coexpression, not from R's own columns --
    genuinely independent of the (failed) perturbation-response-column hypothesis. `R` is still
    used as the thing being TRANSFERRED (real measured full response rows) -- that part hasn't
    changed, only what defines "similar".

    No `center` option here (unlike gene_transfer_b2.py): centering only changes a KNN transfer's
    output when it also shifts the EMBEDDING used to pick neighbors -- in gene_transfer_b2.py the
    embedding was derived from R itself, so subtracting R's mean moved it. Here the embedding
    (coexpr_embed) is a completely separate, independently-computed quantity that centering R
    doesn't touch, so "subtract mu from the transferred rows, then add mu back to a weighted
    average" is an exact algebraic no-op (weighted_avg(r - mu) + mu == weighted_avg(r) whenever the
    weights sum to 1, regardless of mu) -- confirmed empirically before removing it, not assumed.
    """
    n = len(targets)
    landmark_idx_in_targets = {g: i for i, g in enumerate(targets)}  # landmarks == targets here
    knn_preds = np.zeros((n, R.shape[1]))

    for i in range(n):
        others = [j for j in range(n) if j != i]
        mu = R[others].mean(axis=0) if center else np.zeros(R.shape[1])
        responses_others = R[others] - mu

        # coexpression embedding restricted to the "others" landmark set, for query and bank alike
        other_gene_names = [targets[j] for j in others]
        query_embedding = coexpr_embed[gene_pos[targets[i]]][[landmark_idx_in_targets[g] for g in other_gene_names]].reshape(1, -1)
        embeddings_others = coexpr_embed[[gene_pos[g] for g in other_gene_names]][:, [landmark_idx_in_targets[g] for g in other_gene_names]]

        pred, _ = knn_predict(query_embedding, embeddings_others, responses_others, k=k)
        knn_preds[i] = pred + mu

    return {
        "k": k, "center": center,
        "raw_cosine_to_truth": float(cosine_row(knn_preds, R).mean()),
        "discrimination_score": discrimination_score(knn_preds, R),
    }


if __name__ == "__main__":
    print("[coexpr] loading vcc-perturb response matrix ...")
    targets, gene_cols, R = load_response_matrix()

    print(f"[coexpr] building real coexpression embedding from context A controls "
          f"({N_CELLS_FOR_COEXPR} cells x 18,533 genes) ...")
    gene_names, standardized = build_coexpression_embedding()
    gene_pos = {g: i for i, g in enumerate(gene_names)}
    missing = [g for g in targets if g not in gene_pos]
    print(f"[coexpr] {len(missing)}/{len(targets)} known targets missing from context A's gene axis "
          f"(should be 0 -- all 150 are on the 2026 axis): {missing[:5]}")

    coexpr_embed = correlation_embedding(standardized, gene_pos, targets)  # (18533, 150)
    del standardized

    zero_norm = np.linalg.norm(coexpr_embed[[gene_pos[g] for g in targets]], axis=1) == 0
    dropped = [g for g, z in zip(targets, zero_norm) if z]
    print(f"[coexpr] dropping {len(dropped)}/{len(targets)} known targets with a zero-variance "
          f"(essentially undetected in context A) embedding row: {dropped}")
    targets_use = [g for g in targets if g not in dropped]
    keep_idx = [targets.index(g) for g in targets_use]
    R_use = R[keep_idx]

    print(f"[coexpr] running corrected leave-one-out validation with the COEXPRESSION embedding "
          f"on {len(targets_use)} genes ...")
    for k in (1, 3, 5, 10, 20, 40):
        r = leave_one_out_coexpr(targets_use, gene_pos, coexpr_embed, R_use, k=k, center=False)
        print(f"  k={k:3d}  raw_cosine={r['raw_cosine_to_truth']:.4f}  "
              f"discrimination={r['discrimination_score']:.4f}  (chance=0.500)")
