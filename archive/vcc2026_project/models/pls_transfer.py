"""
Third attempt at a gene-effect transfer signal. KNN (both the perturbation-response-column
embedding and the coexpression embedding, see gene_transfer_b2.py and coexpression_transfer.py)
scored at chance discrimination. KNN is a local method and can struggle in 150-300 dimensions with
only ~150 training examples (curse of dimensionality); a regularized global regression has a
different inductive bias and is worth checking on its own rather than concluding "no signal exists"
from one method family alone.

PLS (partial least squares) regression is specifically built for exactly this shape of problem
(high-dimensional multivariate OUTPUT, small n, correlated predictors) -- fits a small number of
latent components that jointly explain covariance between the embedding (predictors) and the full
response matrix (targets), which a plain nearest-neighbor average cannot represent (it can only
output convex combinations of training rows; PLS/any linear regression can extrapolate).

Tested on both embeddings from the two prior attempts, cross-validated leave-one-out (proper: PLS
is refit from scratch excluding the held-out gene every time, not fit once and evaluated in-sample).

Usage:
    python models/pls_transfer.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from sklearn.cross_decomposition import PLSRegression

from models.gene_transfer_b2 import load_response_matrix, discrimination_score, cosine_row
from models.coexpression_transfer import build_coexpression_embedding, correlation_embedding


def loo_pls(embed_matrix, R, n_components, loo_idx=None):
    """embed_matrix: (n_genes_total, D) full embedding table indexed the same way as R's rows
    (embed_matrix[i] corresponds to R[i]). Refits PLS per held-out gene.

    loo_idx: optional subset of row indices to actually hold out and score (each fit still excludes
    only that one gene from its own training set, drawing on all OTHER genes including ones not in
    loo_idx -- this only trims which genes get scored, at ~2.2s/fit, not the training data each fit
    sees). Discrimination score is still computed against the full R so "did we correctly identify
    which of all ~149 genes this is" keeps its original meaning even when only a subset is queried.
    """
    n = R.shape[0]
    loo_idx = np.arange(n) if loo_idx is None else np.asarray(loo_idx)
    preds = np.zeros((len(loo_idx), R.shape[1]))
    for out_i, i in enumerate(loo_idx):
        others = [j for j in range(n) if j != i]
        pls = PLSRegression(n_components=n_components, scale=True)
        pls.fit(embed_matrix[others], R[others])
        preds[out_i] = pls.predict(embed_matrix[i:i + 1])[0]

    # discrimination score against the FULL gene set (not just the scored subset) -- rank each
    # prediction's true match among all n candidates, exactly like the full-scale version
    A = preds / (np.linalg.norm(preds, axis=1, keepdims=True) + 1e-12)
    B = R / (np.linalg.norm(R, axis=1, keepdims=True) + 1e-12)
    S = A @ B.T  # (len(loo_idx), n)
    ranks = (-S).argsort(axis=1)
    rank_of_true = np.array([np.where(ranks[k] == loo_idx[k])[0][0] for k in range(len(loo_idx))])
    disc = float((1 - rank_of_true / (n - 1)).mean())

    truth = R[loo_idx]
    return {
        "n_components": n_components,
        "raw_cosine_to_truth": float(cosine_row(preds, truth).mean()),
        "discrimination_score": disc,
    }


if __name__ == "__main__":
    print("[pls] loading vcc-perturb response matrix ...")
    targets, gene_cols, R = load_response_matrix()
    target_col_idx = np.array([gene_cols.index(g) for g in targets])

    print("[pls] building perturbation-response-column embedding (column-based, as corrected) ...")
    # embed_pert[i] = gene i's column across the 150 known rows (its own row's own-column entry
    # included is fine here -- PLS, unlike the LOO-embedding code, re-excludes row i entirely at fit
    # time via `others`, so there's no leakage; this matches exactly what's available for a real new
    # gene: R[:, col_idx(gene)] using ALL known rows)
    embed_pert = R[:, target_col_idx].T  # (150, 150): row i = gene i's column, i.e. how each of the 150 knockdowns affected gene i

    print("[pls] building real coexpression embedding from context A controls ...")
    gene_names, standardized = build_coexpression_embedding()
    gene_pos = {g: i for i, g in enumerate(gene_names)}
    coexpr_full = correlation_embedding(standardized, gene_pos, targets)  # (18533, 150)
    del standardized
    embed_coexpr = coexpr_full[[gene_pos[g] for g in targets]]  # (150, 150) -- restricted to the 150 known genes as entities

    zero_norm = np.linalg.norm(embed_coexpr, axis=1) == 0
    keep = ~zero_norm
    print(f"[pls] dropping {(~keep).sum()} genes with zero-variance coexpression embedding")
    R_use = R[keep]
    embed_pert_use = embed_pert[keep]
    embed_coexpr_use = embed_coexpr[keep]
    embed_combined_use = np.concatenate([embed_pert_use, embed_coexpr_use], axis=1)

    rng = np.random.default_rng(0)
    n_use = R_use.shape[0]
    loo_idx = rng.choice(n_use, size=min(60, n_use), replace=False)
    print(f"[pls] scoring on a random {len(loo_idx)}/{n_use}-gene LOO subset (each fit still trains "
          f"on all other {n_use-1} genes -- ~2.2s/fit, full 150-gene LOO would take ~1hr per config)")

    for name, embed in [("perturbation-response-column", embed_pert_use),
                         ("coexpression", embed_coexpr_use),
                         ("combined (concat both)", embed_combined_use)]:
        print(f"\n[pls] embedding = {name} (dim={embed.shape[1]})")
        for nc in (5, 15):
            if nc >= embed.shape[1]:
                continue
            r = loo_pls(embed, R_use, n_components=nc, loo_idx=loo_idx)
            print(f"  n_components={nc:3d}  raw_cosine={r['raw_cosine_to_truth']:.4f}  "
                  f"discrimination={r['discrimination_score']:.4f}  (chance=0.500)", flush=True)
