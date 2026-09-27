"""
B2 -- gene-feature transfer model, built around a "transferable gene->response
representation" rather than target-gene lookup (per explicit instruction).

Why not a lookup table (B1): data/audit/audit_vcc_perturb.py found only
13/300 (4.3%) of the 2026 target genes were themselves knocked down in
vcc-perturb.csv. A lookup table covers 4.3% of any context and falls back to
B0 (no-op) for the other 95.7% -- not a real strategy on its own.

The representation used here: vcc-perturb.csv is 150 measured knockdowns x
18,080 response genes (z-scores, see audit -- NOT logFC/deltas). Transposing
it gives every one of those 18,080 genes a 150-dim "perturbation-response
embedding": how much does gene G move when each of the 150 known genes is
knocked down? Because all 300 of 2026's target genes appear as response
columns (confirmed in the audit) -- even the 287 never themselves knocked
down -- every 2026 target gene has this embedding, not just the 13 overlap
genes. That is the "gene->response representation" transferred here: no
target-gene identity is used as a lookup key, only its position in this
learned response space.

Transfer rule (K-nearest-neighbors in embedding space): to predict target
gene G's own knockdown response vector, find the K known-target genes whose
150-dim embeddings are most similar (cosine) to G's, and predict G's response
as the similarity-weighted average of THEIR measured response vectors. This
assumes response-space neighbors tend to have symmetric perturbation
behavior (a real, named assumption -- guilt-by-association across a
functional module/complex/pathway -- not proven, hence the leave-one-out
check below before trusting it).

VALIDATION FIRST (Phase 5, Experiment A -- held-out target genes): before
this is wired into any submission, every one of the 150 known target genes
is held out in turn, its embedding built only from the OTHER 149, and its
predicted response vector compared to its real measured one. This is
run as the __main__ block and its numbers are what should be trusted, not
the paragraph above.

Usage:
    python models/gene_transfer_b2.py
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

import config as cfg

# Override with the VCC_PERTURB_CSV env var when running outside the original sandbox (e.g. on a
# machine where this reference file lives at a different path).
VCC_PERTURB_CSV = Path(os.environ.get("VCC_PERTURB_CSV", "/mnt/user-data/uploads/vcc-perturb.csv"))


def _ensure_assembled(path: Path) -> Path:
    """This reference file is ~149MB, too large for some delivery channels, so it was split into
    numbered '<name>.NN.part' siblings for handoff. If `path` itself isn't present but those parts
    are (in the same folder), transparently concatenate them into `path` first -- byte-exact in
    numeric order, verified once against the original file before it was split. Leaves the .part
    files in place (harmless) and only assembles once, since later calls find `path` already there.
    """
    if path.exists():
        return path
    parts = sorted(path.parent.glob(path.name + ".*.part"))
    if not parts:
        raise FileNotFoundError(
            f"{path} not found, and no '{path.name}.NN.part' split files found beside it either. "
            f"Set the VCC_PERTURB_CSV environment variable to this reference file's actual location, "
            f"or place its split .part files in the same folder as the path it points to."
        )
    tmp_path = path.with_name(path.name + ".assembling")
    with open(tmp_path, "wb") as out:
        for p in parts:
            with open(p, "rb") as f:
                out.write(f.read())
    tmp_path.rename(path)
    print(f"[gene_transfer_b2] assembled {path} from {len(parts)} split part(s)")
    return path


def load_response_matrix():
    """Return (targets: list[str] len T, gene_cols: list[str] len G, R: (T,G) mean-over-replicates z-score matrix)."""
    df = pd.read_csv(_ensure_assembled(VCC_PERTURB_CSV))
    gene_cols = [c for c in df.columns if c != "__index_level_0__"]
    grouped = df.groupby("__index_level_0__")[gene_cols].mean()  # average the 5 noisy pseudobulk "replicates"
    return grouped.index.tolist(), gene_cols, grouped.to_numpy(dtype=np.float64)


def knn_predict(query_embeddings, bank_embeddings, bank_responses, k, exclude_self_idx=None):
    """
    query_embeddings: (Q, D), bank_embeddings: (N, D), bank_responses: (N, G).
    exclude_self_idx: optional array of length Q giving, for each query, a bank index to exclude
    (used only in leave-one-out validation so a gene never predicts itself).
    Returns (Q, G) predictions, plus the mean similarity of the neighbors actually used (diagnostic).
    """
    sim = 1 - cdist(query_embeddings, bank_embeddings, metric="cosine")  # (Q, N)
    if exclude_self_idx is not None:
        for qi, bi in enumerate(exclude_self_idx):
            sim[qi, bi] = -np.inf
    preds = np.zeros((query_embeddings.shape[0], bank_responses.shape[1]))
    mean_top_sim = np.zeros(query_embeddings.shape[0])
    for qi in range(sim.shape[0]):
        top_idx = np.argsort(-sim[qi])[:k]
        w = sim[qi, top_idx]
        w = np.clip(w, 1e-6, None)  # cosine sim on z-score vectors can be negative; weight only positive-ish signal
        w = w / w.sum()
        preds[qi] = w @ bank_responses[top_idx]
        mean_top_sim[qi] = sim[qi, top_idx].mean()
    return preds, mean_top_sim


def cosine_row(a, b):
    num = (a * b).sum(axis=1)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    den = np.where(den == 0, 1e-12, den)
    return num / den


def discrimination_score(pred_matrix, truth_matrix):
    """
    Local proxy for the official pds_cosine metric's actual logic (report S1.3): for each
    predicted vector, rank ALL n true vectors by cosine similarity and see where the gene's own
    true vector lands. 1.0 = always the single closest match, 0.5 = chance, 0.0 = always dead last.
    This is a materially different question from "is the prediction close to the truth" (raw
    cosine, computed by `cosine_row` above) -- a predictor can be reasonably close to every truth
    in raw terms (because most of a response vector's variance is a shared, gene-independent
    stress/arrest signature -- see the k-dependence finding in __main__) while still being useless
    at telling genes apart, which is what this -- and pds_cosine -- actually measure.
    """
    A = pred_matrix / (np.linalg.norm(pred_matrix, axis=1, keepdims=True) + 1e-12)
    B = truth_matrix / (np.linalg.norm(truth_matrix, axis=1, keepdims=True) + 1e-12)
    S = A @ B.T
    n = S.shape[0]
    ranks = (-S).argsort(axis=1)
    rank_of_true = np.array([np.where(ranks[i] == i)[0][0] for i in range(n)])
    return float((1 - rank_of_true / (n - 1)).mean())


def leave_one_out_validation(targets, gene_cols, R, k=5, center=False):
    """
    IMPORTANT -- embedding direction, fixed after an initial bug: a real 2026 target gene that
    vcc-perturb never itself knocked down has no ROW in R (no measured "what does knocking THIS
    gene down do to everything else") -- the only thing we ever have for it is a COLUMN ("how did
    the 150 known knockdowns affect this gene's own expression", R[:, col_idx(gene)], a 150-dim
    vector). An earlier version of this function embedded each held-out gene using its own ROW
    (restricted to the other genes' columns) -- that only works because held-out genes in this
    leave-one-out loop happen to already have a row (they're drawn from the known 150), and it does
    NOT match what's actually available for the ~96% of real 2026 targets that were never tested.
    Fixed here to use COLUMN-based embeddings throughout, so this validation is honest about what
    it's actually certifying for production (generate_b2_submission.py uses the identical
    column-based embedding).

    For each of the 150 known targets, held out as if it were a fresh, never-tested query gene: its
    embedding is R[others, col_idx(gene)] -- how the OTHER 149 known knockdowns affected it -- using
    only its column, never its own row. Each of those 149 other genes gets the same treatment (its
    column, restricted to the same 149-row reference set) so query and bank embeddings live in the
    same space. Nearest neighbors are found by that embedding similarity, and their OWN measured
    full ROWS (their real 18,080-gene knockdown effect, which the query gene doesn't have) are
    weighted-averaged as the prediction -- transferring through "responds similarly to the same
    reference perturbations" (co-regulation), not through "has a similar effect on others."

    center=True subtracts the (leave-one-out-honest) mean, computed over the other 149 genes' full
    rows, from the responses being transferred (and adds it back to the prediction) -- isolating
    gene-specific signal from the shared common-mode response before the neighbor match. See
    __main__ for why this matters. Compared against zero (B0-equivalent) and global-mean baselines.
    """
    target_col_idx = np.array([gene_cols.index(g) for g in targets])  # position of each of the 150 known genes within the 18080 response columns
    n = len(targets)
    knn_preds = np.zeros((n, R.shape[1]))
    mean_sim_used = np.zeros(n)

    for i in range(n):
        others = [j for j in range(n) if j != i]
        mu = R[others].mean(axis=0) if center else np.zeros(R.shape[1])
        responses_others = R[others] - mu               # (149, 18080) -- what actually gets transferred (real full rows)

        # column-based embeddings, all restricted to the same 149-row reference set for a fair LOO
        embeddings_others = R[others][:, target_col_idx[others]].T - mu[target_col_idx[others]].reshape(-1, 1)
        # ^ (149 entities x 149 features): entity h's embedding = column h of R[others], i.e. how
        #   each of the 149 reference knockdowns affected gene h. .T turns "column h across rows"
        #   into "row h across the 149 reference features" so it matches knn_predict's (N, D) shape.
        query_embedding = (R[others, target_col_idx[i]] - mu[target_col_idx[i]]).reshape(1, -1)
        # ^ gene i's own column (excluding nothing extra -- row i is already outside `others`):
        #   how each of the 149 reference knockdowns affected gene i.

        pred, sims = knn_predict(query_embedding, embeddings_others, responses_others, k=k)
        knn_preds[i] = pred + mu
        mean_sim_used[i] = sims[0]

    global_mean_pred = np.tile(R.mean(axis=0, keepdims=True), (n, 1))
    zero_pred = np.zeros_like(R)
    knn_cos = cosine_row(knn_preds, R)

    return {
        "k": k,
        "center": center,
        "n_genes": n,
        "knn_transfer_mean_cosine_to_truth": float(knn_cos.mean()),
        "global_mean_baseline_cosine_to_truth": float(cosine_row(global_mean_pred, R).mean()),
        "pct_genes_knn_beats_mean_baseline_on_raw_cosine": float((knn_cos > cosine_row(global_mean_pred, R)).mean() * 100),
        "knn_transfer_discrimination_score": discrimination_score(knn_preds, R),
        "global_mean_baseline_discrimination_score": discrimination_score(global_mean_pred, R),
        "zero_baseline_discrimination_score": discrimination_score(zero_pred, R),
        "mean_embedding_similarity_of_neighbors_used": float(mean_sim_used.mean()),
    }


if __name__ == "__main__":
    print("[B2] loading and averaging vcc-perturb.csv replicates ...")
    targets, gene_cols, R = load_response_matrix()
    print(f"[B2] response matrix: {R.shape[0]} targets x {R.shape[1]} response genes")

    print("[B2] running leave-one-out zero-shot transfer validation (Phase 5, Experiment A) ...")
    print("     -- two different questions, deliberately kept separate:")
    print("     (a) raw cosine to truth: is the prediction numerically close to the real response?")
    print("     (b) discrimination score: does the prediction correctly tell THIS gene's effect apart")
    print("         from the other 149 genes' effects? (this is what pds_cosine, report S1.3, actually measures)")
    print()
    print("  -- uncentered --")
    for k in (1, 3, 5, 10, 20, 40):
        r = leave_one_out_validation(targets, gene_cols, R, k=k, center=False)
        print(f"  k={k:3d}  raw_cosine={r['knn_transfer_mean_cosine_to_truth']:.4f} "
              f"(baseline={r['global_mean_baseline_cosine_to_truth']:.4f})  "
              f"discrimination={r['knn_transfer_discrimination_score']:.3f} "
              f"(baseline={r['global_mean_baseline_discrimination_score']:.3f}, chance=0.500)")

    print("\n  -- centered (common-mode response direction removed before matching) --")
    for k in (1, 3, 5, 10, 20):
        r = leave_one_out_validation(targets, gene_cols, R, k=k, center=True)
        print(f"  k={k:3d}  raw_cosine={r['knn_transfer_mean_cosine_to_truth']:.4f}  "
              f"discrimination={r['knn_transfer_discrimination_score']:.3f}")

    print("\n[B2] FINDING: raw cosine-to-truth rises with k (more neighbors -> smoother, closer-to-mean")
    print("     predictions), but discrimination FALLS with k (over-smoothing erases the gene-specific")
    print("     signal that separates one gene's effect from another's). These trade off in opposite")
    print("     directions -- k should be chosen per which of the 6 official metrics matters more for")
    print("     a given model, not fixed once. Centering partially decouples them: it raises the")
    print("     discrimination score at low-to-moderate k without giving up much raw accuracy. See")
    print("     REPORT_phase0-5.md addendum for the full read on this.")
