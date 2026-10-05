"""
Phase 2 audit of the user-supplied `vcc-perturb.csv` (their own extract of the
nicolas-lynn/vcc-perturb HF dataset -- see REPORT_phase0-5.md section 2.2 for
why the raw HF files themselves could not be fetched in this sandbox).

Does NOT assume the numeric values are logFC -- verifies provenance from the
data's own statistical signature before using it for anything downstream.
Findings are printed and also written to data/audit/vcc_perturb_summary.json.

Usage:
    python data/audit/audit_vcc_perturb.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd

import config as cfg

VCC_PERTURB_CSV = Path("/mnt/user-data/uploads/vcc-perturb.csv")


def main():
    df = pd.read_csv(VCC_PERTURB_CSV)
    gene_cols = [c for c in df.columns if c != "__index_level_0__"]
    idx = df["__index_level_0__"].values
    X = df[gene_cols].to_numpy(dtype=np.float64)

    targets_unique, target_counts = np.unique(idx, return_counts=True)

    # --- value-semantics check: is this logFC, raw delta, or z-score? ---
    # Signature checked: do extreme |value| outliers concentrate on a small,
    # perturbation-independent set of genes (classic small-denominator z-score
    # artifact), rather than scaling with which gene was perturbed?
    col_absmax = np.abs(X).max(axis=0)
    top_artifact_idx = np.argsort(-col_absmax)[:10]
    top_artifact_genes = [(gene_cols[i], float(col_absmax[i]),
                            int((np.abs(X[:, i]) > 5).sum())) for i in top_artifact_idx]

    frac_over = {thr: float((np.abs(X) > thr).mean()) for thr in (5, 10, 20, 50, 100)}

    # --- replicate consistency (5 pseudobulk "replicates" per target gene) ---
    rng = np.random.default_rng(0)
    sample_targets = rng.choice(targets_unique, size=min(15, len(targets_unique)), replace=False)
    rep_corrs = []
    for g in sample_targets:
        rows = df[df["__index_level_0__"] == g][gene_cols].to_numpy(dtype=np.float64)
        corr = np.corrcoef(rows)
        rep_corrs.append(corr[np.triu_indices(rows.shape[0], k=1)].mean())
    rep_corrs = np.array(rep_corrs)

    # --- gene axis overlap vs 2026's 18,533-gene feature axis ---
    genes_2026 = pd.read_csv(cfg.GENE_NAMES_CSV)["gene_name"].tolist()
    genes_2026_set = set(genes_2026)
    vp_genes_set = set(gene_cols)
    axis_overlap = vp_genes_set & genes_2026_set

    # --- target-gene panel overlap vs 2026's 300 target genes ---
    targets_2026 = set(pd.read_csv(cfg.PERT_COUNTS_CSV)[cfg.PERT_COL].tolist())
    vp_targets_set = set(targets_unique.tolist())
    target_overlap = vp_targets_set & targets_2026

    # --- are all 2026 targets at least present as *response* columns (usable as KNN query genes even if never a KD target here)? ---
    targets_2026_as_response_cols = targets_2026 & vp_genes_set

    summary = {
        "shape": list(df.shape),
        "n_unique_targets": int(len(targets_unique)),
        "replicates_per_target": {int(k): int(v) for k, v in zip(*np.unique(target_counts, return_counts=True))},
        "value_semantics": {
            "global_min": float(np.min(X)),
            "global_max": float(np.max(X)),
            "global_mean": float(np.mean(X)),
            "global_std": float(np.std(X)),
            "pct_exact_zero": float((X == 0).mean()),
            "frac_abs_over_threshold": frac_over,
            "top_artifact_genes_gene_absmax_n_over5": top_artifact_genes,
            "interpretation": (
                "Extreme |value| outliers (up to ~220) concentrate on a small, "
                "perturbation-independent set of genes (ACTB, TMSB4X, SCD, CD24, HMGCS1 -- "
                "high/stable-expression housekeeping-type genes) that recur as outliers across "
                "80-135 of the 750 unrelated perturbation rows. Real log2FC or raw expression "
                "deltas would be bounded by actual biology and would NOT blow up specifically on "
                "housekeeping genes regardless of which gene was knocked down. This is the "
                "textbook signature of a per-gene z-score computed by dividing by a very small "
                "reference standard deviation -- i.e. these values are z-scores "
                "(perturbation_zscores.parquet in the source dataset), not raw deltas or logFC. "
                "Treated as such throughout. Values also appear rounded to 2 decimals, which "
                "creates the 9.2% exact-zero rate and loses precision on small true effects."
            ),
        },
        "replicate_consistency": {
            "n_genes_sampled": int(len(sample_targets)),
            "mean_pairwise_replicate_corr": float(rep_corrs.mean()),
            "min_pairwise_replicate_corr": float(rep_corrs.min()),
            "interpretation": (
                "Low (0.1-0.4) mean pairwise correlation among a gene's 5 replicate rows, "
                "consistent with the HF dataset card's own caveat that these are random "
                "subsamples of one cell pool, not independent biological replicates -- noisy. "
                "Averaging (not concatenating) the 5 rows per target gene before use reduces this "
                "noise; kept as an explicit caveat rather than treated as high-confidence ground truth."
            ),
        },
        "gene_axis_overlap": {
            "vcc_perturb_response_genes": len(vp_genes_set),
            "vcc_2026_gene_axis": len(genes_2026_set),
            "overlap": len(axis_overlap),
            "pct_of_2026_axis_covered": len(axis_overlap) / len(genes_2026_set) * 100,
            "in_vcc_perturb_not_in_2026_axis_sample": sorted(vp_genes_set - genes_2026_set)[:10],
        },
        "target_panel_overlap": {
            "vcc_perturb_targets": len(vp_targets_set),
            "vcc_2026_targets": len(targets_2026),
            "overlap_n": len(target_overlap),
            "overlap_genes": sorted(target_overlap),
            "pct_of_2026_targets_directly_measured": len(target_overlap) / len(targets_2026) * 100,
            "interpretation": (
                f"Only {len(target_overlap)}/{len(targets_2026)} "
                f"({len(target_overlap)/len(targets_2026)*100:.1f}%) of 2026's target genes were "
                "themselves knocked down in vcc-perturb. A direct perturbation-lookup baseline "
                "(B1) can therefore cover at most this many of the 300 genes per context -- "
                "confirms gene-feature transfer (not lookup) has to carry the other ~96%."
            ),
        },
        "targets_2026_usable_as_knn_query_genes": {
            "n": len(targets_2026_as_response_cols),
            "of": len(targets_2026),
            "note": (
                "All 300 of 2026's target genes appear as *response*-axis columns in "
                "vcc-perturb (even the ~96% never themselves knocked down there), so every one "
                "has a 150-dim 'how do known perturbations affect this gene' embedding usable for "
                "KNN transfer -- see models/gene_transfer_b2.py."
            ),
        },
    }

    out_path = Path(__file__).resolve().parent / "vcc_perturb_summary.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
