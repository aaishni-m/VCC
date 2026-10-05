"""
Confirmatory follow-up to validate_transform_b2.py: the initial 25-gene sweep found
T2_log1p_additive (beta=2.0) best on composite score, clearly ahead of T1_multiplicative, and
T3_quantile both weaker AND ~50x slower (also showing an unexplained sparsity anomaly -- see
REPORT addendum). This script re-runs T0/T1/T2 (cheap: <1s/call) across ALL 150 known vcc-perturb
genes with a refined beta/alpha grid around the initial winner, for a statistically firmer choice
before committing real compute to full-scale generation. T3 is dropped here, not for speed alone --
its 25-gene composite score was already the second-worst of the four and its distributional
behavior was flagged as suspicious.

Usage:
    python models/validate_transform_b2_confirm.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import anndata as ad

import config as cfg
from models.gene_transfer_b2 import load_response_matrix
from models.pseudobulk_reference import load_or_build
from models.transform_candidates import t0_null, t1_multiplicative, t2_log1p_additive
from models.validate_transform_b2 import align_z_to_context, score_candidate, VALIDATION_CONTEXT

N_SYNTH_CELLS = 120
CANDIDATES_CONFIRM = {
    "T0_null": (t0_null, {}),
    "T1_multiplicative": (t1_multiplicative, {"alpha": [1.0, 2.0, 3.0]}),
    "T2_log1p_additive": (t2_log1p_additive, {"beta": [1.5, 2.0, 2.5, 3.0]}),
}


def main():
    rng = np.random.default_rng(cfg.RANDOM_SEED + 1)  # different draw than the first sweep
    targets, gene_cols, R = load_response_matrix()
    adata = ad.read_h5ad(cfg.CONTEXT_FILES[VALIDATION_CONTEXT])
    context_genes = adata.var_names.to_numpy()
    X = adata.X.tocsr()
    ref = load_or_build(VALIDATION_CONTEXT)

    n_genes = len(targets)
    print(f"[confirm] scoring on ALL {n_genes} known vcc-perturb genes (T0/T1/T2 only) ...")
    rows = []
    for gi, gene in enumerate(targets):
        z_row = R[gi]
        z_full = align_z_to_context(z_row, gene_cols, context_genes)
        cell_idx = rng.choice(X.shape[0], size=N_SYNTH_CELLS, replace=False)
        batch = X[cell_idx]
        de_ctrl_idx = rng.choice(X.shape[0], size=N_SYNTH_CELLS, replace=False)
        de_ctrl = np.asarray(X[de_ctrl_idx].todense())

        for name, (fn, param_grid) in CANDIDATES_CONFIRM.items():
            if name == "T0_null":
                rows.append(score_candidate(name, fn, None, None, gene, z_full, batch, X,
                                             de_ctrl, context_genes, ref, rng))
                continue
            (param_name, values), = param_grid.items()
            for v in values:
                rows.append(score_candidate(name, fn, param_name, v, gene, z_full, batch, X,
                                             de_ctrl, context_genes, ref, rng))
        if (gi + 1) % 25 == 0:
            print(f"  ... {gi+1}/{n_genes} genes scored", flush=True)

    df = pd.DataFrame(rows)
    out_csv = cfg.PROJECT_ROOT / "experiments" / "b2_transform_confirmation.csv"
    df.to_csv(out_csv, index=False)

    agg = df.groupby(["candidate", "param"], dropna=False).agg(
        n_genes=("gene", "nunique"),
        mean_direction_agree=("direction_agreement_strong_genes", "mean"),
        std_direction_agree=("direction_agreement_strong_genes", "std"),
        mean_magnitude_corr=("magnitude_correlation_strong_genes", "mean"),
        mean_de_jaccard=("de_sig_jaccard_vs_z_derived_truth", "mean"),
        mean_de_dir_agree=("de_direction_agreement_on_overlap", "mean"),
        mean_synth_zero_frac=("synth_zero_frac", "mean"),
        mean_ctrl_zero_frac=("ctrl_zero_frac", "mean"),
    ).reset_index()
    agg["composite_score"] = agg[["mean_direction_agree", "mean_magnitude_corr", "mean_de_jaccard"]].mean(axis=1, skipna=True)
    agg = agg.sort_values("composite_score", ascending=False)
    agg_path = cfg.PROJECT_ROOT / "experiments" / "b2_transform_confirmation_summary.csv"
    agg.to_csv(agg_path, index=False)
    print(agg.to_string(index=False))
    print(f"\n[confirm] wrote {out_csv} and {agg_path}")
    best = agg.iloc[0]
    print(f"[confirm] BEST on full 150-gene set: {best['candidate']} (param={best['param']}) "
          f"composite_score={best['composite_score']:.4f}")


if __name__ == "__main__":
    main()
