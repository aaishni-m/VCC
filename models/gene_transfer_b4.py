"""
B4 -- fixes two gaps in B2/B3 for the 287/300 target genes vcc-perturb never knocked down.

1. Mean-response profile is USED, not zeroed. gene_transfer_b3.py's own LOO beta sweep (b3_meta.json)
   shows the flat mean profile beats null on raw MSE (0.118 at scale 1.0 vs 0.207 at 0). B3's gate
   only established that coexpr does not beat the mean -- it never showed the mean loses to zero, yet
   beta_287 was collapsed to 0.0. B4 keeps the mean (B3's discrimination verdict against coexpr stands).

2. Self-knockdown. In B2/B3 the 287 targets' OWN expression is left at control level in their own
   perturbed cells. vcc-perturb shows the target's own column is by far its strongest signal (median
   self-z -3.14 across the 150 measured knockdowns, 72% below -2, vs off-target |z| mostly < 0.9).
   B4 sets each unmeasured target's own entry to the median self-z of the known knockdowns --
   calibrated from data, not chosen by hand.

Both are validated here with the same LOO harness as B3 (each of the 150 known genes held out; mean
and median self-z computed from the other 149 only), scored on raw MSE and discrimination_score.

Outputs models/b4_profiles.npy (300, n_vcc_perturb_genes) z-profiles in vcc-perturb gene space and
models/b4_meta.json; generate_b2_submission.py --model b4 consumes them.

Usage:
    python models/gene_transfer_b4.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

import config as cfg
from models.gene_transfer_b2 import load_response_matrix, discrimination_score, cosine_row

OUT_PROFILES = cfg.PROJECT_ROOT / "models" / "b4_profiles.npy"
OUT_META = cfg.PROJECT_ROOT / "models" / "b4_meta.json"
SCALE_SWEEP = [0.0, 0.3, 0.6, 0.8, 1.0, 1.2, 1.5]


def mean_plus_self_profile(R_ref, self_z_ref, self_col, scale):
    """Mean response of the reference knockdowns (times `scale`) with the target's own column set to
    the reference median self-z. self_col=None -> no self override (plain mean)."""
    prof = scale * R_ref.mean(axis=0)
    if self_col is not None:
        prof[self_col] = np.median(self_z_ref)
    return prof


def loo(targets, gene_cols, R):
    col = {g: i for i, g in enumerate(gene_cols)}
    self_cols = np.array([col[g] for g in targets])
    self_z = R[np.arange(len(targets)), self_cols]
    n = len(targets)

    results = {}
    for with_self in (False, True):
        key = "mean+self" if with_self else "mean"
        mse_row, preds_at = [], {}
        for s in SCALE_SWEEP:
            preds = np.zeros_like(R)
            for i in range(n):
                others = np.arange(n) != i
                preds[i] = mean_plus_self_profile(R[others], self_z[others],
                                                  self_cols[i] if with_self else None, s)
            mse_row.append(float(np.mean((preds - R) ** 2)))
            preds_at[s] = preds
        best = SCALE_SWEEP[int(np.argmin(mse_row))]
        results[key] = {
            "mse_by_scale": dict(zip(map(str, SCALE_SWEEP), mse_row)),
            "best_scale": best,
            "best_mse": min(mse_row),
            "discrimination_at_best": discrimination_score(preds_at[best], R),
            "raw_cosine_at_best": float(cosine_row(preds_at[best], R).mean()),
        }
    results["null_mse"] = float(np.mean(R ** 2))
    results["median_self_z"] = float(np.median(self_z))
    return results


def build_profiles(target_genes, targets, gene_cols, R, scale):
    """(300, len(gene_cols)) z-profiles: real row for measured genes, mean*scale + self-knockdown otherwise.
    Returns (profiles, is_real mask)."""
    col = {g: i for i, g in enumerate(gene_cols)}
    known = {g: i for i, g in enumerate(targets)}
    self_z = R[np.arange(len(targets)), [col[g] for g in targets]]
    out = np.zeros((len(target_genes), R.shape[1]))
    is_real = np.zeros(len(target_genes), dtype=bool)
    for qi, gene in enumerate(target_genes):
        if gene in known:
            out[qi] = R[known[gene]]
            is_real[qi] = True
        else:
            out[qi] = mean_plus_self_profile(R, self_z, col.get(gene), scale)
    return out, is_real


if __name__ == "__main__":
    targets, gene_cols, R = load_response_matrix()
    print(f"[B4] response matrix: {R.shape[0]} known knockdowns x {R.shape[1]} response genes")

    r = loo(targets, gene_cols, R)
    print(f"\n[B4] LOO over {len(targets)} known genes (null MSE = {r['null_mse']:.4f})")
    for key in ("mean", "mean+self"):
        row = r[key]
        print(f"  {key:<10} MSE by scale: " +
              "  ".join(f"{s}:{v:.4f}" for s, v in row["mse_by_scale"].items()))
        print(f"  {'':<10} best scale={row['best_scale']}  MSE={row['best_mse']:.4f}  "
              f"discrimination={row['discrimination_at_best']:.3f}  raw_cosine={row['raw_cosine_at_best']:.4f}")

    scale = r["mean+self"]["best_scale"]
    target_genes = pd.read_csv(cfg.PERT_COUNTS_CSV)[cfg.PERT_COL].str.strip().tolist()
    profiles, is_real = build_profiles(target_genes, targets, gene_cols, R, scale)
    missing_self = [g for g, real in zip(target_genes, is_real) if not real and g not in set(gene_cols)]
    np.save(OUT_PROFILES, profiles.astype(np.float32))
    with open(OUT_META, "w") as f:
        json.dump({"target_genes": target_genes, "is_real": is_real.tolist(), "mean_scale": scale,
                   "genes_without_self_column": missing_self, "loo": r}, f, indent=2)
    print(f"\n[B4] profiles: {is_real.sum()} real, {(~is_real).sum()} mean*{scale} + self-knockdown "
          f"(median self-z {r['median_self_z']:.2f}); {len(missing_self)} targets have no vcc-perturb "
          f"column for the self override")
    print(f"[B4] saved {OUT_PROFILES} and {OUT_META}")
