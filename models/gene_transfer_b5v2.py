"""
B5 v2 -- B5 (v1, tag b5-v1) with two changes to how the external direction is built:

  1. Zhu confidence shrinkage: Zhu's rows are z-scores (logFC / SE); every entry is soft-thresholded,
     z -> sign(z) * max(|z| - ZHU_SOFT, 0), before unit-normalising, so genes the T-cell screen is unsure
     about drop out instead of adding noise. K562 is left as is (thresholding it did not help locally).
  2. Agreement weighting: per target and gene, the combined direction is multiplied by
     agreement ** AGREE_P, agreement = |sum_s u_s| / sum_s |u_s| (1 when K562 and Zhu agree in sign,
     0 when they cancel). Targets only one source covers are unchanged.

Everything else (sources K562 + Zhu T_mean, MEAN_WEIGHT, EXT_MULT, LS scale, real rows for the 13 measured
targets, self-knockdown, T2 transform) is identical to v1, and --zhu-soft 0 --agree-p 0 reproduces v1 exactly.

Local check (vcc-perturb, 146 knockdowns, pds geometry, held-out LS scale): disc 0.630 (v1) -> 0.641 (v2),
inside one SE (~0.024). Both changes target which genes are called significant (fid / jac), which this check
cannot see; the local cell-eval2 bench is the real test.

Usage:
    python models/gene_transfer_b5v2.py [--zhu-soft 1] [--agree-p 1] [--mean-weight 0.5] [--ext-mult 10] [--npz path]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

import config as cfg
from models.gene_transfer_b2 import discrimination_score, load_response_matrix
from models.gene_transfer_b5 import SOURCES, unit_rows


def combined_direction(npz, prefix, zhu_soft, agree_p, self_cols=None):
    """Unit-normalised sources (Zhu soft-thresholded), summed, scaled by cross-source sign agreement; the
    target's own gene zeroed afterwards, as in v1 (handled by self-knockdown)."""
    U = []
    for s in SOURCES:
        X = np.nan_to_num(npz[f"{prefix}{s}"].astype(np.float64))
        if s.startswith("T_") and zhu_soft > 0:
            X = np.sign(X) * np.maximum(np.abs(X) - zhu_soft, 0)
        U.append(unit_rows(X))
    D = sum(U)
    if agree_p > 0:
        A = sum(np.abs(u) for u in U)
        agreement = np.divide(np.abs(D), A, out=np.zeros_like(D), where=A > 0)
        D = D * agreement ** agree_p
    if self_cols is not None:
        have = self_cols >= 0
        D[np.where(have)[0], self_cols[have]] = 0.0
    return D


def main(zhu_soft, agree_p, mean_weight, ext_mult, npz_path):
    npz = np.load(npz_path)
    targets, gene_cols, R = load_response_matrix()
    assert list(npz["vp_genes"]) == list(gene_cols), "external_profiles.npz gene axis differs from vcc-perturb"
    col = {g: i for i, g in enumerate(gene_cols)}
    n = len(targets)
    vp_self = np.array([col[g] for g in targets])
    self_z = R[np.arange(n), vp_self]
    mu = R.mean(axis=0)

    # least-squares scale of the external direction, fitted on the vcc-perturb knockdowns sources cover
    loo_mu = (R.sum(0) - R) / (n - 1)
    Dvp = combined_direction(npz, "Dvp_", zhu_soft, agree_p, vp_self)
    have = np.abs(Dvp).sum(1) > 0
    a = float((Dvp[have] * (R - loo_mu)[have]).sum() / (Dvp[have] ** 2).sum())

    # local report in the official pds geometry (panel target genes removed)
    keep = np.setdiff1d(np.arange(R.shape[1]), np.unique(vp_self))
    P = mean_weight * loo_mu[have] + ext_mult * a * Dvp[have]
    disc = discrimination_score(P[:, keep], R[have][:, keep])
    print(f"[B5v2] LS scale a={a:.3f}; on {have.sum()} vcc-perturb knockdowns: disc {disc:.3f} "
          f"(zhu_soft {zhu_soft}, agree_p {agree_p}, mean x{mean_weight}, external x{ext_mult})")

    target_genes = pd.read_csv(cfg.PERT_COUNTS_CSV)[cfg.PERT_COL].str.strip().tolist()
    assert list(npz["targets"]) == target_genes, "external_profiles.npz target order differs from pert_counts.csv"
    self_cols = np.array([col.get(g, -1) for g in target_genes])
    D300 = combined_direction(npz, "D_", zhu_soft, agree_p, self_cols)
    known = {g: i for i, g in enumerate(targets)}
    profiles = np.zeros((len(target_genes), R.shape[1]))
    source = []
    for qi, gene in enumerate(target_genes):
        if gene in known:
            profiles[qi] = R[known[gene]]
            source.append("real")
            continue
        profiles[qi] = mean_weight * mu + ext_mult * a * D300[qi]
        if self_cols[qi] >= 0:
            profiles[qi, self_cols[qi]] = np.median(self_z)
        source.append("mean+external" if D300[qi].any() else "mean")

    out_dir = cfg.PROJECT_ROOT / "models"
    np.save(out_dir / "b5v2_profiles.npy", profiles.astype(np.float32))
    counts = {s: source.count(s) for s in sorted(set(source))}
    (out_dir / "b5v2_meta.json").write_text(json.dumps({
        "target_genes": target_genes, "is_real": [s == "real" for s in source], "source": source,
        "zhu_soft": zhu_soft, "agree_p": agree_p, "mean_weight": mean_weight, "ext_mult": ext_mult,
        "ls_scale": a, "sources": SOURCES, "local_disc": disc, "counts": counts}, indent=2))
    print(f"[B5v2] profiles: {counts}; saved models/b5v2_profiles.npy and models/b5v2_meta.json")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--zhu-soft", type=float, default=1.0)
    p.add_argument("--agree-p", type=float, default=1.0)
    p.add_argument("--mean-weight", type=float, default=0.5)
    p.add_argument("--ext-mult", type=float, default=10.0)
    p.add_argument("--npz", default=None, help="external_profiles.npz (default: search /kaggle/input, then CWD)")
    args = p.parse_args()
    npz = args.npz or next((str(x) for root in (Path("/kaggle/input"), Path.cwd()) if root.exists()
                            for x in root.rglob("external_profiles.npz")), None)
    if npz is None:
        sys.exit("external_profiles.npz not found; pass --npz")
    main(args.zhu_soft, args.agree_p, args.mean_weight, args.ext_mult, npz)
