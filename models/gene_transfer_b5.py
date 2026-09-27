"""
B5 -- B4 plus gene-specific direction from external CRISPRi screens (Replogle 2022 K562 genome-wide,
Zhu 2025 CD4+ T cells), produced by kaggle/external_profiles.py (external_profiles.npz).

Why (see that job and the official vcc2026 metric spec, cell-eval2 docs/vcc2026_metrics):
  - pds_cosine ranks predicted EFFECTS by cosine with all 300 panel target genes removed; the mean-response
    baseline scores exactly 0.5 (scaled 0). Only gene-specific direction can move it.
  - On vcc-perturb (H1), the combined external direction alone discriminates at 0.755 (chance 0.5; B4 0.491
    with target genes removed) -- real transferable signal, but at its MSE-optimal size (x1) it is swamped by
    the shared mean response and discrimination barely moves.
  - The scaled expression-error member is clamped to [0, 1] (can't go negative), and the z-space
    fold-change-error proxy improves rather than degrades up to x10. So B5 down-weights the shared mean
    (MEAN_WEIGHT) and up-weights the external direction (EXT_MULT) -- defaults are the balanced point of the
    local sweep (disc proxy 0.657, NMAE proxy 0.922 vs B4 0.491 / 0.955). These are H1 proxies; the
    leaderboard is the real test.
  - The target gene's own row is excluded from every metric, so B4's self-knockdown is kept only because
    it is harmless.

Profiles for the 13 targets vcc-perturb measured stay their real rows (unchanged from B2/B4). Targets no
external source covers get MEAN_WEIGHT * mean.

Usage:
    python models/gene_transfer_b5.py [--mean-weight 0.5] [--ext-mult 10] [--npz path]
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

SOURCES = ["K562", "T_mean"]


def unit_rows(D):
    D = np.nan_to_num(D)
    norm = np.linalg.norm(D, axis=1, keepdims=True)
    return np.divide(D, norm, out=np.zeros_like(D), where=norm > 0)


def combined_direction(npz, prefix, self_cols=None):
    """Sum of unit-normalized source directions; the target's own gene zeroed (handled by self-knockdown)."""
    D = sum(unit_rows(npz[f"{prefix}{s}"].astype(np.float64)) for s in SOURCES)
    if self_cols is not None:
        have = self_cols >= 0
        D[np.where(have)[0], self_cols[have]] = 0.0
    return D


def main(mean_weight, ext_mult, npz_path):
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
    Dvp = combined_direction(npz, "Dvp_", vp_self)
    have = np.abs(Dvp).sum(1) > 0
    a = float((Dvp[have] * (R - loo_mu)[have]).sum() / (Dvp[have] ** 2).sum())

    # local report in the official pds geometry (panel target genes removed)
    keep = np.setdiff1d(np.arange(R.shape[1]), np.unique(vp_self))
    P = mean_weight * loo_mu[have] + ext_mult * a * Dvp[have]
    disc = discrimination_score(P[:, keep], R[have][:, keep])
    disc_b4 = discrimination_score(loo_mu[have][:, keep], R[have][:, keep])
    print(f"[B5] LS scale a={a:.3f}; on {have.sum()} vcc-perturb knockdowns: disc {disc_b4:.3f} (B4) -> {disc:.3f} "
          f"(mean x{mean_weight}, external x{ext_mult})")

    target_genes = pd.read_csv(cfg.PERT_COUNTS_CSV)[cfg.PERT_COL].str.strip().tolist()
    assert list(npz["targets"]) == target_genes, "external_profiles.npz target order differs from pert_counts.csv"
    self_cols = np.array([col.get(g, -1) for g in target_genes])
    D300 = combined_direction(npz, "D_", self_cols)
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
    np.save(out_dir / "b5_profiles.npy", profiles.astype(np.float32))
    counts = {s: source.count(s) for s in sorted(set(source))}
    (out_dir / "b5_meta.json").write_text(json.dumps({
        "target_genes": target_genes, "is_real": [s == "real" for s in source], "source": source,
        "mean_weight": mean_weight, "ext_mult": ext_mult, "ls_scale": a, "sources": SOURCES,
        "local_disc_b4": disc_b4, "local_disc_b5": disc, "counts": counts}, indent=2))
    print(f"[B5] profiles: {counts}; saved models/b5_profiles.npy and models/b5_meta.json")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mean-weight", type=float, default=0.5)
    p.add_argument("--ext-mult", type=float, default=10.0)
    p.add_argument("--npz", default=None, help="external_profiles.npz (default: search /kaggle/input, then CWD)")
    args = p.parse_args()
    npz = args.npz or next((str(x) for root in (Path("/kaggle/input"), Path.cwd()) if root.exists()
                            for x in root.rglob("external_profiles.npz")), None)
    if npz is None:
        sys.exit("external_profiles.npz not found; pass --npz")
    main(args.mean_weight, args.ext_mult, npz)
