"""
B5 v3 -- dense log2 fold-change transfer from external CRISPRi screens, emitted by thinning / Poisson addition.

Profile for target t: C x mean over the sources that cover t of that source's log2 fold change (Replogle K562
genome-wide, Zhu CD4+ T cells; kaggle/lfc_profiles.py), on the challenge's 18,533-gene axis. Genes a source did not
measure count as 0 (no change) in its row -- the same averaging the local benchmark scored. The target's own gene is
left unchanged, as in the benchmark (every metric excludes it). Targets neither source covers get no change.

Why (results/localbench_rpe1_v3_lfc.csv, official cell-eval2 vcc2026 scoring on real held-out RPE1 cells): dense
log-fold-change at C = 2 beat B5 on pds (0.696 vs 0.593) and jac (0.120 vs 0.094) at similar nmae (-0.116 vs
-0.101); C = 3 overshoots nmae (-0.278). B5's local fidelity (0.66) did not transfer to the leaderboard (-0.003),
so the comparison leaves fid out: 0.117 (C = 2) vs 0.104 (B5).

Unlike v1/v2, the 13 targets vcc-perturb measured (H1, z-scores) are NOT used: the benchmark only validated the
external-lfc route, and H1 z-scores would need the z-fold emitter, which scored poorly there.

Usage:
    python models/gene_transfer_b5v3.py [--c 2] [--npz path]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

import config as cfg


def build_profiles(K, T, c, self_cols):
    """K, T: (n_targets, n_genes) log2 FC, all-NaN rows where a source lacks the target. Returns c x source mean,
    self gene zeroed, and each target's source label."""
    stack = np.stack([K, T])                                   # (2, n, G)
    have = ~np.isnan(stack).all(axis=2)                         # (2, n)
    n_src = have.sum(0)
    total = np.nansum(stack, axis=0)
    prof = c * np.divide(total, n_src[:, None], out=np.zeros_like(total), where=n_src[:, None] > 0)
    ok = self_cols >= 0
    prof[np.where(ok)[0], self_cols[ok]] = 0.0
    source = np.where(have[0] & have[1], "K562+T", np.where(have[0], "K562", np.where(have[1], "T", "none")))
    return prof.astype(np.float32), source.tolist()


def main(c, npz_path):
    npz = np.load(npz_path, allow_pickle=True)   # our own vcc-lfc output; v1 of it stored names as objects
    target_genes = pd.read_csv(cfg.PERT_COUNTS_CSV)[cfg.PERT_COL].str.strip().tolist()
    assert list(npz["targets"]) == target_genes, "lfc_profiles.npz target order differs from pert_counts.csv"
    genes = npz["genes"].astype(str)
    gpos = {g: i for i, g in enumerate(genes)}
    self_cols = np.array([gpos.get(t, -1) for t in target_genes])
    prof, source = build_profiles(npz["K562"].astype(np.float64), npz["T"].astype(np.float64), c, self_cols)

    out_dir = cfg.PROJECT_ROOT / "models"
    np.save(out_dir / "b5v3_profiles.npy", prof)
    counts = {s: source.count(s) for s in sorted(set(source))}
    nz = (prof != 0).sum(1)
    (out_dir / "b5v3_meta.json").write_text(json.dumps({
        "target_genes": target_genes, "genes": genes.tolist(), "source": source, "c": c,
        "is_real": [False] * len(target_genes), "counts": counts,
        "median_changed_genes": float(np.median(nz)), "median_abs_log2fc": float(np.median(np.abs(prof[prof != 0])))},
        indent=1))
    print(f"[B5v3] c={c}; sources {counts}; median changed genes/target {np.median(nz):.0f}; "
          f"median |log2 fold| {np.median(np.abs(prof[prof != 0])):.3f}; saved models/b5v3_profiles.npy")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--c", type=float, default=2.0)
    p.add_argument("--npz", default=None, help="lfc_profiles.npz (default: search /kaggle/input, then CWD)")
    args = p.parse_args()
    npz = args.npz or next((str(x) for root in (Path("/kaggle/input"), Path.cwd()) if root.exists()
                            for x in root.rglob("lfc_profiles.npz")), None)
    if npz is None:
        sys.exit("lfc_profiles.npz not found; pass --npz")
    main(args.c, npz)
