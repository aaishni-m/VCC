"""
Builds a per-context reference "pseudobulk noise model" (mu_g, sigma_g in
log1p(CP10K) space) from the REAL 2026 control single cells, by resampling
many small pseudobulks -- deliberately mirroring the vcc-perturb dataset
card's own stated construction ("50 control pseudobulks of 15 cells each")
so that a z-score computed against this reference is on a comparable kind of
scale to vcc-perturb's own z-scores, even though we cannot access vcc-perturb's
actual H1 control pseudobulks to calibrate against directly (that file,
pseudobulks.h5ad, was not part of what could be shared into this sandbox --
see REPORT addendum for why this is a named assumption, not a proven fact).

Used two ways:
  1. As the DENOMINATOR (sigma_g) for inverting a z-score profile into a
     target expression shift: implied_shift_g = z_g * sigma_g (candidate T1/T2).
  2. As the reference baseline distribution (mu_g) that a synthetic
     perturbed cell's expression is defined relative to.

This only ever touches the released 18,400 control cells per context (no
perturbed cells exist locally, by design of the challenge) -- entirely
memory-safe: 200 pseudobulks x 18,533 genes x float32 is ~15MB, nothing like
the multi-GB full-context arrays that needed chunking in control_baseline.py.

Usage:
    python models/pseudobulk_reference.py    # builds and caches for all 3 contexts
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import anndata as ad

import config as cfg

CACHE_DIR = cfg.PROJECT_ROOT / "data" / "audit" / "pseudobulk_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

CELLS_PER_PSEUDOBULK = 15   # matches the vcc-perturb dataset card's own convention
N_PSEUDOBULKS = 300         # more than the card's 50, since it's nearly free here and stabilizes sigma_g


def build_reference(context: str, rng: np.random.Generator, n_pseudobulks=N_PSEUDOBULKS,
                     cells_per_pb=CELLS_PER_PSEUDOBULK):
    """Returns dict with mu_log1p_cp10k (G,), sigma_log1p_cp10k (G,), mu_raw_mean_count (G,),
    and control_cells (the loaded AnnData, backed='r' is NOT used here since we need repeated
    random row access -- one context's 18400x18533 CSR at float32 is ~1-1.5GB dense-equivalent
    but stays sparse in memory, well within budget)."""
    adata = ad.read_h5ad(cfg.CONTEXT_FILES[context])
    X = adata.X.tocsr()
    n_cells = X.shape[0]
    n_genes = X.shape[1]

    pb_log1p_cp10k = np.zeros((n_pseudobulks, n_genes), dtype=np.float64)
    for i in range(n_pseudobulks):
        idx = rng.choice(n_cells, size=cells_per_pb, replace=False)
        summed = np.asarray(X[idx].sum(axis=0)).ravel()
        total = summed.sum()
        cp10k = summed / total * 1e4 if total > 0 else summed
        pb_log1p_cp10k[i] = np.log1p(cp10k)

    mu = pb_log1p_cp10k.mean(axis=0)
    sigma = pb_log1p_cp10k.std(axis=0)
    sigma_floor = np.maximum(sigma, 1e-3)  # avoid div-by-~0 blowups when we later divide, mirroring
                                            # (and capping) the exact artifact we diagnosed in vcc-perturb's own z-scores

    # also keep the raw single-cell mean count per gene (context's true baseline, no resampling noise)
    # for candidates that operate directly on raw counts rather than log1p(CP10K).
    raw_mean_count = np.asarray(X.mean(axis=0)).ravel()
    raw_var_count = np.asarray(X.multiply(X).mean(axis=0)).ravel() - raw_mean_count ** 2
    raw_var_count = np.maximum(raw_var_count, raw_mean_count)  # Poisson floor: var >= mean

    return {
        "context": context,
        "gene_names": adata.var_names.to_numpy(),
        "mu_log1p_cp10k": mu,
        "sigma_log1p_cp10k": sigma,
        "sigma_log1p_cp10k_floored": sigma_floor,
        "raw_mean_count": raw_mean_count,
        "raw_var_count": raw_var_count,
        "n_pseudobulks": n_pseudobulks,
        "cells_per_pseudobulk": cells_per_pb,
        "median_library_size": float(np.median(np.asarray(X.sum(axis=1)).ravel())),
    }


def cache_path(context):
    return CACHE_DIR / f"pb_reference_{context}.npz"


def load_or_build(context, rng=None):
    p = cache_path(context)
    if p.exists():
        d = np.load(p, allow_pickle=True)
        return {k: d[k] for k in d.files}
    rng = rng or np.random.default_rng(cfg.RANDOM_SEED)
    ref = build_reference(context, rng)
    np.savez_compressed(p, **ref)
    return ref


if __name__ == "__main__":
    rng = np.random.default_rng(cfg.RANDOM_SEED)
    for context in cfg.VAL_CONTEXTS:
        print(f"[pb-ref] building pseudobulk reference for context {context} "
              f"({N_PSEUDOBULKS} pseudobulks x {CELLS_PER_PSEUDOBULK} cells) ...", flush=True)
        ref = build_reference(context, rng)
        np.savez_compressed(cache_path(context), **ref)
        print(f"  median_library_size={ref['median_library_size']:.0f}  "
              f"mean(sigma_log1p_cp10k)={ref['sigma_log1p_cp10k'].mean():.4f}  "
              f"pct genes with sigma<1e-3 (floored)={(ref['sigma_log1p_cp10k']<1e-3).mean()*100:.1f}%")
    print("[pb-ref] done, cached under data/audit/pseudobulk_cache/")
