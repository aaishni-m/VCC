"""
B0 -- control-null baseline (Phase 4).

This is NOT a perturbation model. For every one of the 300 target genes in a
context, it predicts "no effect": the post-perturbation profile is the
context's own empirical non-targeting-control distribution. Concretely, we
resample (with replacement, since 18,400 pooled control cells must cover
300 x 400 = 120,000 output cells) CELLS_PER_PERT raw-count control cells per
target gene, per context, and assign them the target_gene / context labels
the submission schema requires.

This corresponds exactly to the "control-pasting" degenerate submission
analyzed in section 1 of the official vcc2026 metrics spec: it is expected to
score ~0 (chance) on pds_cosine, ~1 (no-skill) on expr_mse_unbiased_capped_norm,
and ~0 on all four DE-based metrics. It exists to establish the floor other
models must beat -- not as a competitive entry.

MEMORY NOTE (Phase 17): a full context prediction is 300 x 400 = 120,000
cells at this context's native density (~6,000 genes/cell), i.e. ~7*10^8
stored nonzeros -- a naive single-shot fancy-index build needs >6GB for the
CSR data+indices arrays alone and was observed to OOM-kill on this session's
~7.8GB box. So each context is built in small gene-chunks (CHUNK_GENES target
genes = CHUNK_GENES*400 cells per chunk), each chunk written to its own small
h5ad and immediately freed, then all chunks for that context are merged with
anndata's on-disk concatenation (`concat_on_disk`).

DISK NOTE (Phase 17, continued): the three resulting per-context files are
each several GB uncompressed. A further cross-context merge into one
section-7-compliant upload file needs the destination AND all three sources
on disk at once, which exceeded this sandbox's writable-disk allowance when
attempted (~4x a single context's size, peaking past the session quota even
though `df` reports headroom -- the quota is a separate per-session
allowance, not the filesystem's nominal size). This script therefore stops at
three validated per-context files; see submission/validate_submission.py's
`validate_multi(...)`, which checks the same section-7 contract across the
three files without ever concatenating them, and README.md for how to do the
final single-file merge on a host with more disk.

Usage:
    python models/control_baseline.py
"""
import gc
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import scipy.sparse as sp
import anndata as ad
import pandas as pd

import config as cfg

CHUNK_GENES = 15  # 15 * 400 = 6,000 cells/chunk -> tens of MB per chunk, not GB


def build_context_baseline(context: str, rng: np.random.Generator, chunk_dir: Path) -> Path:
    """Resample control cells into a compliant per-context prediction, chunked to disk."""
    adata = ad.read_h5ad(cfg.CONTEXT_FILES[context])
    assert adata.shape[1] == cfg.N_GENES, f"{context}: unexpected gene count {adata.shape[1]}"
    assert (adata.obs[cfg.PERT_COL] == cfg.CONTROL_LABEL).all(), (
        f"{context}: controls file contains non-control rows; B0 assumes 100% non-targeting"
    )

    n_control = adata.n_obs
    var = adata.var.copy()
    gene_names = adata.var_names.to_numpy()
    X_src = adata.X if sp.isspmatrix_csr(adata.X) else adata.X.tocsr()

    target_genes = pd.read_csv(cfg.PERT_COUNTS_CSV)[cfg.PERT_COL].tolist()
    assert len(target_genes) == cfg.N_TARGET_GENES, (
        f"expected {cfg.N_TARGET_GENES} target genes in pert_counts.csv, got {len(target_genes)}"
    )

    chunk_paths = []
    for start in range(0, len(target_genes), CHUNK_GENES):
        gene_chunk = target_genes[start:start + CHUNK_GENES]
        n_out = len(gene_chunk) * cfg.CELLS_PER_PERT
        idx = rng.integers(0, n_control, size=n_out)
        X_pred = X_src[idx].astype(np.float32)

        obs_pert = np.repeat(gene_chunk, cfg.CELLS_PER_PERT)
        obs = pd.DataFrame(
            {cfg.PERT_COL: pd.Categorical(obs_pert, categories=target_genes), cfg.CONTEXT_COL: context}
        )
        obs.index = [f"{context}_pred_{start + i:06d}" for i in range(n_out)]

        chunk_adata = ad.AnnData(X=X_pred, obs=obs, var=var)
        chunk_adata.var_names = gene_names
        chunk_path = chunk_dir / f"{context}_chunk_{start:04d}.h5ad"
        chunk_adata.write_h5ad(chunk_path)
        chunk_paths.append(chunk_path)
        del X_pred, chunk_adata
        gc.collect()

    del adata, X_src
    gc.collect()

    context_path = cfg.OUTPUTS / f"baseline_B0_{context}.h5ad"
    ad.experimental.concat_on_disk([str(p) for p in chunk_paths], str(context_path))
    for p in chunk_paths:
        p.unlink()
    return context_path


def main():
    cfg.OUTPUTS.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(cfg.RANDOM_SEED)

    total_nnz = 0
    with tempfile.TemporaryDirectory(prefix="b0_chunks_") as tmp:
        chunk_dir = Path(tmp)
        per_context_paths = []
        for context in cfg.VAL_CONTEXTS:
            print(f"[B0] building control-null baseline for context {context} ...", flush=True)
            out_path = build_context_baseline(context, rng, chunk_dir)
            chk = ad.read_h5ad(out_path, backed="r")
            print(f"  -> {out_path}  shape={chk.shape}", flush=True)
            del chk
            import h5py
            with h5py.File(out_path, "r") as f:
                total_nnz += f["X"]["indices"].shape[0]
            per_context_paths.append(out_path)

    print(f"[B0] built {len(per_context_paths)} per-context prediction files.")
    print(f"[B0] total stored entries across contexts: {total_nnz} / cap {cfg.MAX_STORED_ENTRIES}")
    print("[B0] NOTE: these are 3 separate per-context files, not yet merged into one upload file -- see module docstring.")
    return per_context_paths


if __name__ == "__main__":
    main()
