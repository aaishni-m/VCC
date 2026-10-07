"""
B2 -- full submission generator.

WHAT B2 ACTUALLY IS, after honest validation (see REPORT_phase0-5.md's Addenda 2-3 for the full
trail): four independent gene-specific transfer methods (perturbation-response-column KNN,
real-coexpression KNN, baseline-expression correlation, PLS regression on both embeddings and their
concatenation) were tried and NONE showed above-chance discrimination on a corrected, genuinely
zero-shot-honest leave-one-out validation. So B2 does NOT predict a gene-specific effect for target
genes vcc-perturb never measured -- there is no validated basis for that in this session's data.
What it does instead, and why each piece is still a real, defensible improvement over B0:
  1. Effect profile, per target gene:
       - 13/300 2026 target genes that vcc-perturb DID measure: their REAL averaged z-score profile.
         Not a prediction -- real data (still carrying the flagged, untested H1-vs-2026-cell-line
         transfer assumption).
       - The other 287/300: the MEAN response profile across all 150 known knockdowns -- not a
         transfer prediction either, just the one non-gene-specific real structure in the data
         (knockdowns share a common average response shape), which is measurably closer to real
         perturbed profiles than predicting no effect at all (B0). This is, not coincidentally, the
         same construction as the official metric spec's own scaling baseline `b` (report S1.3), so
         B2 should be expected to roughly TIE the documented baseline on these 287 genes and only has
         a chance to beat it on the 13 real-overlap genes.
  2. Count generation: T2_log1p_additive (beta=1.5), the transformation validated in
     validate_transform_b2.py / validate_transform_b2_confirm.py against held-out real vcc-perturb
     genes (150-gene confirmatory sweep: composite score 0.365, clearly ahead of
     T1_multiplicative's best 0.345 and T0_null's 0.195).

STATED LIMITATION carried through, not hidden: the effect profile for a given target gene is
context-invariant (same profile applied in A, B, C; only the baseline cells and each context's own
pseudobulk sigma_g reference differ) -- vcc-perturb is a single (H1) cell line, so nothing available
locally lets a context-specific effect be learned or validated.

Chunking mirrors control_baseline.py exactly (same memory ceiling applies: a naive single-shot
120,000 x 18,533 build OOM-killed this session's ~7.8GB box for B0, and B2's per-cell multinomial
redraw is heavier per row, not lighter, than B0's plain resampling).

SINGLE-FILE OUTPUT (supersedes the old per-context-file / concat_on_disk approach): the official
submission format requires ONE file covering all three contexts, not per-context files. A first
attempt at this restructure used anndata's `concat_on_disk` over all 60 gene-chunks -- but that
still needs every source chunk AND the growing destination on disk simultaneously (same failure mode
noted in the original per-context-merge attempt: "needs the destination AND all three sources on disk
at once ... peaking past the session quota"), and was aborted mid-run once `du` showed the unmerged
chunk directory alone had already reached ~12GB for 2/3 contexts (~18GB projected for all 3) against
a ~29GB session disk allowance -- the subsequent merge would have needed roughly another ~15-17GB on
top of that, well past quota.

Fixed here by never writing per-chunk (or per-context) .h5ad files at all. The combined output file is
created ONCE up front (correct final shape, real obs/var, empty placeholder X), then each gene-chunk's
in-memory CSR block is streamed directly into that single file's resizable `X/data` / `X/indices` /
`X/indptr` HDF5 datasets as soon as it's computed, and discarded. Disk only ever holds one thing: the
one combined file, growing to its final ~14-15GB -- never a second near-full copy of the same data, so
peak disk usage is essentially just that final size, comfortably inside the sandbox's ~29GB allowance.

Usage:
    python models/generate_b2_submission.py              # B2 (unchanged, reproducible)
    python models/generate_b2_submission.py --model b4   # B4: profiles from gene_transfer_b4.py
    python models/generate_b2_submission.py --model b5   # B5: profiles from gene_transfer_b5.py
    python models/generate_b2_submission.py --model b5v2 # B5 v2: Zhu-thresholded, agreement-weighted (gene_transfer_b5v2.py)
    python models/generate_b2_submission.py --model b6   # B6: top-K of B5, sparse T4 transform (gene_transfer_b6.py)
    python models/generate_b2_submission.py --model b7   # B7: magnitude-preserving external direction (gene_transfer_b7.py)
"""
import argparse
import gc
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp
import h5py
import json

import config as cfg
from models.gene_transfer_b2 import load_response_matrix
from models.pseudobulk_reference import load_or_build
from models.transform_candidates import t2_log1p_additive, t4_sparse_thinning

CHUNK_GENES = 15          # matches control_baseline.py's chunk size
BETA_REAL = 1.5           # beta for the 13/300 genes with real vcc-perturb z-score profiles
BETA_MEAN = 0.0           # beta for the 287/300 genes using the mean-response fallback:
                          # applying the population-average z-score to genes that were never
                          # actually measured produces MSE raw ~3.6 (vs null ~1.0) because
                          # the signal is wrong for each individual gene. BETA_MEAN=0 degrades
                          # gracefully to B0 (control-cell resampling) for those genes --
                          # identical predicted expression, which is at the null floor but
                          # does not actively hurt the other metrics the way a false signal does.
                          # SUPERSEDED: gene_transfer_b4.py's LOO sweep (and b3_meta.json) show the mean
                          # profile at scale 1.0 beats null on MSE (0.121 vs 0.212) -- use --model b4.


def predict_all_target_z_profiles(context_pert_genes, targets, gene_cols, R):
    """
    STATUS: gene-specific KNN transfer (the thing "B2" was originally named for) does NOT validate
    once the embedding bug is fixed -- see gene_transfer_b2.py's corrected leave_one_out_validation
    and REPORT_phase0-5.md's correction note. Column-based (i.e. genuinely zero-shot-honest)
    embeddings score at chance (~0.50 discrimination) for every k tested, both centered and
    uncentered -- there is no validated basis for predicting a gene-SPECIFIC effect for the 287/300
    2026 targets vcc-perturb never measured.

    What IS used here, and why each piece is defensible on its own:
      - The 13/300 genes vcc-perturb DID measure: their REAL averaged z-score profile. Not a
        prediction -- real data. (Still carries the flagged, untested cell-line-transfer caveat:
        H1 vs the three 2026 lines.)
      - The other 287/300: the MEAN response profile across all 150 known knockdowns. This is not
        a "transfer prediction" either -- it is exactly the "predict the average perturbation effect
        for every gene" construction the official metric spec itself uses as its own scaling
        baseline `b` (report S1.3). Using it doesn't claim gene-specific signal; it only uses the
        one piece of real, non-gene-specific structure in vcc-perturb (that knockdowns share a
        common average response shape) instead of predicting no effect at all (B0), which the mean
        vs. zero comparison in gene_transfer_b2.py's validation showed is a much closer match to
        real perturbed profiles (cosine ~0.73 vs 0 by construction) even though, like B0, it cannot
        discriminate between different genes' distinct effects (both score at the ~0.50 chance floor
        on that axis by construction).
    Returns (n_context_targets, n_response_genes) z-score matrix.
    """
    known_set = {g: i for i, g in enumerate(targets)}
    mean_response = R.mean(axis=0)
    out = np.tile(mean_response, (len(context_pert_genes), 1))
    real_mask = np.zeros(len(context_pert_genes), dtype=bool)
    n_real = 0
    for qi, gene in enumerate(context_pert_genes):
        if gene in known_set:
            out[qi] = R[known_set[gene]]
            real_mask[qi] = True
            n_real += 1
    print(f"[B2-gen] target-gene z-profiles: {n_real} real (vcc-perturb overlap, beta={BETA_REAL}), "
          f"{len(context_pert_genes) - n_real} mean-response (beta={BETA_MEAN} -- no validated "
          f"gene-specific signal, degrades to B0 to avoid false-signal MSE penalty)")
    return out, real_mask


def align_z_matrix_to_context(z_matrix, vp_gene_cols, context_genes):
    """z_matrix: (n_target_genes, len(vp_gene_cols)) -> (n_target_genes, len(context_genes)), 0
    where vcc-perturb had no measurement for a context gene (no evidence of an effect -- the only
    defensible default, same choice made throughout the validation)."""
    vp_pos = {g: i for i, g in enumerate(vp_gene_cols)}
    col_idx = np.array([vp_pos.get(g, -1) for g in context_genes])
    out = np.zeros((z_matrix.shape[0], len(context_genes)), dtype=np.float64)
    have = col_idx >= 0
    out[:, have] = z_matrix[:, col_idx[have]]
    return out


def build_combined_obs(target_genes: list[str]) -> pd.DataFrame:
    """Builds the obs table for all 3 contexts concatenated in order -- computed directly (no
    dependency on any intermediate file) since the labeling is fully determined by target_genes,
    CELLS_PER_PERT and CHUNK_GENES.

    NOTE -- corrected row-naming bug found during this restructure: the original per-chunk code (both
    B0's control_baseline.py and this file's prior version) named each chunk's obs_names using the
    GENE-chunk start index (`start`, which only advances by CHUNK_GENES=15 per chunk) as if it were a
    cumulative ROW offset, e.g. `f"{context}_b2_{start + i:06d}"` for i in range(6000). Since each
    chunk actually has CELLS_PER_PERT*CHUNK_GENES=6000 rows but `start` only advances by 15 between
    chunks, consecutive chunks' obs_names ranges overlapped almost entirely (chunk at start=0 named
    rows 0-5999, chunk at start=15 named rows 15-6014, etc.) -- confirmed here via anndata's own
    "Observation names are not unique" warning once all rows were assembled into one table at once
    (previously invisible because each chunk lived in its own separate file). This affects only the
    row LABEL strings (obs_names / cell barcodes), never which target_gene or context a row is
    assigned (that comes from the categorical columns below, built independently of these labels, and
    is unaffected). Fixed by using the true cumulative row offset (start * CELLS_PER_PERT) per context.
    """
    pert_pieces, context_pieces, index_pieces = [], [], []
    for context in cfg.VAL_CONTEXTS:
        for start in range(0, len(target_genes), CHUNK_GENES):
            gene_chunk = target_genes[start:start + CHUNK_GENES]
            obs_pert = np.repeat(gene_chunk, cfg.CELLS_PER_PERT)
            pert_pieces.append(obs_pert)
            context_pieces.append(np.full(obs_pert.shape[0], context))
            row_offset = start * cfg.CELLS_PER_PERT  # true cumulative row count within this context
            index_pieces.extend(f"{context}_b2_{row_offset + i:06d}" for i in range(obs_pert.shape[0]))
    all_pert = np.concatenate(pert_pieces)
    all_context = np.concatenate(context_pieces)
    obs = pd.DataFrame(
        {cfg.PERT_COL: pd.Categorical(all_pert, categories=target_genes), cfg.CONTEXT_COL: all_context},
    )
    obs.index = index_pieces
    assert obs.index.is_unique, "obs_names still not unique after cumulative-offset fix"
    return obs


def init_combined_h5ad(dest_path: Path, obs: pd.DataFrame, var: pd.DataFrame, n_var: int) -> None:
    """Writes the real obs/var (and correct final shape) via anndata (cheap -- obs/var are tiny),
    then reopens the file and replaces the placeholder X/data,indices,indptr with empty *resizable*
    datasets that build_and_stream_all_chunks will grow incrementally."""
    n_obs = obs.shape[0]
    placeholder_X = sp.csr_matrix((n_obs, n_var), dtype=np.float32)
    shell = ad.AnnData(X=placeholder_X, obs=obs, var=var)
    shell.write_h5ad(dest_path)
    del shell, placeholder_X
    gc.collect()

    with h5py.File(dest_path, "r+") as f:
        xg = f["X"]
        assert dict(xg.attrs)["encoding-type"] == "csr_matrix"
        assert tuple(xg.attrs["shape"]) == (n_obs, n_var)
        del xg["data"], xg["indices"], xg["indptr"]
        xg.create_dataset("data", shape=(0,), maxshape=(None,), dtype=np.float32, chunks=(1_000_000,))
        xg.create_dataset("indices", shape=(0,), maxshape=(None,), dtype=np.int32, chunks=(1_000_000,))
        xg.create_dataset("indptr", shape=(1,), maxshape=(None,), dtype=np.int64, data=np.zeros(1, dtype=np.int64))


def append_csr_chunk(f: h5py.File, X_chunk: sp.csr_matrix, state: dict) -> None:
    """Appends one in-memory CSR chunk's data/indices/indptr onto the destination file's resizable
    X datasets, offsetting indptr by the running nnz cursor. Never holds more than one chunk's worth
    of data in memory beyond what the caller already built."""
    xg = f["X"]
    data_ds, idx_ds, indptr_ds = xg["data"], xg["indices"], xg["indptr"]

    n_new = X_chunk.data.shape[0]
    old_nnz = data_ds.shape[0]
    data_ds.resize((old_nnz + n_new,))
    data_ds[old_nnz:] = X_chunk.data.astype(np.float32, copy=False)
    idx_ds.resize((old_nnz + n_new,))
    idx_ds[old_nnz:] = X_chunk.indices.astype(np.int32, copy=False)

    old_len = indptr_ds.shape[0]
    tail = (X_chunk.indptr[1:].astype(np.int64) + state["nnz"])
    indptr_ds.resize((old_len + tail.shape[0],))
    indptr_ds[old_len:] = tail

    state["nnz"] += n_new
    state["rows"] += X_chunk.shape[0]


def load_saved_profiles(model, target_genes):
    """z-profiles saved by gene_transfer_b4.py / gene_transfer_b5.py (real rows for the 13 measured genes,
    model-specific profiles for the rest). All genes use BETA_REAL: each model's z-space scale was already
    fitted by its own LOO sweep."""
    meta = json.loads((cfg.PROJECT_ROOT / "models" / f"{model}_meta.json").read_text())
    assert meta["target_genes"] == [g.strip() for g in target_genes], f"{model}_meta.json target order differs"
    profiles = np.load(cfg.PROJECT_ROOT / "models" / f"{model}_profiles.npy").astype(np.float64)
    print(f"[{model.upper()}-gen] target-gene z-profiles: {sum(meta['is_real'])} real, "
          f"{len(target_genes) - sum(meta['is_real'])} modelled")
    return profiles, np.full(len(target_genes), BETA_REAL)


def main(model="b2"):
    cfg.OUTPUTS.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(cfg.RANDOM_SEED)

    print("[B2-gen] loading vcc-perturb response matrix ...")
    targets, gene_cols, R = load_response_matrix()

    target_genes = pd.read_csv(cfg.PERT_COUNTS_CSV)[cfg.PERT_COL].tolist()
    assert len(target_genes) == cfg.N_TARGET_GENES, (
        f"expected {cfg.N_TARGET_GENES} target genes in pert_counts.csv, got {len(target_genes)}"
    )

    # Gene axis is shared across A/B/C (verified separately) -- read it once from context A.
    a0 = ad.read_h5ad(cfg.CONTEXT_FILES["A"], backed="r")
    assert a0.shape[1] == cfg.N_GENES, f"unexpected gene count {a0.shape[1]}"
    context_genes = a0.var_names.to_numpy()
    var = a0.var.copy()
    var.index = pd.Index(context_genes)
    del a0
    for c in cfg.VAL_CONTEXTS[1:]:
        ac = ad.read_h5ad(cfg.CONTEXT_FILES[c], backed="r")
        assert np.array_equal(ac.var_names.to_numpy(), context_genes), (
            f"context {c} has a different gene axis than context A -- cannot share one var block"
        )
        del ac

    n_obs_total = len(cfg.VAL_CONTEXTS) * cfg.N_TARGET_GENES * cfg.CELLS_PER_PERT
    print(f"[B2-gen] building combined obs table ({n_obs_total} rows) ...")
    combined_obs = build_combined_obs(target_genes)
    assert combined_obs.shape[0] == n_obs_total

    if model in ("b4", "b5", "b5v2", "b6", "b7"):
        z_matrix_vp, gene_betas = load_saved_profiles(model, target_genes)
    else:
        z_matrix_vp, real_mask = predict_all_target_z_profiles(target_genes, targets, gene_cols, R)
        gene_betas = np.where(real_mask, BETA_REAL, BETA_MEAN)
    z_matrix_full = align_z_matrix_to_context(z_matrix_vp, gene_cols, context_genes)  # (300, 18533)

    transform = t4_sparse_thinning if model in ("b6", "b7") else t2_log1p_additive  # see gene_transfer_b6.py
    out_path = cfg.OUTPUTS / f"{model}_submission.h5ad"
    print(f"[B2-gen] initializing combined output file at {out_path} "
          f"(shape=({n_obs_total}, {cfg.N_GENES})) ...")
    init_combined_h5ad(out_path, combined_obs, var, cfg.N_GENES)

    state = {"nnz": 0, "rows": 0}
    with h5py.File(out_path, "r+") as f:
        for context in cfg.VAL_CONTEXTS:
            print(f"[B2-gen] building and streaming B2 predictions for context {context} ...", flush=True)
            adata = ad.read_h5ad(cfg.CONTEXT_FILES[context])
            assert adata.shape[1] == cfg.N_GENES, f"{context}: unexpected gene count {adata.shape[1]}"
            assert (adata.obs[cfg.PERT_COL] == cfg.CONTROL_LABEL).all(), (
                f"{context}: controls file contains non-control rows; B2 assumes 100% non-targeting inputs"
            )
            X_src = adata.X if sp.isspmatrix_csr(adata.X) else adata.X.tocsr()
            n_control = adata.n_obs

            ref = load_or_build(context)

            for start in range(0, len(target_genes), CHUNK_GENES):
                gene_chunk = target_genes[start:start + CHUNK_GENES]
                chunk_blocks = []
                for gi, gene in enumerate(gene_chunk):
                    idx = rng.choice(n_control, size=cfg.CELLS_PER_PERT, replace=True)
                    baseline_batch = X_src[idx]
                    z_full = z_matrix_full[start + gi]
                    synth = transform(baseline_batch, z_full, ref, rng, beta=gene_betas[start + gi])
                    chunk_blocks.append(synth.astype(np.float32))
                X_pred = np.vstack(chunk_blocks)
                X_chunk_csr = sp.csr_matrix(X_pred)
                append_csr_chunk(f, X_chunk_csr, state)
                del X_pred, chunk_blocks, X_chunk_csr
                gc.collect()

            del adata, X_src, ref
            gc.collect()
            print(f"  -> context {context} done. running total: {state['rows']} rows, "
                  f"{state['nnz']} stored entries", flush=True)

    assert state["rows"] == n_obs_total, f"row count mismatch: {state['rows']} != {n_obs_total}"

    # Structural verification pass (no re-derivation of randomness -- checks shape/contract validity).
    print("[B2-gen] verifying combined file structure ...", flush=True)
    with h5py.File(out_path, "r") as f:
        xg = f["X"]
        shape = tuple(xg.attrs["shape"])
        nnz = xg["indices"].shape[0]
        indptr = xg["indptr"][:]
        assert shape == (n_obs_total, cfg.N_GENES), f"final shape mismatch: {shape}"
        assert indptr.shape[0] == n_obs_total + 1, f"indptr length mismatch: {indptr.shape[0]}"
        assert indptr[-1] == nnz, f"indptr[-1]={indptr[-1]} != nnz={nnz}"
        assert np.all(np.diff(indptr) >= 0), "indptr is not monotonically non-decreasing"
        # Spot-check indices are in range across a sample of the index array (avoid loading all of it).
        sample_n = min(50_000_000, nnz)
        sample = xg["indices"][:sample_n]
        assert sample.min() >= 0 and sample.max() < cfg.N_GENES, "indices out of gene-axis range"
        print(f"[B2-gen] OK: shape={shape}, total stored entries={nnz} / cap {cfg.MAX_STORED_ENTRIES}, "
              f"indptr length={indptr.shape[0]} (n_obs+1), indices sample range within [0,{cfg.N_GENES})")

    print(f"[B2-gen] built 1 combined prediction file covering all {len(cfg.VAL_CONTEXTS)} contexts: {out_path}")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["b2", "b4", "b5", "b5v2", "b6", "b7"], default="b2")
    main(parser.parse_args().model)
