"""
Candidate z-score -> single-cell raw-count transformations for B2.

None of these is assumed correct -- validate_transform_b2.py fits each one's
single free scale parameter and scores all of them on held-out vcc-perturb
genes before any of them touches the 2026 controls for real. See that file
and REPORT_phase0-5.md's B2-generation addendum for the selection.

All four operate on a baseline batch of REAL control single cells (raw
integer counts, one context) and a z-score profile aligned to that context's
full 18,533-gene axis (0 where vcc-perturb had no measurement for a gene --
"no evidence of an effect" is the only defensible default there), and return
a same-shape array of non-negative integer counts. Three of them work by
building a target per-gene proportion vector per cell and drawing the final
integer counts from Multinomial(cell's own original total, target
proportions) -- this guarantees non-negative integers summing to the cell's
original total by construction (so it can never violate the section-7
integer/non-negativity/count-cap contract), and deliberately preserves each
cell's own total library size, because a z-score computed on normalized data
carries no information about absolute library-size shift -- there's nothing
in the input to condition a library-size change on, so inventing one would
be exactly the kind of unvalidated assumption this task said not to make.
This is stated as an explicit modeling limitation, not hidden.

T0_null       -- ignore z entirely (this is what B0 already does; kept here
                 as the in-harness floor every real candidate must beat).
T1_multiplicative -- fold_g = 2^(alpha * z_g) applied directly to each cell's
                 own raw count for gene g, then renormalized to proportions
                 and redrawn. `alpha` is the one free parameter.
T2_log1p_additive -- shift each cell's own log1p(CP10K) value for gene g by
                 beta * z_g * sigma_g, where sigma_g is this context's
                 pseudobulk-resampled reference std (pseudobulk_reference.py)
                 -- i.e. a direct algebraic inversion of a z-score built the
                 same way vcc-perturb's card describes ("z = shift / std over
                 many pseudobulks"), NOT assumed to be on an identical scale
                 to vcc-perturb's own (H1) reference std, hence still a fitted
                 `beta` rather than beta=1.
T3_quantile   -- rank-preserving: map each cell's percentile within this
                 context's own empirical per-gene count distribution through
                 a normal-quantile shift of gamma * z_g, then invert through
                 the empirical distribution to read off a new count. Respects
                 the true empirical shape (including zero-inflation) instead
                 of assuming a multiplicative or log-normal form.
"""
import numpy as np
import scipy.sparse as sp


def _renorm_and_redraw(baseline_counts: sp.csr_matrix, target_props: np.ndarray, rng: np.random.Generator):
    """target_props: (n_cells, n_genes) non-negative, need not sum to 1. Redraw each cell's
    original total from Multinomial(total, normalized proportions). Returns dense int32 array
    (validation-scale only -- a handful of genes-of-interest x a few hundred cells, never the
    full 18,533-gene axis at production scale, where this is called per gene-chunk exactly like
    control_baseline.py's chunking, so it stays small there too)."""
    totals = np.asarray(baseline_counts.sum(axis=1)).ravel().astype(np.int64)
    target_props = np.clip(target_props, 0, None)
    row_sums = target_props.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    props = target_props / row_sums
    out = np.zeros(props.shape, dtype=np.int32)
    for i in range(props.shape[0]):
        out[i] = rng.multinomial(totals[i], props[i])
    return out


def t0_null(baseline_counts, z_profile, ref, rng, **kw):
    return np.asarray(baseline_counts.todense(), dtype=np.int32)


def t1_multiplicative(baseline_counts, z_profile, ref, rng, alpha=1.0):
    dense = np.asarray(baseline_counts.todense(), dtype=np.float64)
    fold = np.power(2.0, alpha * z_profile)  # (n_genes,), broadcasts over cells
    target = dense * fold[np.newaxis, :]
    return _renorm_and_redraw(baseline_counts, target, rng)


def t2_log1p_additive(baseline_counts, z_profile, ref, rng, beta=1.0):
    dense = np.asarray(baseline_counts.todense(), dtype=np.float64)
    totals = dense.sum(axis=1, keepdims=True)
    totals[totals == 0] = 1.0
    cp10k = dense / totals * 1e4
    log_baseline = np.log1p(cp10k)
    sigma = ref["sigma_log1p_cp10k_floored"]
    shift = beta * z_profile[np.newaxis, :] * sigma[np.newaxis, :]
    log_target = log_baseline + shift
    cp10k_target = np.expm1(np.clip(log_target, 0, None))
    return _renorm_and_redraw(baseline_counts, cp10k_target, rng)


def build_sorted_reference(all_context_counts):
    """Precompute once per context (expensive: sorts N_control x n_genes) and pass the result as
    `sorted_ref` into t3_quantile for every subsequent call -- avoids re-sorting ~18,400 x n_genes
    on every single test gene / hyperparameter combination.

    float32, not float64: at full-context scale (~18,400 x 18,533) a float64 dense copy is ~2.7GB,
    and np.sort needs a second buffer that size -- OOM-killed this session's ~7.8GB box the first
    time this was tried. float32 halves both, matching X's own on-disk dtype exactly (counts are
    small integers, no precision lost)."""
    dense = np.asarray(all_context_counts.todense(), dtype=np.float32)
    dense.sort(axis=0)  # in-place: avoids allocating a second full-size buffer on top of `dense`
    return dense


def t3_quantile(baseline_counts, z_profile, ref, rng, gamma=1.0, sorted_ref=None, all_context_counts=None):
    """
    sorted_ref: precomputed via build_sorted_reference(all_context_counts) -- pass this (not
    all_context_counts) when calling this many times against the same context, or a fresh sort
    is done from all_context_counts (or the batch itself) as a fallback.
    """
    from scipy.stats import norm

    dense = np.asarray(baseline_counts.todense(), dtype=np.float64)
    n_cells, n_genes = dense.shape
    out = np.zeros_like(dense)

    if sorted_ref is None:
        ref_pool = np.asarray(all_context_counts.todense(), dtype=np.float64) if all_context_counts is not None else dense
        sorted_ref = np.sort(ref_pool, axis=0)  # (N_ref, n_genes)
    n_ref = sorted_ref.shape[0]

    for g in range(n_genes):
        col_ref = sorted_ref[:, g]
        if col_ref[-1] == col_ref[0]:  # constant gene (e.g. always 0) -- nothing to shift
            out[:, g] = dense[:, g]
            continue
        # percentile of each cell's value within the reference distribution (average-rank style)
        ranks = np.searchsorted(col_ref, dense[:, g], side="right")
        pct = np.clip(ranks / n_ref, 1.0 / (2 * n_ref), 1 - 1.0 / (2 * n_ref))
        z_pct = norm.ppf(pct)
        shifted_pct = norm.cdf(z_pct + gamma * z_profile[g])
        new_rank = np.clip((shifted_pct * n_ref).astype(int), 0, n_ref - 1)
        out[:, g] = col_ref[new_rank]

    return _renorm_and_redraw(baseline_counts, out, rng)


def t4_sparse_thinning(baseline_counts, z_profile, ref, rng, beta=1.0, max_fold=20.0):
    """Edits ONLY genes with a non-zero z; every other gene keeps the real control cell's exact counts.

    Why: T1-T3 redraw every gene of every cell from a multinomial. That keeps each gene's mean but turns
    many 1-counts into 0s, which the official Wilcoxon DE test reads as a decrease -- measured on context A,
    a z=0 (no-effect) T2 run already gets ~7,200 of 9,929 tested genes called significant vs the real
    controls, while untouched resampled controls get 0. Those artifact calls swamp the DE metrics.

    Fold per gene f_g = exp(beta * z_g * sigma_g) -- the same log1p(CP10K) shift T2 applies, read as a fold.
      f < 1: binomial thinning, count -> Binomial(count, f)      (a real knockdown also adds zeros)
      f > 1: count + Poisson((f - 1) * expected count), with expected count = cell total x the gene's
             control share, so zero cells can gain reads too.
    Library sizes change only by the edited genes' net change. Returns dense int32 counts."""
    out = np.asarray(baseline_counts.todense(), dtype=np.int64)
    genes = np.flatnonzero(z_profile)
    if genes.size == 0:
        return out.astype(np.int32)
    fold = np.clip(np.exp(beta * z_profile[genes] * ref["sigma_log1p_cp10k_floored"][genes]), 1 / max_fold, max_fold)
    share = ref["raw_mean_count"] / ref["raw_mean_count"].sum()
    totals = out.sum(axis=1, keepdims=True)
    down, up = genes[fold < 1], genes[fold > 1]
    if down.size:
        out[:, down] = rng.binomial(out[:, down], fold[fold < 1][np.newaxis, :])
    if up.size:
        out[:, up] += rng.poisson((fold[fold > 1] - 1)[np.newaxis, :] * totals * share[up][np.newaxis, :])
    return out.astype(np.int32)


CANDIDATES = {
    "T0_null": (t0_null, {}),
    "T1_multiplicative": (t1_multiplicative, {"alpha": [0.5, 1.0, 1.5, 2.0]}),
    "T2_log1p_additive": (t2_log1p_additive, {"beta": [0.5, 1.0, 2.0, 4.0]}),
    # T3's grid is deliberately smaller than T1/T2's: its per-gene Python loop (transform_candidates.t3_quantile)
    # costs ~12s/call at validation scale vs <1s for T1/T2, so the full validation sweep's wall-clock is
    # dominated by however many T3 points are tested. Two points (rather than one) still lets the held-out
    # scoring show whether performance is sensitive to gamma in this range before committing to a value.
    "T3_quantile": (t3_quantile, {"gamma": [0.5, 1.5]}),
}
