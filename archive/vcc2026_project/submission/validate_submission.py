"""
Fails loudly if a submission does not match the section-7 upload/scoring
contract of the official vcc2026 metrics spec (cell-eval2, docs/vcc2026_metrics/
vcc2026-metrics.md, retrieved 2026-09-07). Implemented independently here
because that spec's checks are documented precisely enough to reproduce, but
the official `vcc prep` CLI was not installed/run in this session (see
README caveats). Treat this as a strong local pre-check, not a substitute for
running the real Arc tooling before an actual submission.

Supports two shapes:
  - validate(path): a single h5ad already holding all contexts (the real
    upload format -- "one file, all three contexts").
  - validate_multi(paths): several h5ad files, each holding one context's
    predictions (what this sandbox's control_baseline.py produces, because
    disk constraints here prevented the final cross-context merge -- see
    models/control_baseline.py's docstring). Checks the identical section-7
    contract context-by-context without ever materializing a combined matrix.

Both stream `.X` through HDF5 in row-chunks rather than loading it whole, so
a several-GB context file can be checked well under this session's ~7.8GB RAM.

Usage:
    python submission/validate_submission.py path/to/submission.h5ad
    python submission/validate_submission.py path/to/ctx_A.h5ad path/to/ctx_B.h5ad path/to/ctx_C.h5ad
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import h5py
import numpy as np
import pandas as pd
import anndata as ad

import config as cfg

ROW_CHUNK = 20000


class ValidationError(Exception):
    pass


def _check_obs_var(adata_backed, gene_names_ref, expected_genes, errors, warnings):
    if adata_backed.n_vars != cfg.N_GENES:
        errors.append(f"expected {cfg.N_GENES} genes in .var, got {adata_backed.n_vars}")
    elif list(adata_backed.var_names) != gene_names_ref:
        errors.append("gene order in .var does not exactly match gene_names.csv")

    for col in (cfg.PERT_COL, cfg.CONTEXT_COL):
        if col not in adata_backed.obs.columns:
            errors.append(f"missing required obs column '{col}'")
    if errors:
        return

    obs = adata_backed.obs
    n_ctrl_rows = int((obs[cfg.PERT_COL] == cfg.CONTROL_LABEL).sum())
    if n_ctrl_rows > 0:
        errors.append(
            f"found {n_ctrl_rows} rows labelled '{cfg.CONTROL_LABEL}' -- "
            f"control rows are model inputs only and must not appear in the prediction"
        )

    contexts_present = sorted(obs[cfg.CONTEXT_COL].unique().tolist())
    for context in contexts_present:
        sub = obs[obs[cfg.CONTEXT_COL] == context]
        genes_present = set(sub[cfg.PERT_COL].unique())
        extra = genes_present - expected_genes
        missing = expected_genes - genes_present
        if extra:
            errors.append(f"context {context}: {len(extra)} unexpected perturbations, e.g. {list(extra)[:5]}")
        if missing:
            errors.append(f"context {context}: {len(missing)} missing perturbations, e.g. {list(missing)[:5]}")
        counts = sub[cfg.PERT_COL].value_counts()
        bad = counts[counts != cfg.CELLS_PER_PERT]
        if len(bad) > 0:
            errors.append(
                f"context {context}: {len(bad)} perturbations do not have exactly "
                f"{cfg.CELLS_PER_PERT} cells (e.g. {bad.index[0]}={bad.iloc[0]})"
            )
    return contexts_present


def _check_X_streaming(path, errors, warnings):
    """Stream the CSR `.X` in row-chunks: nnz cap, non-negativity, integrality, per-cell total cap."""
    with h5py.File(path, "r") as f:
        xg = f["X"]
        is_csr = xg.attrs.get("encoding-type") == "csr_matrix" if hasattr(xg, "attrs") else False
        if not is_csr:
            warnings.append("X is not stored as a sparse CSR group -- cannot cheaply verify without densifying")
            return
        n_obs = xg.attrs["shape"][0]
        indptr = xg["indptr"][:]
        nnz = xg["data"].shape[0]

        if nnz > cfg.MAX_STORED_ENTRIES:
            errors.append(f"stored entries {nnz} exceed cap {cfg.MAX_STORED_ENTRIES}")

        saw_negative = saw_noninteger = saw_nonfinite = False
        over_cap_cells = 0
        for start in range(0, n_obs, ROW_CHUNK):
            end = min(start + ROW_CHUNK, n_obs)
            lo, hi = indptr[start], indptr[end]
            block = xg["data"][lo:hi]
            if block.size:
                if not saw_negative and np.any(block < 0):
                    saw_negative = True
                if not saw_noninteger and not np.allclose(block, np.round(block)):
                    saw_noninteger = True
                if not saw_nonfinite and not np.all(np.isfinite(block)):
                    saw_nonfinite = True
            # per-row sums via indptr slicing within the block
            row_bounds = indptr[start:end + 1] - lo
            for i in range(end - start):
                row_sum = block[row_bounds[i]:row_bounds[i + 1]].sum()
                if row_sum > cfg.MAX_COUNTS_PER_CELL:
                    over_cap_cells += 1

        if saw_negative:
            errors.append("negative values found in .X -- counts must be non-negative")
        if saw_noninteger:
            errors.append(".X contains non-integer values -- submit raw counts, not normalized/log data")
        if saw_nonfinite:
            errors.append(".X contains non-finite values (NaN/Inf)")
        if over_cap_cells > 0:
            errors.append(f"{over_cap_cells} cells exceed max_counts_per_cell={cfg.MAX_COUNTS_PER_CELL}")
        return nnz


def _validate_one_file(path, gene_names_ref, expected_genes):
    errors, warnings = [], []
    adata_backed = ad.read_h5ad(path, backed="r")
    contexts_present = _check_obs_var(adata_backed, gene_names_ref, expected_genes, errors, warnings)
    shape = list(adata_backed.shape)
    del adata_backed
    nnz = None
    if not errors:
        nnz = _check_X_streaming(path, errors, warnings)
    return {
        "path": str(path),
        "shape": shape,
        "contexts_present": contexts_present,
        "nnz": nnz,
        "errors": errors,
        "warnings": warnings,
    }


def validate(path: str, expected_contexts=None) -> dict:
    """Single file holding one or more contexts."""
    expected_contexts = expected_contexts or cfg.VAL_CONTEXTS
    gene_names_ref = pd.read_csv(cfg.GENE_NAMES_CSV)["gene_name"].tolist()
    expected_genes = set(pd.read_csv(cfg.PERT_COUNTS_CSV)[cfg.PERT_COL].tolist())
    result = _validate_one_file(path, gene_names_ref, expected_genes)
    missing_contexts = set(expected_contexts) - set(result["contexts_present"] or [])
    if missing_contexts:
        result["errors"].append(f"missing context(s) entirely: {sorted(missing_contexts)}")
    result["passed"] = len(result["errors"]) == 0
    result["n_errors"] = len(result["errors"])
    return result


def validate_multi(paths, expected_contexts=None) -> dict:
    """Several files, each expected to hold one (or more) contexts; union must cover all expected contexts."""
    expected_contexts = expected_contexts or cfg.VAL_CONTEXTS
    gene_names_ref = pd.read_csv(cfg.GENE_NAMES_CSV)["gene_name"].tolist()
    expected_genes = set(pd.read_csv(cfg.PERT_COUNTS_CSV)[cfg.PERT_COL].tolist())

    per_file = [_validate_one_file(p, gene_names_ref, expected_genes) for p in paths]
    covered = set()
    for r in per_file:
        covered.update(r["contexts_present"] or [])
    missing_contexts = set(expected_contexts) - covered
    total_errors = sum(r["n_errors"] if "n_errors" in r else len(r["errors"]) for r in per_file)
    if missing_contexts:
        total_errors += 1

    return {
        "files": per_file,
        "contexts_covered": sorted(covered),
        "missing_contexts": sorted(missing_contexts),
        "total_stored_entries": sum(r["nnz"] or 0 for r in per_file),
        "n_errors": total_errors,
        "passed": total_errors == 0,
    }


if __name__ == "__main__":
    targets = sys.argv[1:] if len(sys.argv) > 1 else [str(cfg.OUTPUTS / "submission_B0.h5ad")]

    if len(targets) == 1:
        report = validate(targets[0])
        print(f"Validating: {report['path']}")
        print(f"Shape: {report['shape']}  contexts present: {report['contexts_present']}")
        if report["warnings"]:
            print("WARNINGS:")
            for w in report["warnings"]:
                print(f"  - {w}")
        if report["passed"]:
            print("PASSED: submission is structurally compliant with the vcc2026 section-7 contract.")
        else:
            print(f"FAILED: {report['n_errors']} error(s):")
            for e in report["errors"]:
                print(f"  - {e}")
            sys.exit(1)
    else:
        report = validate_multi(targets)
        print(f"Validating {len(targets)} per-context files as one logical submission:")
        for r in report["files"]:
            status = "OK" if not r["errors"] else f"{len(r['errors'])} error(s)"
            print(f"  - {r['path']}: shape={r['shape']} contexts={r['contexts_present']} nnz={r['nnz']} [{status}]")
            for e in r["errors"]:
                print(f"      ERROR: {e}")
            for w in r["warnings"]:
                print(f"      WARNING: {w}")
        print(f"Contexts covered: {report['contexts_covered']}  missing: {report['missing_contexts']}")
        print(f"Total stored entries across files: {report['total_stored_entries']} / cap {cfg.MAX_STORED_ENTRIES}")
        if report["passed"]:
            print("PASSED: all files structurally comply with the vcc2026 section-7 contract "
                  "(still need to be merged into one file before actual upload).")
        else:
            print(f"FAILED: {report['n_errors']} total error(s) across files/coverage.")
            sys.exit(1)
