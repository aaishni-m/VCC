# vcc2026

Working repo for the 2026 Virtual Cell Challenge (Arc Institute). See
`REPORT_phase0-5.md` for the full competition-intelligence / dataset /
strategy writeup this repo was built to support.

## Layout

```
vcc2026/
  config.py                    schema constants (genes, contexts, cell counts) sourced from
                                manifest.json + the official vcc2026 metrics spec
  requirements.txt
  data/
    raw/                       (empty here -- see "Data" below)
    audit/                     Phase 1 audit script + summary JSON
  models/
    control_baseline.py        B0: control-null baseline
  submission/
    validate_submission.py     section-7 contract checker (single-file or per-context files)
  experiments/
    experiment_results.csv     Phase 19 experiment log
  outputs/                     generated h5ad files land here (not committed -- see Disk note)
```

## Data

The 2026 controls bundle (`context_A/B/C.h5ad`, `gene_names.csv`,
`manifest.json`, `pert_counts.csv`) lives on the user's machine at
`Downloads/controls` and is staged into this sandbox at
`/mnt/user-data/uploads/controls/` (see `config.CONTROLS_DIR`). It is **not**
copied into this repo.

The 2025 `nicolas-lynn/vcc-perturb` dataset on Hugging Face could **not** be
downloaded in this session: this sandbox's network egress policy blocks
`huggingface.co` outright (proxy returns 403 on CONNECT). What we know about
it here comes from Hugging Face's web dataset-card pages via a text-only
fetch, not from loading the actual parquet/h5ad files -- see the "Blocked:
2025 dataset access" section of the report for what that does and doesn't
let us claim, and the options for unblocking it.

## Disk / memory notes (Phase 17)

This sandbox has ~7.8GB RAM and a writable-disk allowance well under what
`df` reports as the filesystem's nominal size. A single context's B0
prediction (120,000 cells x 18,533 genes, ~30% dense) is already ~2*10^8-7*10^8
stored nonzeros -- multiple GB as float32+int32 CSR. Concretely, in this repo:

- `control_baseline.py` builds each context in small gene-chunks (15 genes x
  400 cells = 6,000 cells/chunk) and merges chunks with anndata's
  `concat_on_disk`, never holding a full context's resampled matrix and its
  source in memory at once. A naive single-shot build OOM-killed at ~4.6GB
  resident.
- It stops at three per-context files rather than also producing the single
  cross-context upload file section 7 requires: merging three ~5.5GB files
  into a fourth needs roughly 4x a context's size on disk at once, which hit
  this session's disk allowance. `validate_submission.py`'s `validate_multi`
  checks the identical contract across the three files without concatenating
  them. Producing the final single file is a mechanical step best done on a
  host with more disk (or by writing a streaming HDF5-level merge that
  deletes each source right after folding it in).
- `submission/validate_submission.py` streams `.X` through HDF5 in
  20,000-row chunks rather than loading it whole, for the same reason.

Generated `outputs/*.h5ad` files were deleted after validation to keep this
session's disk usage low; re-run `python models/control_baseline.py` to
regenerate them (~3 minutes).
