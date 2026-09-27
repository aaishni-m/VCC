# Reproducing and submitting B2 (VCC 2026)

This sandbox hit two hard resource limits that require finishing this on a different machine:

1. **Network**: this sandbox's outbound network policy blocks `virtualcellchallenge.org`
   (confirmed via the CLI's own error and the proxy's status endpoint -- an org egress policy
   denial, not a bug). `vcc login`, `vcc datasets`, and `vcc submit` all need real network access.
2. **Memory**: `vcc prep` was killed by the OOM killer twice on this sandbox's ~7.8GB RAM (even
   after adding 4GB of swap, it still hit a combined memory+swap ceiling around ~10GB and was
   still climbing) trying to process the 14GB combined B2 file. It needs a machine with
   meaningfully more headroom -- 16GB+ RAM recommended.

Everything else is done and verified: B2's predictions are generated, packaged into ONE combined
file (not per-context, per the official requirement), and passed every LOCAL structural check this
sandbox could run.

## What's included here

- `config.py`, `models/generate_b2_submission.py`, `models/gene_transfer_b2.py`,
  `models/pseudobulk_reference.py`, `models/transform_candidates.py` -- the exact code that
  produced the validated B2 predictions (deterministic, fixed seed 0 -- running this on any
  machine with the same controls bundle reproduces a byte-identical file; total stored entries
  matched exactly, 1,820,583,430, between the original per-context run and this restructured
  single-file run).

## Steps to finish on a machine with more RAM and normal internet access

1. Get the controls bundle onto that machine: `context_A.h5ad`, `context_B.h5ad`,
   `context_C.h5ad`, `gene_names.csv`, `pert_counts.csv`, `manifest.json`. (You already have a
   copy at `Downloads\controls` on your linked Windows desktop, per this session's device info --
   confirm it matches the `vcc2026-val-1` panel: `n_genes: 18533`, `n_constructs: 300`,
   `cells_per_pert: 400`, `control_cells: 18400` per context.)

2. Install dependencies: Python 3.10+, then
   `pip install numpy pandas anndata scipy h5py`

3. Install the VCC CLI: `uv tool install vcc-cli` (or see
   https://vcc-cli-wiki.virtualcellchallenge.org/ if you don't have `uv`).

4. Set the controls path and regenerate B2 (from the directory containing this README, with
   `config.py` and `models/` alongside it):

   ```
   export VCC_CONTROLS_DIR=/path/to/controls      # or set it as a Windows env var
   python models/generate_b2_submission.py
   ```

   This writes `outputs/b2_submission.h5ad` (~14.6GB, shape (360000, 18533), 1,820,583,430 stored
   entries -- verify these numbers match before proceeding; they should, exactly, since generation
   is deterministic).

5. Package it into the official `.vcc` format -- dry-run first:

   ```
   vcc prep outputs/b2_submission.h5ad -g /path/to/controls/gene_names.csv \
     --perts /path/to/controls/pert_counts.csv -o outputs/b2_submission.vcc --dry-run
   ```

   Read its output carefully. **Do not change any prediction values to force this to pass** -- if
   it reports something genuinely invalid, stop and share the exact message rather than guessing a
   fix. If the dry run looks clean, drop `--dry-run` to actually write `b2_submission.vcc`.

6. Log in: get a fresh API key from https://virtualcellchallenge.org/app/credentials (generating a
   new one revokes any existing key -- this session already generated one during this attempt,
   named "vcc2026-sandbox-cli", which you may want to revoke or just let a new one replace).
   Then: `export VCC_TOKEN=<the key>` (or `vcc login --token-stdin`). Never put the key in a
   committed file.

7. **Before submitting**, confirm this summary matches what you intend to submit:

   ```
   Model:                  B2 (hybrid: real vcc-perturb response for 13/300 overlap target genes +
                            validated cross-knockdown mean-response profile for the other 287/300;
                            T2 log1p-additive count transform, beta=1.5)
   File:                   outputs/b2_submission.vcc
   Number of cells:        360,000  (300 perturbations x 400 cells/pert x 3 contexts)
   Number of perturbations: 300  (same 300 target genes in every context)
   Number of contexts:     3  (A, B, C -- one combined file)
   Number of genes:        18,533
   Validation status:      Passed all local structural checks in the original sandbox (correct
                            shape, indptr/index integrity, gene-axis alignment vs controls,
                            360,000 unique obs_names, density 1,820,583,430 / 4,750,000,000 cap).
                            `vcc prep`'s own authoritative validation must still be run on this
                            machine (step 5) -- treat ITS result as final, not the sandbox checks.
   Submission description: "B2 hybrid: real vcc-perturb knockdown response profiles for 13/300
                            2026 target genes with vcc-perturb overlap; validated cross-knockdown
                            mean-response profile (matches the official metric spec's own scaling
                            baseline construction) for the remaining 287/300; T2 log1p-additive
                            count transform (beta=1.5, validated via held-out confirmatory sweep)."
   ```

   Also worth re-checking before you commit a daily submission slot: your account's one prior
   submission ("target-gene-knockdown-baseline-1AM", Overall -0.3026) looked, from leaderboard
   pattern-matching, like a null/no-effect-style baseline similar in kind to B0 -- not proof, but a
   reasoned signal that B2 (which uses real signal for 13/300 genes) has a real chance to beat it.
   That was inference, not certainty; the only way to know for sure is this submission's own score.

8. Submit: `vcc submit outputs/b2_submission.vcc -m "B2 hybrid" -d "<description above>" --wait`

   Record the entry ID, timestamp, and status/scores it reports.

## What NOT to do

- Don't modify prediction values (the X matrix) to make `vcc prep` or the scorer happy -- if
  something's invalid, that's a stop-and-report situation, not a data-massaging one.
- Don't submit anything beyond this single B2 file (no B0, no other variants) per the original
  instructions.
- Don't submit until you've reviewed the block in step 7 yourself, since I can't gate the final
  click on this machine.
