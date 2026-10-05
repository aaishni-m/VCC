# VCC 2026 — Project Summary

_Last updated: 2026-09-27_

## Challenge Overview
- 300 target genes to predict knockdown effects for, across a ~18,533-gene genome.
- Validation contexts A/B/C (submitted phase); held-out final contexts D/E/F release ~Oct 22 2026.
- Scored via official `cell_eval2` `vcc2026` metric spec (pds_cosine, expr_mse_unbiased_capped_norm, de_wilcoxon_* metrics).

## Models Built

**B0 — null baseline**: no-op, already submitted. Scored Overall -0.3026.

**B1 — rejected**: lookup-table idea; only 13/300 target genes overlap vcc-perturb's 150 known H1 genes. Not enough coverage to justify on its own.

**B2 — current real submission**:
- Real z-profile for 13/300 overlap genes (`BETA_REAL = 1.5`).
- `BETA_MEAN = 0.0` for the other 287/300 — deliberate: those genes get zero shift (pure null), after testing showed a flat nonzero beta made things worse (raw MSE ~3.6 vs null ~1.0).
- Submitted via `vcc` CLI (`vcc prep` → `vcc login` → `vcc submit`).

**B3 — evaluated, not adopted**:
- Rebuilt co-expression KNN model for the 287 unknown genes, tested against a rigorous LOO + discrimination-score gate (requires ≥2×SE margin over baseline before trusting a new signal).
- Result: discrimination margin 0.0247 < required 0.0471 → gate failed → final betas for the 287 genes collapsed back to 0.0, identical to B2. No improvement justified yet.

## Bugs Fixed
1. **Gene-identity column bug** (`phase3_coexpr.py`, `diag_names.py`): vcc-perturb.csv's true gene-identity column is `__index_level_0__`, not the first CSV column. Fixed by loading via that column and grouping/averaging the 5 noisy replicate rows per gene.
2. **Gene-order misalignment** (`phase3_coexpr.py`): `gene_names.csv` has a real header row; reading with `header=None` shifted every gene index by +1. Fixed by sourcing gene order from `adata.var_names` directly (cross-checked equal across contexts A/B/C).

## Infra Fixes
- `vcc prep` RAM warning (needs ~19GB vs. ~15GB available): fixed with an 8GB WSL swapfile.
- `vcc login` keyring failure: fixed via `export VCC_TOKEN=<token>` (never pasted into chat).

## External Datasets Investigated (to fill the 287-gene coverage gap)

| Source | Coverage of our 287 | Verdict |
|---|---|---|
| Replogle et al. 2022 (K562/RPE1 essential screens) | Check script ready, not yet run | Pending — `models/check_replogle_overlap.py` |
| Tahoe-100M | N/A | Ruled out — drug/compound perturbations only, not genetic knockdown |
| X-Atlas/Orion (Xaira) | Near-complete gene coverage (18,903 genes) | Ruled out — 126–521GB, no smaller summary file found, CC-BY-NC-SA-4.0 non-commercial license |
| VIPerturb-seq genome-wide (Satija Lab) | 124/300 filtered | Available, Zenodo 18460279, ~3.6GB filtered file — usable but noisy (~42 cells/gene median) |
| VIPerturb-seq pilot | — | Not useful — mTOR-focused feasibility test only |
| **Zhu et al. 2025 (primary CD4+ T cell Perturb-seq)** | **297/300** | **Most promising** — MIT licensed, public S3 (`s3://genome-scale-tcell-perturb-seq/`), DE-stats-only file `GWCD4i.DE_stats.h5ad` is 16.8GB (vs. >1.6TB raw); a 7.3MB `DE_stats.suppl_table.csv` may already be enough |
| FlexPlex (Satija) | — | Not useful — only 3 genes (TSC1, RPTOR, CD55) |

## Current Status / Open Items
- B2 is the live submission; same result as the earlier B0-adjacent baseline in terms of the 287 unmeasured genes.
- Next lever: pull Zhu et al.'s `DE_stats.suppl_table.csv` (7.3MB) first, check its gene coverage/schema, and if it holds up, run the 297 known genes through the same LOO + discrimination-score gate used for B3 before adopting anything.
- Unresolved from earlier: whether `-0.0383` was a real submitted score, what `outputs/b2_split_beta.vcc` is, and whether `vcc submit` for `b2_submission_fresh.vcc` was actually run.

## Files
- `models/gene_transfer_b2.py` / `models/generate_b2_submission.py` — current submitted model.
- `models/gene_transfer_b3.py`, `models/b3_profiles.npy`, `models/b3_betas.npy`, `models/b3_meta.json` — B3 (evaluated, not adopted).
- `models/phase3_coexpr.py` — co-expression KNN model, bug-fixed.
- `models/validate_phase3_discrimination.py` — standalone discrimination-score validation.
- `models/check_replogle_overlap.py` — Replogle overlap checker (not yet run).
