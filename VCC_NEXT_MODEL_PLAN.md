# VCC 2026 — Next model plan: direction-aware cross-cell perturbation transfer

Started 22 Oct 2026, branch `next-model`. B5 (tag `b5-v1`) and B5v3 stay unchanged as baselines.
Final deadline: **5 Nov 2026 23:59 UTC**, two final entries per UTC day, only the last counts, no leaderboard
until winners are announced (rules page, read 22 Oct).

## 1. Audit: reported findings vs what the code and data show

Sources of the reported findings: `D:\VCC docs\VCC_2026_review.md` (21 Oct) and
`D:\VCC docs\VCC_2026_code_review.md` (22 Oct). `VCC_2026_RESEARCH_AUDIT.md` does not exist anywhere on this
machine (searched the repo, D:\VCC docs, Downloads, Documents). The code review's emission diagnostic (§5) is an
unfilled placeholder (`RESULTS_PLACEHOLDER`): no numbers to verify.

| Reported finding | Verified how | Verdict |
|---|---|---|
| B5 overall ≈ 0.128 | `SNAPSHOT_B6.md`, submission log | Confirmed: 0.1283 |
| B5 raw pds ≈ 0.72 | Not stored locally; the review derives it as 0.5 + 0.487 × span | Plausible, unverified (needs a token-authenticated `vcc status`, not run) |
| K562 vs Zhu median cosine ≈ 0.047 over 267 shared targets | Recomputed from `lfc_profiles.npz` and `external_profiles.npz` | Confirmed: 0.051 (log2FC, genes both measure), 0.040 (B5's z-score inputs), IQR 0.012–0.079 |
| B5 = generic H1 response + transferred K562/T-cell effects | `models/gene_transfer_b5.py` | Confirmed: `0.5·mean(H1 z) + 10·a·[unit(K562) + unit(T)]`, a = 2.51 |
| B5 is only weakly "amplified" (median shift 0.033 log1p) | Recomputed: T2 shift 1.5·z·σ_g on context A, 287 modelled targets, 9,844 expressed genes | Confirmed: median 0.026 log1p (0.038 log2), p99 0.149. The research log's "×10 amplification" description was wrong |
| B5v3 c=2 raised pds but hurt nmae and reach vs c=1 | Leaderboard: B5v3 (c=2) vs B5 (not a c=1 run). Local RPE1 bench: c=2 *improved* nmae (−0.116 vs −0.197) and reach (−0.114 vs −0.145) over c=1 | Partly wrong as worded: no c=1 run was ever submitted. True that B5v3 (c=2) lost nmae/reach vs **B5** on the leaderboard, and that the RPE1 bench could not see that cost |
| RPE1 bench is not matched to the official panel | `kaggle/localbench.py`: essential-gene panel, `N_PRED = 100`, `--anchor-splits 2` | Confirmed |
| Final phase needs `vcc-cli ≥ 0.3.0`; we pin 0.2.2 | PyPI (0.3.0 is latest); `run_kaggle.py` and `stream_package.py` | Pin confirmed (0.2.2). The ≥ 0.3.0 requirement is from the CLI wiki (not re-read); upgrade regardless |
| Published results may train a model but not be submitted or used to revise predictions | Official rules page, "Machine Learning Predictions Only" (revised 16 Sep 2026) | Confirmed verbatim. B2's copied H1 rows violate it; B5/B5v3 (published LFCs × a constant, nothing fitted) are arguable. The next model must be a fitted model |
| Two submissions per day; validation entries not prize-eligible | Rules page | Confirmed |
| Generator samples the 400 control cells with replacement | `generate_b2_submission.py:338` | Confirmed |
| `VAL_CONTEXTS` hard-coded in the generator | `config.py:41–42`, generator loop | Confirmed; `FINAL_CONTEXTS = ["D","E","F"]` exists but is unused |

## 2. Validation protocol (built before any new model)

**Unit of evaluation:** a held-out *cell line*. A model is fitted only on the other lines and predicts the held-out
line's knockdown responses from source screens alone.

**Lines with measured responses** (all from the PIE processed release, one DE pipeline: Wilcoxon, BH, CPM > 5 gate,
arithmetic-mean log2FC): H1 (VCC-2025, 10x Flex, 146 knockdowns after the on-target filter), RPE1, HepG2, Jurkat, K562
(Replogle/Nadig essential screens), HCT116 and HEK293T (X-Atlas/Orion, genome-wide), and Jiang et al.'s cytokine-
stimulated lines.

**Features available at prediction time for the challenge targets** (and therefore the only features allowed):
K562 genome-wide (Replogle GWPS; 272/300 validation targets), Zhu CD4+ T cells (291), Orion HCT116 and HEK293T
(148). Replogle essential and Nadig lines cover none of the validation targets, so they are targets only, never
features.

**Two levels of scoring:**
1. *Pseudobulk level, every held-out line* (cheap, local): cosine of predicted vs real log2FC on tested genes
   (pds proxy), direction agreement on reference-significant genes (fid/reach proxy), normalised absolute log2FC
   error on reference-significant genes (nmae proxy), fraction of variance explained (mse proxy). Bootstrapped over
   knockdowns for confidence intervals.
2. *Official, H1 primary and RPE1 secondary*: emit 400 cells per knockdown from real controls with a fixed emitter
   (T5), score with `cell-eval2` `vcc2026` preset, `prep-real-bundle` with 5 anchor splits, real perturbed groups
   capped at 400 cells. Report all six scaled and raw metrics, n_pred vs N_conf per knockdown.

**Leakage rules:** the held-out line's responses never enter fitting, feature scaling, weight choice or
hyperparameter choice. Regularisation is chosen by an inner leave-one-line-out loop over the training lines only.
A target line is never also a feature (K562 essential is excluded as a target when K562 GWPS is a feature).
No leaderboard feedback is used.

**Known limits:** only H1 shares the challenge's chemistry and depth; the essential-gene lines respond far more
strongly than the challenge's regulator-heavy panel; Jiang's contexts are all cytokine-stimulated. Gains must show
on H1 *and* on the majority of other held-out lines before adoption.

## 3. Candidates, in increasing complexity (one change at a time; emitter fixed at T5 throughout)

- **Baselines:** zero effect; context mean; B5-style direction; B5v3-style mean log2FC at c = 1 and c = 2.
- **A. Source denoising:** shrink each source's log2FC by its own significance (soft threshold on |z| or a
  posterior-mean shrinkage). Each source alone vs combined. Statistics are never mixed across sources: Replogle uses
  its z, Zhu its z, Orion its FDR.
- **B. Weighted source average:** equal weights × α (α fitted leave-one-line-out); global learned weights; weights
  from basal-expression similarity of the target line's controls to each source line's controls (tested, not
  assumed). Fallback when a source lacks the target: renormalise over available sources; none available: zero effect.
- **C. Ridge transfer:** y_line(g,p) ~ Σ_s w_s·x_s(g,p) + missing-source indicators, shared across genes and
  knockdowns, features standardised on training lines only, λ by inner leave-one-line-out; compared with a
  shrinkage-only baseline. Per-gene ridge only if per-gene training examples suffice (reported either way).
- **D. Combination** of whatever A–C validate. At most a handful of runs.

Then, with the transfer model fixed, emitters T2 vs T4 vs T5 and edited-set sizing are tested separately.

## 4. What would change the final submission

A candidate replaces B5 only if, versus B5 rebuilt the same way: official overall on H1 is higher, the pseudobulk
proxies improve on most held-out lines, and no metric collapses. Otherwise B5 (fitted-scale variant, for
compliance) is rebuilt for D/E/F.
