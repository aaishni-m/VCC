# VCC 2026 next model — first transfer-model comparison (22 Oct 2026)

Branch `next-model`. B5 (`b5-v1`), B5v3 and all submission files are unchanged.
Status labels used below: **[measured]** = produced by code that ran on real data; **[proxy]** = a pseudobulk stand-in for
an official metric, not an official score; **[pending]** = launched, no result yet; **[not run]**; **[hypothesis]**.

## 1. What ran

| Step | Status | Detail |
| --- | --- | --- |
| Training table (`models/xline_data.py`) | [measured] done | 55 min, 12 files, validator: no hard failures (`VCC_NEXT_MODEL_DATA_AUDIT.md`) |
| Leave-one-line-out comparison (`models/xline_models.py`) | [measured] done | 11 held-out lines × 13 candidates, 68 min, 0 failed fits. Pure-speed caching added mid-run; verified identical to the uncached code (max difference 2e-16) before restarting |
| Paired analysis (`models/xline_analyze.py`) | [measured] done | per-knockdown paired bootstrap, 2,000 resamples |
| Official-scorer H1 benchmark (`kaggle/localbench_h1.py`, Kaggle `vcc-localbench-h1` v6) | [measured] done, **reduced mode** | 5 variants; runs 3-5 were OOM-killed (exit -9) in `cell-eval2 baseline` on Kaggle's 31 GB box, so v6 caps real groups at 250 cells, uses 8,000 held-out controls and 11,957 genes (> 1 CPM); 400 predicted cells/knockdown, 5 anchor splits. See section 10 |
| Official scores on A/B/C or D/E/F | [not run] | validation leaderboard closes 22 Oct; final phase shows no scores |

Reproduce: `python models/xline_data.py` → `python models/xline_check.py` → `python models/xline_models.py` →
`python models/xline_analyze.py`. Code at the commit that adds this report; data: PIE processed release (HF revisions
current on 22 Oct), Replogle figshare files 35774443/35773217, Zhu `GWCD4i.DE_stats.h5ad`. Seeds: Orion subsample 0,
bootstraps 0.

## 2. Protocol and its leakage safeguards

- Outer loop: each of 11 lines held out; every scale, weight, ridge λ and shrink threshold is fitted on the other 10.
- Inner loop: λ (on r²) and the shrink threshold (on the ranking proxy) chosen by leave-one-line-out over the 10
  training lines only. The statistics cache is keyed by line and only ever read for training lines of the current fold.
- A line is never its own source (validator check); K562 is never a target. Sources are independent experiments, so a
  held-out knockdown's own response cannot enter its prediction.
- All candidates, including zero, are scored on the same knockdowns and genes of each held-out line. A knockdown with
  no source coverage gets a zero prediction and stays in the average (as a submission would).

## 3. Proxy metrics, and how they differ from the official ones

| Proxy | Computed as | Official counterpart and differences |
| --- | --- | --- |
| ranking (pds_proxy) | rank of a knockdown's own real log2FC among all real log2FCs of the line by cosine to the prediction; the line's target genes removed | pds: same idea, but on pseudobulk log1p(CP50K) effects of 400 emitted cells, including emission noise |
| dir_sig | sign agreement on reference-significant genes (FDR < 0.05); zero prediction = 0.5 | close to fid/reach in spirit; the official ones depend on which genes OUR cells make significant (n_pred vs N_conf), which pseudobulk cannot see |
| nmae_sig | mean\|pred − y\| / mean\|y\| on reference-significant genes, knockdowns with ≥ 10 such genes | nmae: same formula, but on log2FC re-measured from emitted cells |
| r2 | 1 − Σ(pred − y)² / Σy² over tested genes | mse: ratio of sums in log1p space against the control; the official 0 point is the oracle context mean, not zero |

No proxy predicts jac. Proxy values are **not** comparable to leaderboard numbers.

## 4. Results on the well-powered lines [measured, proxy]

Well-powered = H1, RPE1, HepG2, Jurkat (≥ 80 reference-significant genes per knockdown, 190–1,453 knockdowns,
6,642–10,693 tested genes). Ranking proxy with its 95 % bootstrap interval; higher is better except nmae.

| Candidate | H1 rank | RPE1 rank | HepG2 rank | Jurkat rank | dir (H1 / RPE1 / HepG2 / Jurkat) | nmae (H1 / mean of other 3) | r² (H1 / mean of other 3) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| zero | 0.500 | 0.500 | 0.500 | 0.500 | 0.50 / 0.50 / 0.50 / 0.50 | 1.000 / 1.000 | 0.000 / 0.000 |
| K562 alone | 0.826 (0.787–0.863) | 0.783 (0.769–0.797) | 0.798 (0.783–0.811) | 0.846 (0.835–0.857) | 0.54 / 0.69 / 0.79 / 0.79 | 0.985 / 0.971 | −0.005 / 0.011 |
| Zhu alone | 0.738 (0.700–0.775) | 0.577 | 0.560 | 0.621 | 0.54 / 0.57 / 0.59 / 0.60 | 0.988 / 0.990 | −0.001 / 0.003 |
| **Equal average, fitted scale** | **0.860 (0.831–0.885)** | 0.781 (0.767–0.796) | 0.794 (0.780–0.809) | **0.846 (0.837–0.857)** | **0.56 / 0.71 / 0.80 / 0.80** | **0.962** / 0.930 | −0.011 / 0.031 |
| Learned weighted average | 0.855 (0.811–0.884) | 0.777 | 0.792 | 0.839 | 0.56 / 0.71 / 0.80 / 0.80 | 0.968 / 0.928 | −0.018 / 0.033 |
| Ridge (λ by inner LOLO) | 0.855 (0.821–0.883) | 0.777 | 0.792 | 0.839 | 0.56 / 0.71 / 0.80 / 0.80 | 0.987 / 0.963 | −0.044 / 0.022 |
| Denoised average | 0.770 (0.730–0.810) | 0.714 | 0.689 | 0.725 | 0.52 / 0.56 / 0.61 / 0.59 | 1.064 / 0.919 | −0.284 / 0.052 |
| Ridge + denoise | 0.762 | 0.705 | 0.680 | 0.717 | 0.52 / 0.56 / 0.61 / 0.59 | 1.069 / 0.946 | −0.266 / 0.041 |
| B5-direction (unit K562 z + unit Zhu z, fitted scale) | 0.851 (0.820–0.883) | 0.773 | 0.775 | 0.820 | 0.55 / 0.71 / 0.79 / 0.80 | 0.975 / 0.939 | **0.008** / 0.010 |
| B5v3-style mean (K562 + Zhu) × 1 | 0.845 | 0.770 | 0.784 | 0.841 | 0.55 / 0.70 / 0.79 / 0.80 | 1.060 / **0.864** | −0.317 / −0.015 |
| B5v3-style mean × 2 | 0.845 | 0.770 | 0.784 | 0.841 | 0.55 / 0.70 / 0.79 / 0.80 | 1.373 / 0.843 | −1.536 / −0.221 |

Paired differences on identical knockdowns (mean, 95 % interval):

| Comparison | H1 | RPE1 | HepG2 | Jurkat |
| --- | --- | --- | --- | --- |
| Equal average − zero, ranking | +0.360 (+0.327, +0.392) | +0.281 (+0.268, +0.295) | +0.294 (+0.281, +0.308) | +0.346 (+0.335, +0.358) |
| Equal average − B5-direction, ranking | +0.009 (−0.023, +0.040) | +0.008 (+0.000, +0.015) | +0.019 (+0.011, +0.027) | +0.027 (+0.019, +0.034) |
| Equal average − B5-direction, direction | +0.003 (−0.004, +0.010) | +0.007 (+0.003, +0.011) | +0.007 (+0.002, +0.012) | +0.007 (+0.003, +0.011) |
| Ridge − equal average, ranking | −0.005 (−0.009, −0.001) | −0.004 (−0.006, −0.002) | −0.003 (−0.004, −0.001) | −0.008 (−0.009, −0.006) |
| Denoised − equal average, direction | −0.041 (−0.049, −0.033) | −0.156 (−0.162, −0.150) | −0.196 (−0.204, −0.187) | −0.214 (−0.221, −0.207) |
| K562 alone − equal average, ranking | −0.034 (−0.061, −0.009) | +0.002 (−0.006, +0.009) | +0.004 (−0.005, +0.012) | −0.000 (−0.007, +0.007) |

Share of knockdowns improved, equal average vs B5-direction (ranking): H1 38 %, RPE1 53 %, HepG2 56 %, Jurkat 57 %. The
mean gains come from many small shifts, not a few targets, and on H1 most knockdowns do not improve.

## 5. Exploratory lines [measured, proxy; do not decide]

HCT116, HEK293T and the five Jiang lines have a median of 2–182 (mostly 2–9) reference-significant genes per knockdown
and 27–1,500 knockdowns. Ranking means: equal average 0.597, B5-direction 0.612, denoised 0.615, ridge 0.594. Here
denoising helps (e.g. HEK293T 0.782 vs 0.666, BxPC-3 0.676 vs 0.574), the reverse of the well-powered lines: shrinking
noisy source entries pays only when the target's own response is tiny.

## 6. Challenge-target overlap [measured, proxy; too small to decide]

Knockdowns that are also validation targets: H1 16, HCT116 52, HEK293T 10, Jiang A549 5. On H1's 16: B5v3-style 0.983,
B5-direction 0.978, equal average 0.972, ridge 0.970, K562 0.927, denoised 0.858. Intervals at n = 16 are too wide to
separate the top four.

## 7. Answers to the questions in the brief

- **Beats zero?** Yes, every source-based candidate, on every well-powered line, by +0.28 to +0.36 on ranking and +0.06
  to +0.30 on direction; intervals far from 0. On effect sizes, barely: best r² 0.05 (Jurkat), nmae 0.92–0.96.
- **Ridge vs equal and learned averaging?** Ridge is slightly *worse* than the equal average on all four lines
  (intervals exclude 0) and no better than the learned average on direction. Not worth continuing in this form.
- **Denoising consistent?** No. It hurts on all four well-powered lines and helps on weak lines.
- **Learned weights stable?** Yes, across all 11 folds (ridge: K562 ≈ 0.08, Zhu ≈ 0.11, HCT116 ≈ 0.13, HEK293T ≈ 0.16
  at λ = 1; learned average: ≈ 0.17 / 0.22 / 0.27 / 0.33). But they put most weight on HEK293T, which covers 11 of 300
  challenge targets, so on the real task they collapse towards the equal average of the sources that exist.
- **Ridge learns transfer or just shrinks?** Mostly shrinks: λ = 1 in 10 of 11 folds, small near-uniform weights.
- **Driven by one context?** No: the equal average's gain over B5-direction is positive on all four lines, significant
  on three, and not significant on H1, the one line in the challenge's chemistry.
- **Fitted scale:** 0.39–0.44 across folds. The raw source mean should be shrunk to ~40 %; B5v3's ×2 is ~5× too large,
  consistent with its leaderboard nmae loss (−0.058 vs B5's +0.084).
- **Enough observations?** For ranking and direction on RPE1/HepG2/Jurkat, yes (n > 1,200 each). For H1, n = 190 and
  the gain over B5-direction is within noise. For the challenge overlap, no.

## 8. Limitations

- All numbers above are proxies; official six-metric scores exist only after the H1 benchmark finishes.
- B5's real profile also contains 0.5 × the H1 average response and is emitted with T2; the "B5-direction" baseline here
  keeps only its transferred direction (the H1 average cannot be reproduced for other held-out lines without using a
  response average from the held-out line). So "beats B5-direction" ≠ "beats B5 as submitted".
- HCT116 shares the targets' DE processing (PIE release); its agreement with targets may be inflated.
- The essential-gene lines respond far more strongly than the challenge panel; H1 is the closest match and is where
  the evidence is weakest.

## 9. Final-phase blockers (not mixed into the comparison)

- `vcc-cli` pinned to 0.2.2 in `kaggle/run_kaggle.py` and mirrored by `models/stream_package.py`; the final phase needs
  ≥ 0.3.0 → bump, re-run `stream_package.py --selftest`.
- `generate_b2_submission.py` and `build_combined_obs` loop over `cfg.VAL_CONTEXTS` (A/B/C); final needs
  `FINAL_CONTEXTS` from the new `manifest.json`, new `pert_counts.csv`, new pseudobulk caches for D/E/F.
- Source tables for the final 300 targets: re-run `xline_data.py` with the final `pert_counts.csv` as an argument.
- Compliance (rules page, 22 Oct): published results may train or fine-tune a model but not be submitted or used to
  revise predictions. The equal-weight average with a scale fitted leave-one-line-out on public screens is a fitted
  model; B2's copied H1 rows are not allowed; drop any copied published rows.

## 10. Official-scorer H1 comparison [measured, reduced mode] (22 Oct, Kaggle `vcc-localbench-h1` v6)

Real VCC-2025 H1 cells, `cell-eval2` `vcc2026` preset, scaled like the leaderboard ((u - b)/(r - b) against this
bundle's own context-mean baseline and 5 replicate anchors). 144 knockdowns with >= 50 real cells. **Reduced mode**
(250 real cells/knockdown instead of 400, 8,000 held-out controls, 11,957 genes): fewer real cells give the reference fewer
significant genes, so fid / jac / reach are not comparable to official numbers; every variant gets identical settings,
so the comparison between variants is fair. Predictions built without any H1 response (`results/h1_profiles_bench.npz`,
`results/h1_ablation_profiles.npz`, `models/h1_ablation.py`). Per-knockdown files: `results/h1_official_reduced_perkd_*.csv`.

| Variant | Emitter | Overall | pds | mse | nmae | fid | reach | jac |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| B5-like as submitted (direction at B5's level + training-line average response at B5's 0.576 ratio) | T2 | **0.229** | 0.568 | 0 | **-0.027** | **0.540** | **0.070** | **0.225** |
| Equal average + average response (alpha 0.49, beta 0.019, fitted on training lines) | T5 | 0.194 | 0.640 | 0 | -0.046 | 0.393 | 0.042 | 0.133 |
| B5-direction + average response (alpha 4.12, beta 0.019) | T5 | 0.190 | 0.572 | 0 | -0.041 | 0.414 | 0.069 | 0.126 |
| Equal average, fitted scale (0.44) | T5 | 0.181 | **0.645** | 0 | -0.049 | 0.346 | 0.031 | 0.111 |
| B5-direction, fitted scale | T5 | 0.176 | 0.595 | 0 | -0.043 | 0.350 | 0.052 | 0.104 |

Paired bootstrap over the 144 knockdowns' per-knockdown raw metrics (2,000 resamples; positive = first variant better):

| Comparison | pds | nmae | fid | reach | jac |
| --- | --- | --- | --- | --- | --- |
| Equal - B5-direction (both T5) | +0.023 (-0.017, +0.062) | -0.004 (-0.010, +0.002) | -0.003 (-0.027, +0.022) | -0.019 (-0.043, +0.006) | +0.003 (-0.003, +0.010) |
| B5-like (T2) - equal (T5) | -0.036 (-0.075, +0.006) | +0.013 (+0.004, +0.022) | +0.144 (+0.114, +0.177) | +0.035 (+0.008, +0.064) | +0.047 (+0.032, +0.061) |
| Equal + mean - equal | -0.002 (-0.042, +0.038) | +0.002 (-0.003, +0.007) | +0.035 (+0.023, +0.050) | +0.010 (-0.014, +0.035) | +0.009 (+0.007, +0.012) |
| B5-dir + mean - B5-dir | -0.011 (-0.047, +0.028) | +0.001 (-0.004, +0.006) | +0.047 (+0.032, +0.065) | +0.015 (-0.016, +0.047) | +0.009 (+0.007, +0.011) |

Reading:
- **Transfer model (emitter fixed, T5): equal average = B5-direction within noise.** No metric differs significantly;
  pds favours the equal average and reach favours B5-direction, neither beyond noise. The proxy gain of section 4 does
  not survive the official scorer on H1.
- **The average-response term helps** both directions on fid and jac (intervals above 0), without hurting pds.
- **B5-like as submitted is best overall (0.229)**, through fid, jac, nmae and reach. It differs from the T5 variants in
  two ways at once (the T2 emitter and a larger average-response term), so this run cannot say which carries the gain.
- **Strong caution on transfer to the challenge:** H1's responses are dense (median ~2,000 reference-significant genes per
  knockdown at full depth), which rewards dense calling such as T2's redraw. On the A/B/C leaderboard the same B5
  structure scored fid -0.003 and jac -0.010, i.e. at chance. H1, like the RPE1 bench (fid 0.66), likely overstates
  what T2 earns on the challenge's weaker regulator panel.
- mse is 0 (clamped) for every variant and nmae is slightly negative for every variant: no candidate predicts effect
  sizes better than this bundle's context-mean baseline.
