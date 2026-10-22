# VCC 2026 next model — data audit (22 Oct 2026)

All numbers below are measured from the built table (`D:/VCC_outputs/xline/*.npz`, built by `models/xline_data.py`,
validated by `models/xline_check.py`; machine-readable copies in `coverage.json` and `check_report.json`).

## 1. Build status

- Built successfully in 55 min (12:01–12:56 IST), one run, no errors. 12 files: 11 held-out lines + `VCC_val` (the
  challenge targets, features only).
- Validator (`xline_check.py`): **no hard failures.** No duplicate knockdown or gene identifiers; all arrays share each
  line's (knockdowns × genes) shape; reference FDR within [0, 1]; no line carries itself as a source.

## 2. Harmonisation (how, and what was checked)

| Item | Handling | Check |
| --- | --- | --- |
| Gene identifiers | HGNC symbols throughout. K562 files carry Ensembl ids as the index and symbols in `var.gene_name`, which is what is used; Zhu `var.gene_name`; PIE files list symbols in `meta.genes`. First occurrence kept where a symbol repeats | No duplicates in any output |
| Knockdown identifiers | Target gene symbol. Jiang rows are `context:gene` (one base line, five stimuli) and match sources on the gene part | No duplicates |
| Effect size | log2 fold change everywhere. Targets: PIE `lfc_true` (log2 of arithmetic-mean CP10K ratio, tested genes only). K562: log2((CPM_kd + 0.5)/(CPM_ntc + 0.5)) from raw bulk. Zhu: its `log_fc` layer, mean of 3 culture conditions. Orion: PIE `lfc_true` | Scales differ by source (pseudocounts, condition means); every candidate fits its own scale/weights per source, so no cross-source scale is assumed |
| Significance | Kept separate from effect size (`Z_*` arrays, never used as effects except in the explicitly z-based B5-direction baseline). K562: Replogle normalized-bulk z. Zhu: `zscore` layer. Orion: \|z\| implied by its BH-adjusted p (a lower bound, documented in code) | Sign of effect vs sign of z where \|z\| > 3: **100 %** agreement in every line with ≥ 100 such entries (K562 on H1/Orion/Jiang had < 100 strong entries; not checkable there) |
| Missing values | NaN for untested target genes and for absent source knockdowns/genes. Zero is never a fill value in the stored table | 0 covered source rows that are all-zero; 0 % exact zeros among tested Y |
| Duplicates / replicates | Orion and Replogle use the first row per knockdown; Zhu averages its 3 culture conditions; Jiang keeps each stimulus as its own row | — |

## 3. Coverage per held-out line and source

Source coverage = knockdowns in that line for which the source has a measurement.

| Held-out line | Knockdowns | Genes | Tested genes | Ref.-significant genes / knockdown (median) | K562 | Zhu | HCT116 | HEK293T |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| H1 (VCC-2025, 10x Flex) | 190 | 12286 | 87% | 2016 | 173 | 166 | 87 | 13 |
| RPE1 | 1453 | 6642 | 100% | 463 | 1453 | 900 | 577 | 58 |
| HepG2 | 1253 | 6642 | 100% | 80 | 1253 | 692 | 480 | 42 |
| Jurkat | 1439 | 6642 | 100% | 107 | 1439 | 944 | 608 | 58 |
| HCT116 (1,500 sampled) | 1500 | 22182 | 51% | 3 | 1500 | 1306 | — (self) | 231 |
| HEK293T (all 831 with K562 coverage) | 831 | 22182 | 56% | 2 | 831 | 653 | 586 | — (self) |
| Jiang A549 | 58 | 22454 | 54% | 3 | 43 | 55 | 27 | 11 |
| Jiang BxPC-3 | 96 | 22454 | 54% | 182 | 76 | 88 | 40 | 10 |
| Jiang HAP1 | 119 | 22454 | 55% | 7 | 104 | 112 | 45 | 12 |
| Jiang HT-29 | 75 | 22454 | 51% | 9 | 60 | 72 | 34 | 9 |
| Jiang MCF7 | 27 | 22454 | 53% | 70 | 18 | 25 | 18 | 6 |

**Challenge validation targets (300):** K562 272, Zhu 291, HCT116 146, HEK293T 11. Within covered targets, the share of
the 18,533 challenge genes each source measures: K562 41 %, Zhu 55 %, HCT116 54 %, HEK293T 58 %.

## 4. How well each source matches each line (raw, before any model)

Median per-knockdown cosine between a source's log2FC and the line's measured log2FC (genes both measure, ≥ 50 genes).

| Held-out line | K562 | Zhu | HCT116 | HEK293T | K562 vs Zhu |
| --- | ---: | ---: | ---: | ---: | ---: |
| H1 | 0.149 | 0.121 | 0.252 | 0.089 | 0.063 |
| RPE1 | 0.153 | 0.137 | 0.349 | 0.196 | 0.082 |
| HepG2 | 0.138 | 0.093 | 0.327 | 0.228 | 0.081 |
| Jurkat | 0.183 | 0.129 | 0.429 | 0.222 | 0.081 |
| HCT116 | 0.084 | 0.081 | — | 0.051 | 0.044 |
| HEK293T | 0.027 | 0.025 | 0.045 | — | 0.023 |
| Jiang A549 | 0.023 | 0.079 | 0.014 | 0.011 | 0.036 |
| Jiang BxPC-3 | 0.038 | 0.086 | 0.026 | 0.002 | 0.036 |
| Jiang HAP1 | 0.082 | 0.118 | 0.468 | 0.021 | 0.035 |
| Jiang HT-29 | 0.078 | 0.108 | 0.070 | 0.005 | 0.032 |
| Jiang MCF7 | 0.041 | 0.069 | 0.009 | 0.031 | 0.039 |

Reading: every source agrees only weakly with every line (cosines 0.02–0.47). HCT116 matches the essential-gene lines
best (0.33–0.43), but it also shares the targets' DE pipeline (PIE's processing), which may inflate its apparent
agreement relative to K562 and Zhu, whose effects we computed differently. Treated as a hypothesis, tested by the
models.

## 5. Leakage and availability checks

- **Held-out line never a feature:** the HCT116 table has no `X_hct116`, HEK293T no `X_hek293t` (validator check).
  K562 is never a target, because K562 genome-wide is a feature.
- **Held-out line never in fitting:** `xline_models.py` fits every scale, weight, threshold and ridge λ on the other lines;
  hyperparameters are chosen by an inner leave-one-line-out over the training lines only.
- **Same knockdown through another feature:** the sources are independent experiments in other cell types; none
  contains the held-out line's measurement. Jiang's five stimuli of one base line are held out together.
- **Features available for the real task:** only K562, Zhu, HCT116 and HEK293T, the four sources that exist for the
  challenge targets. Replogle essential and Nadig lines (0/300 challenge targets) are targets only.

## 6. Limitations

- Only H1 shares the challenge's chemistry and depth; the Replogle/Nadig lines are 3′ essential-gene screens whose
  responses are much stronger than the challenge panel's.
- HCT116, HEK293T, A549, HAP1 and HT-29 have a median of 2–9 reference-significant genes per knockdown; the
  significance-based proxies (direction agreement, fold-change error) are uninformative or undefined for most of their
  knockdowns, and their ranking proxy is noisy.
- HEK293T covers 11/300 challenge targets: kept as a feature for completeness, but it cannot matter for the real task.
- Overlap of each line's knockdowns with the challenge targets is small (H1: 13); results on that overlap are reported
  but carry wide intervals.
