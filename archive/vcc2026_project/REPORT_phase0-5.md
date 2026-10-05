# 2026 Virtual Cell Challenge — Competition Intelligence, Data Audit, and Strategy

*Compiled 2026-09-07. Sources are cited inline; anything not attributed to a source is this
session's own derivation or measurement, marked as such.*

---

## 1. Competition intelligence report

### 1.1 Official task definition

The 2026 VCC is a **zero-shot cross-context perturbation prediction** challenge. Participants are
given the unperturbed (non-targeting-control) transcriptomic state of cells from six cell lines and
a list of genes to knock down by CRISPR interference (CRISPRi); they must predict the post-knockdown
expression profile Arc will measure experimentally. Three of the six cell lines are the **validation**
contexts (`A`/`B`/`C`, live now); the other three are **final-test** contexts (`D`/`E`/`F`, released
October 22, 2026), so the validation-phase leaderboard is itself a proxy for the real zero-shot test —
context generalization is being scored twice, at increasing distance from anything a team has seen.
[Arc Institute: 2026 VCC](https://arcinstitute.org/news/virtual-cell-challenge-2026)

**Confirmed exactly** (from the manifest shipped with the controls you gave me, cross-checked against
the official `cell-eval2` scoring library's packaged `vcc2026.yaml` config and its metric
specification document — see §1.3):

| quantity | value |
|---|---|
| genes (feature axis) | 18,533, in the exact order of `gene_names.csv` |
| target genes per context | 300 (identical gene set across A/B/C — confirmed by `pert_counts.csv`) |
| non-targeting control constructs per context | 46 |
| control cells released per context | 18,400 |
| cells per perturbation required on upload | exactly 400 |
| ground-truth cells per context (held out) | 138,400 = 300×400 (perturbed) + 18,400 (control) |
| contexts, validation phase | A, B, C |
| contexts, final phase | D, E, F (different cell lines; released ~Oct 22) |
| perturbation modality | dual-guide CRISPRi (catalytically dead Cas9 + KRAB) |
| sequencing | 10x Genomics Flex chemistry, Ultima UG100 |

This is a validated, internally-consistent reading: your `manifest.json` (`panel_id:
"vcc2026-val-1"`, `partition: "val"`) states these numbers directly, and they match the official
metric spec's own text (`docs/vcc2026_metrics/vcc2026-metrics.md` in the public
[`ArcInstitute/cell-eval2`](https://github.com/ArcInstitute/cell-eval2) repo) verbatim, down to "300
target constructs... downsampled to exactly 400 cells... on a common axis of 18,533 genes."

### 1.2 Exact submission format

From `cell-eval2`'s own documentation (section 7 of the metrics spec — this is the most concrete,
load-bearing part of the whole intelligence-gathering exercise, so I'm quoting it closely rather than
paraphrasing):

- **One file, all contexts.** A single `.h5ad` covers A, B, and C together (D/E/F later); every cell
  carries a `context` label in `.obs` matching the phase's control files exactly.
- **Perturbation column is `target_gene`**, holding **gene symbols** (`ADNP`, not `ADNP-1`).
- **Exactly the perturbations in `pert_counts.csv`** — no extras, none missing — at **exactly 400
  cells each**, in every context.
- **No non-targeting rows on upload.** The platform (`vcc prep`) splices the real held-out control
  cells in itself before scoring, specifically so the control side of every metric is provably
  real reference data, never something a submission supplied.
- **Genes: exactly 18,533, in `gene_names.csv` order.**
- **Values: raw, untransformed, non-negative integer counts, no cell total above 10⁶.** Do not
  normalize or log-transform — the scorer needs raw counts because its expression metrics compare
  group-summed profiles, which can't be recovered from log data. A matrix failing any of this is
  **rejected outright, not scored**.
- **Stored-entries cap: 4,750,000,000** (sparse nonzeros — *"store the prediction sparse, without
  explicit zeros"*; a naive dense array at this shape is already 1.4× over the cap on its own).

I implemented this exact contract as `submission/validate_submission.py` and it is source-verified
against the spec text, not guessed — see §5.

### 1.3 Exact evaluation metrics

This was the single hardest-to-pin-down part of the brief (the 2026 Arc Institute announcement page
only says "an aggregate across a broader panel of six metrics" without naming them), so it's worth
explaining how I actually found the answer rather than guessing at it. I traced it through Arc's own
open-source scoring tool, `cell-eval2` (successor to the 2025 `cell-eval`), which ships a packaged
`vcc2026` config and — remarkably — a full LaTeX-sourced technical specification document at
`docs/vcc2026_metrics/vcc2026-metrics.md` in that public repo, dated 2026-08-19 (the week the 2026
challenge opened). I read that document directly rather than summarizing a rendering of it, since it
contains exact formulas.

**The six `vcc2026` metrics**, two from expression profiles directly and four from a
Wilcoxon-rank-sum differential-expression table:

| metric | what it measures |
|---|---|
| `pds_cosine` | Perturbation discrimination — is the predicted effect closer to its own true effect than to any of the other 299 target genes' true effects, by cosine distance on **all 300 target genes removed** from the feature space (not just the row's own gene — this closes a 2025-era exploit, see §1.5) |
| `expr_mse_unbiased_capped_norm` | Sampling-noise-corrected squared expression error, as a fraction of the true perturbation-to-control distance; a jackknife estimate removes the part of the error that's just finite-cell sampling noise |
| `de_wilcoxon_direction_fidelity_yield_raw` | Of the genes the reference calls significantly changed, what fraction did the submission call in the right direction — with coverage penalized if you call fewer genes than the reference found |
| `de_wilcoxon_direction_reach_raw` | How deep into the submission's own confidence ranking directional calls stay ≥90% correct |
| `de_wilcoxon_sig_jaccard` | Jaccard overlap of the reference's and the submission's significant-gene sets (symmetric — a missed gene and an invented one cost the same) |
| `de_wilcoxon_lfc_nmae` | Normalized MAE of predicted log2 fold-changes, gated to genes the reference calls significant |

**Scoring.** Each metric is placed on a common scale `s = (u − b) / (r − b)`, where `b` is a
published **mean-response baseline** (predict the context's average perturbation effect for every
gene — this is *not* what our B0 does, see §4) and `r` is a **replicate** (the reference split in
half and compared with itself). So 0 = the naive mean-effect baseline, 1 ≈ reproducing the experiment
exactly, and a context's score is the equal-weight mean of all six scaled metrics. This is a genuinely
unusual and well-thought-through design: it's *relative* to how reproducible the underlying biology
even is at this sequencing depth, not an absolute target.

The perturbed gene's own row is excluded from every metric (predicting your own knockdown is the
premise of the experiment, not a prediction). The DE-based metrics additionally restrict to genes
expressed above 5 CPM in the control and apply Benjamini-Hochberg at α=0.05, computed **per
perturbation**, not once across the panel.

The full spec, `docs/vcc2026_metrics/vcc2026-metrics.md`, is >700 lines with worked formulas for
every term; I've distilled what's decision-relevant above and copied the constants into
`vcc2026/config.py`. I have **not** independently re-derived or numerically re-verified these
formulas against real ground truth (we have none locally); I'm reporting what the source document
states.

### 1.4 External-data policy

Confirmed, in Arc's own words: *"Participants may use any modeling strategy and train their models
on any data."* Public perturbation data through Arc's Virtual Cell Atlas is explicitly called out as
available. [Arc Institute: 2026 VCC](https://arcinstitute.org/news/virtual-cell-challenge-2026) —
so using the 2025 `nicolas-lynn/vcc-perturb` derivative (or the raw 2025 Arc training set it's built
from) is squarely allowed, not a gray area.

### 1.5 2025 results, winning approaches, and what changed for 2026

**2025 winners** (from Arc's official wrap-up post,
[arcinstitute.org/news/virtual-cell-challenge-2025-wrap-up](https://arcinstitute.org/news/virtual-cell-challenge-2025-wrap-up)):

| Rank | Team | Method | Data used |
|---|---|---|---|
| 1st ($100k) | BM_xTVC (BioMap Research) | xTrimoSCPerturb — hybrid encoder-decoder + protein-model embeddings | public perturbation data + challenge training data |
| 2nd ($50k) | XLearning Lab (Sichuan Univ.) | pseudobulk representation, fully-connected nets, ESM-2 protein embeddings | public Perturb-seq only |
| 3rd ($25k) | Outlier | statistical "similarity-aware aggregation" across reference cell lines | public cell-line Perturb-seq only |
| Generalist Prize ($100k) | Altos Labs | flow-matching generative model in gene-expression space (custom U-Net), "go-with-the-flow" | public data + challenge data |

Stated takeaway from Arc directly: *"Hybrid models combining deep learning with statistical features
outperformed pure neural networks,"* and *"no single metric captures model quality."* Three of four
prize approaches used only or mostly public data, not the challenge's own labeled set — a real signal
that gene-representation transfer, not memorizing challenge-specific perturbations, is where the
2025 leaderboard was actually won.

**Why the 2026 metrics changed (from `cell-eval2`'s own changelog, which is unusually candid about
this):** the 2025 evaluation had a documented exploit and a documented instability, and the 2026 spec
was rewritten specifically to close them:
- **Exploit (independently corroborated by a third-party writeup):** a 2025-era submission could spike
  the other 299 target genes' values and win the discrimination metric on no real information, because
  only *each row's own* target gene was excluded from the feature space, leaving every *other*
  perturbation's target gene visible as a giveaway. `cell-eval2`'s changelog documents the fix
  (exclude the whole 300-gene panel, not just the row's own gene) and quantifies the old exploit
  directly: a submission built to exploit it scored 0.798/0.757/0.761 against baselines of
  0.530/0.528/0.510 on the three reference contexts — a huge, illegitimate lift. Separately, a
  November 2025 third-party writeup ([gmdbioinformatics substack, "Leaderboard Collapse"](https://gmdbioinformatics.substack.com/p/arc-virtual-cell-challenge-has-the))
  describes participants finding that the 2025 MAE penalty was effectively capped, letting variance
  inflation or log-transform tricks boost DE/PDS scores without a matching MAE penalty. Both point the
  same direction: 2025's leaderboard rewarded some gaming of the metric geometry, not only genuine
  biological signal, and I'd treat any *specific 2025 leaderboard score* as a description of that
  metric, not a clean measure of biological accuracy.
- **2026's fix, concretely:** the whole 300-gene target panel is excluded from the discrimination
  metric's feature space (not just each row's own gene), the mean-response baseline is re-measured
  and reported as the true no-information point (exactly 0.500 on `pds_cosine` now, by construction),
  and the expression-error metric adds a sampling-noise correction with a submission-side cap so a
  model can't manufacture fake "signal" by inflating cross-perturbation variance. I'd read this as
  meaningful evidence that **naive variance-inflation or degenerate-diversity tricks that worked in
  2025 are specifically disarmed in 2026** — worth keeping in mind so we don't spend effort chasing
  a 2025-style exploit that no longer exists.

### 1.6 Current 2026 leaderboard

**I could not retrieve a real leaderboard table, and I'm not going to fabricate one.** The
competition's own site, `virtualcellchallenge.org`, is a JavaScript single-page app; my web-fetch
tool retrieves only the pre-render HTML shell ("Loading...") and can't execute the JS that would
populate a leaderboard. I did not find a mirrored or cached leaderboard, a Kaggle/eval.ai-hosted
copy, or a third-party writeup quoting current 2026 standings (the challenge opened August 20, 2026,
so it's only been live ~2.5 weeks as of this report). **If you want current standings, someone with
browser access to that page (or Discord access to the challenge server) is the only way I know of to
get them right now** — I'd rather say that plainly than invent numbers.

### 1.7 Timeline

| date | event |
|---|---|
| Aug 20, 2026 | Validation data + leaderboard live |
| Oct 22, 2026 | Final test set released |
| Nov 5, 2026, 23:59 UTC | Final submissions due |
| mid–late Nov 2026 | Winners announced |

Source: [Arc Institute: 2026 VCC](https://arcinstitute.org/news/virtual-cell-challenge-2026).

---

## 2. Dataset compatibility report

### 2.1 What we actually have: the 2026 controls (audited directly, not from a card)

I loaded all three files (`context_A/B/C.h5ad`) in this session and audited them without densifying
anything (script: `data/audit/audit_2026.py`; full numeric output: `data/audit/audit_2026_summary.json`).

| | context A | context B | context C |
|---|---|---|---|
| shape | 18,400 × 18,533 | 18,400 × 18,533 | 18,400 × 18,533 |
| `X` dtype / format | float32, CSR | float32, CSR | float32, CSR |
| values | integral (raw counts confirmed) | integral | integral |
| density | 32.2% | 29.8% | 31.7% |
| obs columns | `target_gene` (100% "non-targeting"), `context`, `ntc_id` (46 levels, exactly 400 cells each) | same | same |
| `var` columns | none (gene symbols only, in `var_names`) | none | none |
| mean counts/cell | 21,134 | 19,990 | 21,157 |
| median counts/cell | 20,109 | 19,946 | 20,034 |
| mean genes detected/cell | 5,973 | 5,519 | 5,881 |
| genes never detected in any cell | 2,398 (12.9%) | 2,853 (15.4%) | 2,317 (12.5%) |
| `var_names` order | matches `gene_names.csv` exactly (checked head) | same | same |

**Finding, not assumption:** the three contexts are measurably different cell states, not
re-labelings of the same pool — B has both lower depth (~20k vs ~21.1k median counts/cell) and
fewer genes detected per cell (~5,519 vs ~5,880–5,973) than A and C. That's consistent with the
official framing (six *different cell lines*) and is direct evidence against assuming a
context-agnostic effect model (relevant to Phase 8 in your plan).

**Confirmed:** every cell in all three files is `target_gene == "non-targeting"` — this genuinely is
control-only, exactly as you said, with zero perturbed cells or ground-truth post-perturbation
profiles present. `manifest.json` (bundled alongside the h5ad files) states the *held-out* ground
truth size directly: 138,400 cells/context = 120,000 perturbed (300 genes × 400 cells) + the 18,400
released controls. That number isn't a guess — it's in the file you gave me.

### 2.2 What we (partially) know about `nicolas-lynn/vcc-perturb` — and a hard access blocker

**I was not able to load this dataset.** `huggingface.co` is blocked outright by this sandbox's
network egress policy (`datasets.load_dataset(...)` and `huggingface_hub` both fail with a `403` at
the proxy, logged as a policy denial, not a transient error). Per my own operating rules I should not
route around an organizational network policy denial — so rather than trying tricks to tunnel around
it, I'm flagging it as a blocker for you to decide on (see the question at the end of this message).

What I *can* report, from Hugging Face's own web dataset-card page (fetched as rendered text, **not**
by opening the actual parquet/h5ad files — treat every number below as a claim from the card, not
something I verified):

- It's built from the **official Arc 2025 VCC training set**
  (`gs://arc-institute-virtual-cell-atlas/virtual-cell-challenge/2025/train/adata_Training.h5ad`),
  H1 human embryonic stem cells, CRISPRi knockdown, CC-BY-4.0.
- Files present: `perturbation_deltas.parquet`, `perturbation_zscores.parquet`, `de_log2fc.parquet`,
  `de_pvalue.parquet`, `de_padj.parquet`, `pseudobulks.h5ad` (50 control pseudobulks of 15 cells
  each), `coexpression.h5ad` (gene×gene Pearson correlation), plus metadata CSVs
  (`retained_perturbations.csv`, `retained_perturbations_de.csv`, `all_candidates_scored.csv`) and
  two manifests.
- **A number I can't reconcile and won't paper over:** one page rendering said "750 perturbations ×
  ~18,000 genes"; the dataset's own `manifest.json` (which I also fetched, and trust more, since it's
  the actual build config rather than a summarized card) says `n_candidates_scored: 150,
  n_retained: 150, n_distinct_hit: 57`. These don't agree, and I'd guess the "750" figure is either a
  hallucination from the small summarization model behind my web-fetch tool, or refers to some other
  count (e.g. individual guide constructs across candidates) that got mislabeled in rendering — but I
  don't know which, and I'm not going to pick one and present it as fact. **This is exactly the kind
  of number that needs verifying against the real files, which we can't reach right now.**
- Per-file "shape" descriptions on the card are given as `(n_perts, n_genes)` — i.e. the card itself
  doesn't commit to concrete numbers either.
- Explicitly documented caveats on the card itself: pseudobulks are "random subsamples from one cell
  pool, not independent biological replicates" (p-values anticonservative), and the QC filter
  ("target genes shifted ≥50% in the expected direction, ≥15 passing cells") selects for
  stronger-than-average effects, which the card itself warns "potentially inflat[es] DE counts."

**Bottom line: I can tell you the file manifest and the documented processing pipeline, but I cannot
give you validated cell/gene/perturbation counts, confirm the train/val/test split structure inside
this specific HF repackaging, or inspect whether z-scores/deltas/pseudobulks actually look sane,
without downloading it.** Phase 2 of your plan is genuinely blocked, not just under-explored.

### 2.3 How the two datasets can (and can't) combine

Given what's confirmed above:

- **Can't** train `target_gene → ΔX` on the 2026 controls alone — there is no perturbed-cell signal
  in them at all, by design (that's the whole point of zero-shot evaluation).
- **Can** use the 2026 controls to characterize each context's baseline distribution (Phase 8/9
  material) and as the literal input to a control-null baseline (§4).
- **Can, in principle,** use `vcc-perturb`'s DE tables / deltas / pseudobulks as the actual
  perturbation-response training signal, transferred to 2026's target genes via gene-level features
  rather than gene-ID lookup — this is squarely allowed (§1.4) and is exactly what 3 of 4 2025 prize
  winners did. But:
  - **Cell line mismatch is real and needs to shape the model, not be ignored.** `vcc-perturb` is H1
    hESC; the 2026 validation contexts A/B/C are three *different* cell lines (the 2026 announcement
    says six cell lines "from different tissue origins," and our own audit in §2.1 shows A/B/C are
    measurably distinct from each other). So H1 is very likely a fourth, different context again —
    any transfer has to go through gene-level or pathway-level features that generalize across cell
    type, not a cell-line-specific effect table. This is precisely why Phase 8 (context conditioning)
    and Phase 11 (gene representation, not gene-ID memorization) matter so much for this specific
    combination of datasets.
  - **Gene panel overlap is unverified.** 2026's panel is 300 specific genes (`pert_counts.csv`); if
    `vcc-perturb`'s retained 150 (per its manifest) perturbations only partially overlap that list —
    plausible, since the 2025 challenge's own target list was different from 2026's — a lookup-table
    model (B1 as literally specified) would only cover the overlap and would need a fallback
    (gene-feature regression, Phase 11) for the rest. **We can't check this overlap without the
    actual gene lists from `vcc-perturb`, which is the same access blocker.**
  - **Effect-size inflation risk.** The card's own QC filter (≥50% shift in expected direction) means
    `vcc-perturb`'s retained perturbations are pre-selected for *bigger, cleaner* effects than a
    random 2026 target gene will show. A model trained only on these will likely over-predict effect
    magnitude on 2026's presumably-more-typical gene panel. Worth an explicit correction or at least
    an explicit caveat once we can see the actual retained-gene effect-size distribution.

---

## 3. What one prediction means (Phase 3)

Based on the confirmed manifest + spec (§1.1–1.2), not guessed:

- **One input** = a context's 18,400 released non-targeting control cells (raw counts, 18,533 genes)
  + one target gene symbol from the 300-gene panel for that context.
- **One target / one output** = a set of predicted single cells for that (gene, context) pair. The
  spec is explicit that **the number of predicted cells scored is unconstrained** at the
  `cell_eval2` scoring layer, but the **upload** layer (`vcc prep`) requires **exactly 400** per
  perturbation per context — so in practice you must produce exactly 400 predicted cells per
  (gene, context) combination, all raw integer counts over the same 18,533-gene axis.
- **One target-gene/context combination** = 400 predicted cells, compared against 400 real
  measured cells for that same combination (via the six metrics of §1.3), and the panel score for a
  context is the mean of each metric over its 300 perturbations.
- **Full submission** = 3 contexts × 300 genes × 400 cells = 360,000 predicted cells in one `.h5ad`
  (D/E/F will be a second, structurally identical submission for the final phase).

**Why 18,400 released controls ≠ 138,400 ground-truth cells, confirmed rather than assumed:**
`manifest.json`'s own `per_context` block states `ground_truth_cells: 138400 = control_cells (18400)
+ n_perturbations(300) × cells_per_pert(400) (120000)`. The 18,400 released cells are the *input*
baseline; the 138,400 is what Arc holds out per context as ground truth (120,000 perturbed + a
second, held-out 18,400 control block used as the reference side of every metric). **Simply
resampling the released 18,400 control cells is not equivalent to knowing the true post-perturbation
distribution — it's a valid null hypothesis (§4), not a shortcut to the answer.**

---

## 4. B0 — control-null baseline (implemented and validated this session)

Implemented as `models/control_baseline.py`: for each context and each of the 300 target genes, it
resamples (with replacement) 400 raw-count cells from that context's own 18,400 released controls,
and labels them with the target gene and context. This is deliberately *not* a perturbation model —
it predicts "no effect" everywhere, exactly the **"control-pasting"** degenerate case the official
metrics spec itself analyzes (§1.3, §1.5): predicted expected behavior is `pds_cosine` ≈ 0 (scaled;
chance-level, since every predicted profile is identical and un-informative), and
`expr_mse_unbiased_capped_norm` ≈ 1 (scaled; "no better than the control").

**What actually ran, this session, on your data (not simulated):**
- Built all three contexts: 120,000 predicted cells each (300 genes × 400 cells), 18,533 genes,
  raw integer counts copied verbatim from real control cells.
- Total stored (sparse) entries across all three contexts: **2,084,990,102**, comfortably under the
  4,750,000,000 cap.
- Validated all three files against the section-7 contract with `submission/validate_submission.py`
  (streams `.X` in 20,000-row chunks rather than loading it whole): **PASSED** — correct gene axis
  and order, correct 300-gene panel with no extras/missing, exactly 400 cells per perturbation per
  context, no non-targeting rows, all values non-negative raw integers, no cell over the 10⁶-count
  cap.
- **Could not compute the actual six metrics locally** — that requires Arc's held-out ground truth,
  which by definition doesn't exist outside their servers. `experiments/experiment_results.csv` logs
  this run with `metric=structural_validation, value=PASSED` rather than a fabricated score.
- Hit real resource limits doing this on a ~7.8GB-RAM, disk-constrained sandbox, worth knowing about
  before scaling up: a naive single-shot build of one context OOM-killed at ~4.6GB resident (fixed by
  chunking to 6,000-cell blocks); merging all three contexts into the one file the platform actually
  wants exceeded this session's disk allowance (fixed by validating three per-context files
  independently instead — see `README.md`'s Disk/memory section for exactly what broke and why).

This baseline is the floor everything else in your plan (B1–B5) needs to beat, and now we have a
concrete, validated artifact and pipeline for it rather than a description of one.

---

## 5. Leakage-safe zero-shot validation strategy (Phase 5)

Designed here; not yet implementable end-to-end because it needs the actual `vcc-perturb` gene list
and per-perturbation data, which is blocked (§2.2). Once unblocked:

- **Experiment A — held-out target genes.** Split `vcc-perturb`'s retained perturbations by gene
  into train/holdout sets (e.g. 80/20, stratified by effect size using `retained_perturbations_de.csv`'s
  DE-gene counts, so both splits span weak-to-strong effects). Train any gene-feature→effect model
  (Phase 11 features only — no gene-ID embedding lookup) on the train genes; evaluate transfer to the
  holdout genes. This is the most direct proxy for the real 2026 task, since 2026's 300 target genes
  are guaranteed unseen during training.
- **Experiment B — held-out context.** Not directly possible with `vcc-perturb` alone if it's
  single-context (H1 only, per §2.2) — this experiment needs a *second* perturbation dataset in a
  different cell line (see Phase 13 candidates) to hold out. Flagging this now: if `vcc-perturb` turns
  out to be exactly one context, Experiment B is unimplementable until we add a second real dataset,
  and the 2026 task is fundamentally a context-transfer problem — so this experiment matters as much
  as Experiment A, arguably more.
- **Experiment C — held-out gene×context combinations.** Only meaningful once we have ≥2 contexts,
  same dependency as B.
- **Leakage rule enforced throughout:** splits are done by (gene) or (gene, context) key, never by
  individual cell — cells from the same perturbation never span train and validation, matching your
  explicit instruction and the actual 2026 failure mode (memorizing a gene's effect vector rather than
  learning gene-feature → effect).

---

## 6. Ranked modeling strategy (Phase 6–15, prioritized)

| priority | approach | expected benefit | computational cost | dependency |
|---|---|---|---|---|
| P1 | B0 control-null | none (it's the floor) — but now implemented, validated, and gives us a real submission pipeline to build on | trivial | none — **done this session** |
| P1 | Gene-feature regression (Phase 6 Model 2 / Phase 11): baseline expression + coexpression + rank features → low-rank perturbation-effect vector (PCA/PLS/ridge), predicted per (gene, context) | Directly addresses the "unseen target gene" requirement (Phase 11's core point) — this is closest in spirit to what 3 of 4 2025 prize winners actually did (gene/protein-embedding-based transfer, not per-gene lookup) | low–moderate | needs real `vcc-perturb` DE/delta tables |
| P2 | Perturbation-effect lookup (Phase 6 Model 1) restricted to genes that actually overlap between `vcc-perturb`'s panel and 2026's 300-gene panel | Fast, simple, interpretable ceiling-check on the overlap subset only | trivial | needs `vcc-perturb` gene list (blocked) to even know overlap size |
| P2 | Context-specific effect scaling (Phase 8 Model B) using the 2026 controls' own per-context distributional differences (already measured in §2.1 — B has lower depth/detection than A/C) | Cheap way to at least partially correct for context shift without needing a second real perturbation context | low | none beyond current data |
| P3 | Cell-state matching / nearest-neighbor transfer (Phase 9), PCA-based first | Plausible improvement if H1→(A/B/C) baseline states are locally similar in some subspace; unproven | moderate | needs `vcc-perturb` cells (blocked) |
| P3 | Context-conditioned MLP / low-rank perturbation model (Phase 12) | Only justified if P1/P2 show real transfer signal on the local zero-shot benchmark — no evidence yet either way | moderate–high | needs P1 results first |
| P4 | Generative / autoencoder latent perturbation model, transformer/attention (Phase 10/12) | Per your own instruction, and per what we've learned about 2025 (Altos Labs' flow-matching model won the *Generalist* prize specifically, not the raw-accuracy prize) — worth real consideration eventually, but not before simpler methods are benchmarked | high | needs P1–P3 benchmarked first |
| — | Ensemble (Phase 15) | Premature until ≥2 real (non-baseline) models exist | — | needs P1–P3 |

The through-line: **almost every priority-1/2 item is blocked on the same thing** — real access to
`vcc-perturb`'s actual per-perturbation data. This isn't a minor gap; it's the dependency the entire
modeling phase sits on top of.

---

## 7. Baseline roadmap (Phase 4 status)

| stage | status |
|---|---|
| B0 — control null | **Done this session.** Implemented, run on real 2026 controls, validated against the real section-7 contract. See §4. |
| B1 — perturbation-effect lookup | Not started — needs `vcc-perturb` gene-level effect data (blocked, §2.2) |
| B2 — low-rank transfer | Not started — same dependency |
| B3 — context conditioning | Partially unblocked (we have real 2026 cross-context distributional differences, §2.1) but the *effect* side still needs real perturbation data |
| B4 — cell-state matching | Not started — needs `vcc-perturb` cells |
| B5 — ensemble | Not started — needs ≥2 real models |

---

## Addendum: `vcc-perturb.csv` audit and B2 gene-transfer model (Phase 2, 6, 11)

You supplied your own extract of `nicolas-lynn/vcc-perturb` as `vcc-perturb.csv` (750 rows × 18,081
columns) after `huggingface.co` access was confirmed blocked in this sandbox (§2.2). Audited directly
(`data/audit/audit_vcc_perturb.py`, full output in `data/audit/vcc_perturb_summary.json`) rather than
taking its shape or semantics on faith:

**Structure.** 150 unique target genes (`__index_level_0__`) × 5 rows each = 750 rows, over 18,080
gene-level response columns.

**Value semantics — verified, not assumed, and it's a real finding.** You explicitly asked not to
assume these are logFC. They are not logFC or raw deltas either: the most extreme values (up to
±220) concentrate almost entirely on a small, *perturbation-independent* set of genes — `ACTB`,
`TMSB4X`, `SCD`, `CD24`, `HMGCS1` — that show up as outliers across 80–135 of the 750 rows regardless
of which unrelated gene was actually knocked down. Real expression deltas or log2FC would be bounded
by biology and wouldn't blow up specifically on stably-expressed housekeeping genes irrespective of
the perturbation. This is the textbook signature of a **per-gene z-score with a very small reference
standard deviation in the denominator** (i.e. these are `perturbation_zscores.parquet`-style values,
not `perturbation_deltas.parquet` or `de_log2fc.parquet`). Treated as z-scores throughout. Values also
appear rounded to 2 decimals (9.2% are exactly 0.00), which loses precision on small true effects.

**Replicate quality.** Mean pairwise correlation among a gene's 5 rows is low (0.1–0.4 across a
15-gene sample) — consistent with the HF card's own caveat that these are random subsamples of one
cell pool, not independent biological replicates. All downstream work averages the 5 rows per gene
rather than treating them as 5 independent data points.

**Gene-axis overlap vs the 2026 18,533-gene feature axis: 18,077/18,533 (97.5%)** — high enough that
almost every 2026-axis gene has a real measured response value to work with; 456 axis genes have none
and need a fallback (zero-effect) when actually generating predictions.

**Target-panel overlap vs 2026's 300 target genes: only 13/300 (4.3%)** — `AKT2, HSBP1, MED13, MTA1,
RNF2, SHPRH, SIN3B, SMARCA5, STAT6, TARBP2, TRAPPC6A, TWF2, ZNF714`. This is the single most
important number in this addendum: **a direct perturbation-lookup baseline (B1) can, at best, cover
13 of 300 genes per context** and would fall back to a B0-style no-op for the other 95.7%. That's not
a viable primary strategy on its own — confirming, with real data rather than a prior, that the
gene-feature transfer approach was the right thing to prioritize (§6).

> **CORRECTION (added during B2 submission-generation work, after the numbers below were already
> shared):** the discrimination-score table immediately below was computed with a bug and is
> **wrong** — kept here struck through rather than deleted, because the correction is more important
> than the tidiness of hiding it. It embedded each held-out gene using its own *row* (what that
> gene's knockdown does to other genes) restricted to the other genes' columns. That embedding is
> only computable because the held-out gene happened to be drawn from the known 150 — a genuinely
> unknown 2026 target gene (287 of the 300 per context) has no row at all, since it was never itself
> knocked down. It only has a *column* (how the 150 known knockdowns affected it). Redone with
> column-based embeddings throughout (matching exactly what production actually has available; see
> `models/gene_transfer_b2.py`'s `leave_one_out_validation` docstring for the fixed derivation), the
> real result is: **discrimination score collapses to chance (0.49–0.53, chance=0.500) for every k
> from 1 to 20, both centered and uncentered.** There is no validated gene-specific zero-shot
> transfer signal in this dataset via this method. Two other candidate signals were checked and also
> ruled out: a gene's own baseline expression level does not predict its knockdown-response
> magnitude (Spearman ρ=0.13, p=0.11), and while a gene's own knockdown *efficacy* does correlate
> with its response magnitude (ρ=-0.22, p=0.006, real), that's not knowable in advance for a
> genuinely new gene, so it isn't usable as a transferable predictor either. Full detail and the
> corrected validation numbers are in the "B2 submission generation" addendum further down this
> document — this note exists so nobody reading top-to-bottom takes the paragraph below at face
> value.

**B2 model, built around transfer rather than lookup** (`models/gene_transfer_b2.py`): every gene in
the 18,080 response columns — including all 300 of 2026's targets, even the 287 never themselves
knocked down — gets a 150-dimensional "perturbation-response embedding": how much it moves under each
of the 150 known knockdowns. No gene *identity* is used as a lookup key, only its position in this
learned response space. To predict an unseen target gene's own knockdown effect, find its k nearest
neighbors in that embedding space among the 150 known genes and predict a similarity-weighted average
of their measured response vectors — a "guilt by association" transfer through shared response
behavior, not memorized per-gene effects.

**Validated before trusting it (Phase 5, Experiment A: leave-one-out over all 150 known targets),
and the validation surfaced a real, non-obvious tradeoff worth designing around:**

| k (neighbors) | raw cosine to truth | discrimination score* |
|---|---|---|
| 1 | 0.7275 | **0.774** |
| 3 | 0.7852 | 0.762 |
| 5 | 0.7963 | 0.754 |
| 10 | 0.8070 | 0.745 |
| 20 | **0.8085** | 0.721 |
| 40 | 0.8036 | 0.692 |
| global-mean baseline (predict the same vector for every gene) | 0.7307 | 0.500 (chance) |

*discrimination score is a local proxy for what `pds_cosine` (§1.3) actually measures — not "is the
prediction numerically close to the truth" but "does the prediction correctly identify *which* gene
was perturbed, out of all 150, by cosine similarity" (1.0 = always the closest match, 0.5 = chance).

**Raw cosine-to-truth improves with more neighbors; the discrimination score gets *worse*.** These
move in opposite directions because most of a response vector's raw magnitude is a shared,
gene-independent stress/arrest signature common to nearly every knockdown — averaging more neighbors
smooths toward that generic signature, which raises numerical closeness to any plausible truth while
erasing the gene-specific signal that distinguishes one perturbation from another. That the
global-mean baseline scores exactly 0.500 (chance) on discrimination while still hitting 0.73 raw
cosine is direct local confirmation of the exact failure mode the official 2026 metric spec was
rewritten to guard against (§1.5): a model can look numerically close to the truth by leaning on
shared structure alone, without discriminating anything — which is precisely why `pds_cosine`
excludes the full 300-gene panel from its feature space rather than just each row's own gene.
**Practical implication: k should be chosen per which of the six official metrics a given model
variant is optimizing for, not fixed once — low k favors discrimination-style metrics, higher k
favors raw-accuracy metrics like `expr_mse_unbiased_capped_norm`.**

Centering (removing the leave-one-out-honest mean response vector before the neighbor match, adding
it back to the prediction) partially decouples the tradeoff: at k=3 it reaches **discrimination
0.798** (the best found in this sweep) while still holding raw cosine at 0.762, better than
uncentered k=1's discrimination-optimal point. Full sweep in
`experiments/experiment_results.csv` (rows `B2-loo-*`) and reproducible via
`python models/gene_transfer_b2.py`.

**What this doesn't yet do:** produce an actual submission file. The response vectors here are
z-scores over 18,080 genes, not raw per-cell counts — turning a predicted z-score profile into
section-7-compliant integer counts (scaling real control cells by the predicted per-gene effect, at
the same 120,000-cells × 18,533-genes × 3-contexts scale B0 was built at) is a further engineering
step with the same memory/disk considerations as B0 (§4, README's Disk/memory section). Logged as
task #11 pending your go-ahead rather than built without checking in, given how much disk/RAM
headroom that took last time.

## Addendum 2: B2 submission generation — transformation validated, gene-transfer did not

You asked for the z-score→count transformation to be validated on real held-out 2025 data before
touching the 2026 controls, and for the gene-transfer step to use the best centered-KNN
configuration found so far. Doing this properly surfaced the correction above — worth reading before
this section, since it changes what "B2" actually is.

### Transformation selection (validated as asked, before any 2026 generation)

Four candidates (`models/transform_candidates.py`), none assumed correct going in:

- **T0_null** — ignore the z-score entirely (this is what B0 already does; the in-harness floor).
- **T1_multiplicative** — `fold_g = 2^(alpha·z_g)` applied to each real control cell's own raw count.
- **T2_log1p_additive** — shift each cell's own `log1p(CP10K)` value by `beta·z_g·sigma_g`, where
  `sigma_g` is a per-context reference standard deviation estimated by resampling 300 fifteen-cell
  pseudobulks from the real 2026 controls (`models/pseudobulk_reference.py`) — deliberately mirroring
  vcc-perturb's own stated construction ("50 pseudobulks of 15 cells"), algebraically inverting a
  z-score definition of that same shape.
- **T3_quantile** — shift each cell's percentile within this context's own empirical per-gene count
  distribution by a normal-quantile amount tied to `z_g`, then read off a new count from that same
  empirical distribution (no assumed multiplicative/log-normal shape).

All three real candidates draw final integer counts from `Multinomial(cell's own original total,
target proportions)` — guaranteeing non-negative integers summing to the cell's own total by
construction, and, as a stated modeling choice rather than a discovered property, preserving each
cell's own library size (a z-score computed on normalized data carries no information about an
absolute library-size shift, so there's nothing in the input to condition one on).

**Test**: for each of the 150 known vcc-perturb genes in turn, its REAL averaged z-score profile was
treated as if it were a model's prediction (per your instruction — this isolates transformation
quality from gene-prediction quality) and applied to real context-A control cells. Three things were
checked against real reference points: pseudobulk effect-recovery (does log2FC of synthetic-vs-control
mean agree in direction/magnitude with the input z, restricted to the top decile of genes by |z| so
near-zero-z noise genes don't dominate), a real Wilcoxon rank-sum DE test of synthetic-vs-control
compared against a z-derived pseudo-ground-truth significant-gene set (|z|≥2), and single-cell QC
(library size, detection, zero fraction, per-gene variance) for degeneracy, not for comparison to a
real perturbed reference we don't have (see the "what this can and cannot validate" note in
`models/validate_transform_b2.py` — there is no real single-cell perturbed data anywhere in this
session, only vcc-perturb's pseudobulk-level z-scores and real 2026 control single cells).

25-gene initial sweep, then confirmed on all 150 known genes for the two cheap candidates:

| candidate | param | direction agreement | magnitude corr. | DE Jaccard vs z-derived truth | composite score |
|---|---|---|---|---|---|
| T2_log1p_additive | β=1.5 | **0.845** | 0.179 | **0.072** | **0.365** |
| T2_log1p_additive | β=2.0 | 0.837 | 0.202 | 0.043 | 0.360 |
| T2_log1p_additive | β=2.5 | 0.820 | 0.222 | 0.026 | 0.356 |
| T1_multiplicative | α=1.0 | 0.701 | 0.324 | 0.009 | 0.345 |
| T1_multiplicative | α=2.0 | 0.568 | 0.411 | 0.005 | 0.328 |
| T0_null (floor) | — | 0.599 | -0.016 | 0.000 | 0.195 |
| T3_quantile (25-gene sweep only) | γ=0.5–1.5 | 0.66–0.80 | 0.17–0.27 | 0.004–0.006 | 0.279–0.359 |

**Selected: T2_log1p_additive, β=1.5.** It wins on composite score, has the highest and most
*consistent* direction agreement across genes (lowest std, 0.118 vs T1's 0.25–0.31), and stays close
to real controls on zero-fraction (0.72 vs 0.68 real) where T1 at higher α badly over-sparsifies
(up to 0.90) and T3 was dropped for a different reason: it was both the slowest candidate by ~50x
(a per-gene Python loop) *and* showed an unexplained anomaly — synthetic zero-fraction *below* real
controls (0.53–0.55 vs 0.68), i.e. it made cells systematically less sparse than reality, the opposite
direction of what T1/T2 show and not something I could explain in the time available. Flagged, not
used. **Honest caveat that survives the winner**: DE-signature Jaccard overlap is low in absolute
terms for every candidate (0.02–0.10, none close to 1.0) — likely a statistical-power limit at the
120-cell validation batch size (production uses the real 400) rather than a flaw specific to T2, but
it means DE-signature fidelity should be treated as unproven, not just "T2 is the best of a good set."

### Gene-effect prediction: corrected, and it did not validate

The "test nearby k" instruction was followed properly: k=2, 3, 4 centered were all checked
(0.793 / 0.798 / 0.793 discrimination in the flawed version) confirming k=3 was a real local optimum
*within that buggy setup*. But fixing the embedding bug (see Addendum 1's correction note) collapses
all of it to chance:

| k | uncentered discrimination | centered discrimination |
|---|---|---|
| 1 | 0.533 | 0.528 |
| 3 | 0.496 | 0.517 |
| 5 | 0.507 | 0.511 |
| 10 | 0.500 | 0.506 |
| 20 | 0.486 | 0.498 |
| chance | 0.500 | 0.500 |

No k, centered or not, clears chance by a margin that looks real rather than noise over 150 genes.
Two alternative transferable signals were also checked and ruled out: baseline expression level vs.
knockdown-response magnitude (Spearman ρ=0.13, p=0.11 — not significant), and a gene's own
knockdown *efficacy* vs. its response magnitude (ρ=-0.22, p=0.006 — real, but not knowable in advance
for a gene that's never been tested, so not usable as a predictor).

**What this means for B2, concretely:** there is no validated basis for predicting a *gene-specific*
knockdown effect for the 287/300 2026 target genes vcc-perturb never measured. What remains
defensible: (a) for the 13/300 genes vcc-perturb DID measure, use the real averaged z-score profile
directly — not a prediction, real data (still carrying the flagged, unverified H1→2026-cell-line
transfer assumption); (b) for the other 287/300, use the mean response profile across all 150 known
knockdowns — not a transfer prediction either, but the one non-gene-specific real structure in the
data (knockdowns share a common average response shape), which is measurably closer to real
perturbed profiles than predicting no effect at all (cosine ~0.73 vs 0 by construction) even though,
like B0, it can't discriminate between different genes' distinct effects. This construction is, not
coincidentally, close to the official metric spec's own scaling baseline `b` (report §1.3) — meaning
**B2, as validated, should be expected to roughly tie the documented baseline on ~96% of each
context's genes and only has a chance to beat it on the 13 real-overlap genes.**

I have not generated the full 2026 submission yet. The transformation (step 6 of your instructions)
validated cleanly and is ready; the gene-effect side (part of "B2 itself") did not, and rather than
quietly substituting the mean-response fallback and calling the result "B2" without you seeing this
finding first, I'm stopping here to explain it, per your own instruction to stop rather than produce
an arbitrary result when validation doesn't hold up. See the chat message for the options going
forward.

### Addendum 3: three more transfer signals tried, all closed out

Asked to investigate further before generating anything, rather than settling for the first null
result. Three more independent attempts, each with a different inductive bias, all against the same
corrected (column-based, self-row-excluded) leave-one-out harness so the discrimination-score numbers
are directly comparable to Addendum 2's:

1. **Real coexpression from the 2026 controls** (`models/coexpression_transfer.py`) — a genuinely
   different hypothesis (baseline co-regulation, not co-response-to-perturbation), computed entirely
   from real context-A single cells rather than vcc-perturb, and available for all 300 2026 targets
   with no missingness. Discrimination: **0.50–0.51 across k=1–40** — chance, like the first attempt.
   (One methodological note worth recording: an initial run using only 6,000 subsampled cells left
   18.5% of the gene axis with exactly zero detected counts — a subsampling artifact, not biology —
   which produced degenerate zero-variance embedding rows and NaN cosine similarities. Fixed by using
   the full 18,400-cell pool, which cut that to 1 real gene, `PRDM14`, excluded explicitly.)
2. **Baseline expression level and self-knockdown efficacy as scalar predictors of response
   magnitude** (reported in Addendum 2) — the first not statistically significant (p=0.11), the
   second real (p=0.006) but not usable since it requires already knowing how well a new gene's
   knockdown would work.
3. **PLS regression** (`models/pls_transfer.py`) on both embeddings and their concatenation — a
   global, regularized method with a different inductive bias than nearest-neighbor (better suited in
   principle to small-n, high-dimensional-output problems, so a real possibility that KNN's
   curse-of-dimensionality was hiding something). Tested at 5 and 15 latent components, on a 60-gene
   random LOO subset (full 150-gene PLS LOO would take roughly an hour per configuration at ~2.2s/fit
   — each fold refits from scratch): discrimination ranged 0.48–0.56, with the single highest point
   estimate (0.563, combined 300-dim embedding, 15 components) still within sampling noise for n=60
   (standard error ≈0.065) — not a result I'd act on.

**Conclusion, across four independent methods and two independent embedding sources**: there is no
validated gene-specific zero-shot transfer signal in the data available in this sandbox. This isn't
"KNN doesn't work" — it's that neither of the two real, available structural signals (how a gene
responds to a shared panel of perturbations, or how it's co-regulated at baseline) predicts its own
knockdown effect better than chance in this dataset, under either a local (KNN) or global
(regularized regression) method. Closing this investigation here rather than continuing to try
methods against signals already shown not to carry the needed information — a fifth method applied
to the same two embeddings is very unlikely to succeed where four didn't. Unblocking real
gene-specific transfer would need either a genuinely new feature source (protein/pathway embeddings —
blocked by the same network policy as the original HF dataset access) or more real perturbation data
covering a larger, more diverse gene panel than these 150.

### Addendum 4: B2 generated and validated

With no gene-specific transfer signal found, you asked me to proceed with the honest hybrid design.
Generated via `models/generate_b2_submission.py`, mirroring `control_baseline.py`'s memory-safe
chunking exactly (a smoke test on 4 genes ran clean before committing to the full build):

| context | shape | stored entries (nnz) |
|---|---|---|
| A | 120,000 × 18,533 | 626,985,766 |
| B | 120,000 × 18,533 | 577,370,770 |
| C | 120,000 × 18,533 | 616,226,894 |
| **total** | 360,000 cells | **1,820,583,430** / 4,750,000,000 cap |

`validate_submission.py`'s full section-7 contract check **PASSED** on all three files (exact gene
axis and order, exact 300-gene panel at exactly 400 cells each per context, no non-targeting rows,
non-negative raw integer counts, no cell over the count cap). Real-lookup genes' library sizes matched
their context's real controls closely (e.g. context A: 21,010 vs 21,034 mean counts/cell over a
2,000-cell sample) — expected, since T2 preserves each cell's own total by construction (§ Addendum
2's stated limitation on library-size shift still applies).

**B0 vs. B2**: B0 predicts the context's own real control distribution verbatim for every gene — by
definition, zero effect anywhere. B2 predicts a real, non-trivial shift for every gene: either that
gene's own true measured knockdown profile (13/300) or the shared mean-response profile (287/300),
both passed through the validated T2 transform. So B2 is a materially different, and more
defensible, submission than B0 — but per Addendum 2/3's findings, it should be expected to roughly
**tie** the officially documented mean-response baseline (report §1.3's `b`) on the 287/300 genes
using that fallback, and only has a real chance to beat it on the 13 genes with true overlap data
(itself still carrying the unverified H1-vs-2026-cell-line transfer assumption).

**Effect magnitude distribution** (L2 norm of the applied z-profile, context-invariant by
construction — see Addendum 2's stated limitation):

| | n genes | z-profile L2 norm |
|---|---|---|
| mean-response fallback | 287 | 40.9 (identical for all 287 — same fixed vector) |
| real-lookup (vcc-perturb overlap) | 13 | 47.8 – 70.0, mean 55.5 |

Every one of the 13 real-lookup genes shows a *larger* effect magnitude than the fallback (1.17×–1.71×)
— `SIN3B` and `SMARCA5` are the strongest (≈70), `TWF2` the weakest of the thirteen (≈48), still above
the fallback floor. This is the expected direction, not a red flag: averaging 150 individual profiles
cancels out each gene's idiosyncratic large entries, so the mean profile is necessarily "smoother" and
smaller in norm than most individual real profiles — a sanity check that the two effect sources
(lookup vs. fallback) are behaving distinctly rather than one accidentally collapsing into the other.
Full per-gene table: `experiments/b2_real_lookup_gene_effect_magnitudes.csv`.

**Genes/contexts that behave suspiciously**: none found beyond two properties that are expected by
design, not defects: (1) all 287 fallback-profile genes are, by construction, driven by the identical
z-profile — any cell-to-cell variation among them comes only from which real control cells were
sampled and the multinomial redraw's own randomness, not from anything gene-specific; (2) 456/18,533
axis genes (2.5%) have no vcc-perturb measurement at all and so receive exactly zero predicted shift
for every one of the 300 target genes — a real coverage gap in the input data, not a bug in the
pipeline.

**Not delivered as files**: the three generated `.h5ad` files (≈15GB combined) were deleted after
validation to conserve this sandbox's disk allowance, matching what was done for B0 — the validated,
re-runnable pipeline (`models/generate_b2_submission.py`, ~24 minutes for all three contexts) and
every validation artifact are in the project zip. Re-run it to regenerate the actual files for upload.
