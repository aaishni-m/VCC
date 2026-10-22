# VCC 2026 next model — decision (22 Oct 2026, updated after the official-scorer H1 comparison)

**Recommendation: D — keep B5's structure (transferred direction + average-response term) as the primary candidate,
rebuilt from training lines only for compliance; do NOT adopt the equal-weight average as a replacement yet. Next: one
decomposition run to separate the emitter (T2 vs T5) from the average-response term.**

The official-scorer H1 comparison (reduced mode, report section 10) overturns the proxy result: with the emitter fixed,
the equal-weight average is statistically indistinguishable from B5-direction on all six metrics. What does move the
official score is (a) adding an average-response term (fid and jac up, significant) and (b) B5's full structure with
its T2 emitter (overall 0.229 vs 0.181), whose two ingredients are confounded in this run. Ridge and denoising remain
dropped on the proxy evidence.

## Evidence from the pseudobulk proxies (11 lines), still valid as far as it goes

- Beats zero on all 4 well-powered lines: ranking +0.28 to +0.36, direction +0.06 to +0.30 (paired intervals far above 0).
- Beats B5-direction on ranking in RPE1 (+0.008), HepG2 (+0.019), Jurkat (+0.027) and on direction in the same three
  (+0.007 each), paired intervals above 0; small gains spread over ~55 % of knockdowns rather than a few.
- Beats ridge and the learned average on all four lines (ridge −0.003 to −0.008, intervals below 0).
- Its single parameter (scale 0.39–0.44) is stable across all 11 folds and explains B5v3's leaderboard nmae loss.
- It is a fitted model (scale learned leave-one-line-out on public screens), which the rules allow.

## Evidence against (strongest first)

1. On H1, the only line in the challenge's 10x Flex chemistry, the gain over B5-direction is not significant
   (+0.009, interval −0.023 to +0.040), and only 38 % of H1 knockdowns improve.
2. All of the above are pseudobulk proxies; emission (400 cells, Wilcoxon calls) can change the ranking.
3. "B5-direction" is not B5 as submitted (no 0.5 × H1 average, no T2 emitter).
4. Effect sizes barely beat zero (r² ≤ 0.05; H1 r² negative), so mse is likely to stay at 0 and nmae to gain little.

## Decision table (official = H1, cell-eval2, reduced mode; proxy = 4 well-powered lines)

| Candidate | Held-out performance | Uncertainty | Beats B5? | Recommendation |
| --- | --- | --- | --- | --- |
| Zero | proxy rank 0.500 | — | No | Baseline only |
| K562 alone | proxy 0.813 | worse than equal on the H1 proxy | No | Drop |
| Zhu alone | proxy 0.624 | clearly worse | No | Drop |
| Equal average, fitted scale (T5) | official H1 0.181; proxy 0.820 | vs B5-direction: no official metric differs (pds +0.023, CI -0.017..+0.062) | No (official H1) | Do not adopt alone |
| Equal average + average response (T5) | official H1 0.194 | +0.035 fid, +0.009 jac vs equal (CIs > 0) | No (0.194 vs B5-like 0.229) | Keep as a component |
| Learned weighted average | proxy 0.816 | -0.003..-0.008 vs equal (proxy) | not evaluated officially | Drop |
| Ridge | proxy 0.816 | -0.003..-0.008 vs equal (proxy) | not evaluated officially | Drop |
| Denoised average | proxy 0.724 | -0.06..-0.12 vs equal (proxy) | No | Drop |
| B5-direction (T5) | official H1 0.176 | — | — | Reference |
| B5-direction + average response (T5) | official H1 0.190 | +0.047 fid, +0.009 jac vs B5-dir | — | Component |
| **B5-like as submitted (T2, training-line mean)** | **official H1 0.229** | beats equal on fid +0.14, jac +0.05, nmae +0.013, reach +0.035 (CIs > 0) | is the B5 structure | **Primary candidate; decompose T2 vs mean term** |
| B5v3-style x1 / x2 | proxy 0.810; leaderboard x2 = 0.107 | nmae overshoot | No | Drop the fixed scale |
| B5 exactly as submitted | cannot be scored on H1 without leakage (its mean term and scale were fitted on H1) | — | — | Fallback = B5-like rebuilt without copied rows |

## Strongest evidence against this recommendation

H1 rewards dense significance calling (its responses are dense), and T2 is a dense caller. On the real A/B/C
leaderboard, B5's T2 emission scored fid -0.003 and jac -0.010 (chance). The H1 gain of B5-like over the T5 variants is
mostly fid and jac, so it may not transfer to the challenge panel at all.

## Next action (smallest experiment that resolves the uncertainty)

1. Decomposition on H1, reusing the saved bundle (~1 h): the B5-like profile emitted with T5, and the equal-average +
   average-response profile emitted with T2. Two variants answer "emitter or mean term?".
2. Check the dense-calling concern directly: report n_pred (our significant genes) vs N_conf per knockdown for T2 and T5
   on H1, and compare with A/B/C (where N_conf passes the >= 10 gate for only 209-261 of 300 knockdowns).
3. In parallel, final-phase plumbing: vcc-cli 0.3.0 + packager self-test, D/E/F contexts, final target coverage.
