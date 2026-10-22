# VCC 2026 next model — decision (22 Oct 2026, provisional until the H1 official benchmark returns)

**Recommendation: B (simplified) — advance the equal-weight source average with a fitted scale (~0.4×); do not continue
ridge or denoising; keep B5 as the fallback until the official H1 benchmark confirms.**

Every source-based model beats zero by a wide margin on ranking and direction, but the four sources carry little
effect-size information (best r² 0.05). Among them, the plain equal-weight average with one fitted shrink factor is the
best or tied-best candidate on all four well-powered held-out lines; learning per-source weights or a ridge model adds
nothing, and denoising hurts.

## Evidence for

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

## Decision table

| Candidate | Held-out performance (proxy ranking, well-powered mean) | Uncertainty | Beats B5? | Recommendation |
| --- | --- | --- | --- | --- |
| Zero | 0.500 | — | No | Baseline only |
| K562 alone | 0.813 | worse than equal on H1 (−0.034, CI excludes 0) | not on H1 | Drop |
| Zhu alone | 0.624 | clearly worse | No | Drop |
| **Equal average, fitted scale** | **0.820** | +0.008 to +0.027 vs B5-dir on 3/4 lines; H1 n.s. | B5-direction: yes on 3/4 proxy lines; B5 as submitted: unknown | **Advance; confirm on official H1** |
| Learned weighted average | 0.816 | −0.003 to −0.008 vs equal | unknown | Drop (no gain, weights favour a source with 11/300 coverage) |
| Ridge | 0.816 | −0.003 to −0.008 vs equal | unknown | Drop in this form |
| Denoised average | 0.724 | −0.06 to −0.12 vs equal | No | Drop (helps only on weak lines) |
| Ridge + denoise | 0.716 | as above | No | Drop |
| B5-direction (proxy of B5) | 0.805 | — | — | Reference |
| B5v3-style ×1 / ×2 | 0.810 / 0.810 | nmae on H1 1.06 / 1.37 (overshoot) | No (×2 lost on leaderboard) | Drop the fixed scale |
| B5 as submitted | not evaluated under this protocol | — | — | Fallback for the final |

## Next action

1. Read the official H1 benchmark (running on Kaggle): adopt the equal average only if its overall scaled score on H1
   beats B5-direction and neither nmae nor reach collapses.
2. Smallest experiment to resolve the H1 doubt if it is a tie: the same comparison with B5's 0.5 × average-response
   term added to both (built from the training lines' mean response, never H1's), on H1 only.
3. Independently, start the final-phase plumbing (vcc-cli 0.3.0, D/E/F contexts, final target coverage).
