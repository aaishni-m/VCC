# Snapshot: B6 baseline (before semi-supervised experiments)

Saved 2026-10-01 so we can return here at any point. Git tag: **`snapshot-b6`**.

## What this state is
| Model | Status | Score |
|---|---|---|
| B5: 0.5 × H1 mean + 10 × external direction (Replogle K562 + Zhu T cells), T2 transform | submitted 2026-09-28, entry `uBfe774RImLP6LLmfOSx` | **Overall 0.1283** (pds 0.487, mse 0, nmae 0.084, fid −0.003, reach 0.212, jac −0.010) |
| B5v3: 2 × dense log2 fold change (Replogle K562 genome-wide + Zhu T cells), fold transform | submitted 2026-10-08, entry `WYupOQbBnAYGoT6copZt` | **Overall 0.1066**, rank 605 (pds 0.544, mse 0, nmae −0.058, fid −0.005, reach 0.159, jac 0.001) — below B5 and the team's previous 0.1457 |
| B6: B5's top-200 genes per perturbation, sparse T4 transform, streamed packaging | built and validated, **not submitted** | — |
| B4: B2 + mean profile + self-knockdown | built, not submitted | — |

Team TAM's leaderboard entry was later replaced by a teammate's "TeAM V23" (0.068, 2026-09-28 08:13 UTC).

## Where everything lives
**Code** — this repo at tag `snapshot-b6` (commit `e5fca90`).

**Kaggle (account `aaishnimulik`, all private)** — do NOT re-push these kernel IDs for new experiments; use new IDs so their outputs stay intact:
| Kaggle item | Holds | State at snapshot |
|---|---|---|
| dataset `vcc-data` | controls bundle (A/B/C, gene_names, pert_counts, manifest) + `vcc-perturb.csv` | version of 2026-09-27 14:15 UTC |
| dataset `vcc-code` | this repo's code | version of 2026-09-28 10:53 UTC (= `1c8a673`, same code as the tag for everything the kernels run) |
| dataset `vcc-snapshot-b6` | model artifacts below | created with this snapshot |
| kernel `vcc-external` (v5) | `external_profiles.npz`, `external_results.json` | feeds B5/B6 |
| kernel `vcc-run` (v2) | `b4_submission.vcc` | |
| kernel `vcc-run-b5` (v1) | `b5_submission.vcc` (the submitted file) | |
| kernel `vcc-run-b6` (v4) | `b6_submission.vcc` (3.3 GB, validated) | |
| kernel `vcc-surrogate` (v1) | uncalibrated surrogate grid (see caveat in its log) | |

**Local** — `C:\VCC\outputs\b5\b5_submission.vcc` (the exact submitted B5 file).

## Artifacts in `vcc-snapshot-b6`
`external_profiles.npz`, `external_results.json`, `models/{b4,b5,b6}_profiles.npy` + `_meta.json`,
`surrogate_grid.csv`, this file.

## How to restore
```bash
git -C C:/VCC checkout snapshot-b6
```
- Rebuild B6 without re-running anything upstream: run `kaggle/run_kaggle.py` with MODEL=b6 on Kaggle with
  sources `vcc-code`, `vcc-data`, kernel `vcc-external` (or attach `vcc-snapshot-b6` for `external_profiles.npz`).
  Generation is seeded (seed 0), so the same inputs give the same file.
- Submit an existing file: `vcc submit <file>.vcc -m <name> -d <description> --wait` (VCC token from
  `C:\Users\User\.kaggle\vcc_api.txt`, never printed).
