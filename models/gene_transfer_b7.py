"""
B7 -- external-direction profiles that KEEP each knockdown's effect size, without the H1 mean component, top-K sparse,
generated with the T4 transform (like B6).

Why (see models/ssl_transfer.py validation on Replogle K562 <-> RPE1, 2,055 shared knockdowns):
  - A knockdown's effect SIZE transfers between cell types (Spearman 0.61), and keeping source magnitudes instead of
    unit-normalizing them raised explained error 0.032 -> 0.051 with no change to discrimination (direction-only).
    B5/B6 unit-normalized every knockdown's direction, giving weak and strong knockdowns the same predicted size.
  - The vcc2026 spec measures the context mean response at 0.986-0.992 on the expression metric (1 = predicting no
    change): nearly worthless, so B5/B6's 0.5 x H1-mean component is dropped.
  - Top-K sparsity kept discrimination in the same validation (0.8035 at K=200 vs 0.8019 dense).

Construction, per target t without a real vcc-perturb profile:
    P(t) = mean over available sources s in {K562, T_mean} of  a_s * D_s(t)
  D_s = the source's centered effect (kaggle/external_profiles.py), a_s = least-squares scale of that source onto
  vcc-perturb's gene-specific responses (R - leave-one-out mean), fitted on the knockdowns the source covers. Then one
  global multiplier puts the median knockdown's top-K values at B6's level -- the effect size at which B6's genes are
  detected by the official Wilcoxon test (diagnostics/de_sim.py: ~90 calls/perturbation, all in the predicted
  direction) -- while weaker/stronger knockdowns stay proportionally weaker/stronger. Calling too few genes is
  heavily penalized by direction fidelity (no calls -> about -1.6 scaled), so detectability is kept on purpose.
The 13 targets vcc-perturb measured keep their real profile. Everything is then cut to each row's top-K genes.

Usage:
    python models/gene_transfer_b7.py [--top-k 400] [--npz path]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

import config as cfg
from models.gene_transfer_b2 import discrimination_score, load_response_matrix
from models.gene_transfer_b6 import sparsify

SOURCES = ["K562", "T_mean"]


def source_rows(npz, prefix, self_cols):
    out = {}
    for s in SOURCES:
        D = npz[f"{prefix}{s}"].astype(np.float64)
        have = ~np.isnan(D).all(1) & (np.abs(np.nan_to_num(D)).sum(1) > 0)
        D = np.nan_to_num(D)
        ok = self_cols >= 0
        D[np.where(ok)[0], self_cols[ok]] = 0.0
        out[s] = (D, have)
    return out


def median_topk_level(profiles, rows, k):
    vals = [np.sort(np.abs(profiles[i]))[-k:] for i in rows if np.abs(profiles[i]).sum() > 0]
    return float(np.median(np.concatenate(vals)))


def main(top_k, npz_path):
    npz = np.load(npz_path)
    targets, gene_cols, R = load_response_matrix()
    assert list(npz["vp_genes"]) == list(gene_cols)
    col = {g: i for i, g in enumerate(gene_cols)}
    n = len(targets)
    vp_self = np.array([col[g] for g in targets])
    resid = R - (R.sum(0) - R) / (n - 1)
    resid[np.arange(n), vp_self] = 0.0

    # per-source least-squares scale onto vcc-perturb's gene-specific responses
    vp = source_rows(npz, "Dvp_", vp_self)
    a = {}
    for s, (D, have) in vp.items():
        a[s] = float((D[have] * resid[have]).sum() / max((D[have] ** 2).sum(), 1e-12))
    print(f"[B7] per-source LS scales: {a}")

    def combine(src):
        num = np.zeros_like(next(iter(src.values()))[0])
        cnt = np.zeros(num.shape[0])
        for s, (D, have) in src.items():
            num[have] += a[s] * D[have]
            cnt += have
        return num / np.maximum(cnt, 1)[:, None], cnt

    # local check on vcc-perturb (H1), target genes removed: direction quality is unchanged by magnitudes
    P_vp, cnt_vp = combine(vp)
    idx = np.where(cnt_vp > 0)[0]
    keep = np.setdiff1d(np.arange(R.shape[1]), np.unique(vp_self))
    disc = discrimination_score(sparsify(P_vp, top_k)[idx][:, keep], R[idx][:, keep])
    r2 = 1 - ((P_vp[idx] - resid[idx]) ** 2).sum() / (resid[idx] ** 2).sum()
    print(f"[B7] vcc-perturb check on {len(idx)} knockdowns: disc(top-{top_k}) {disc:.3f}; explained "
          f"gene-specific variance at LS scale {r2:.4f}")

    target_genes = pd.read_csv(cfg.PERT_COUNTS_CSV)[cfg.PERT_COL].str.strip().tolist()
    assert list(npz["targets"]) == target_genes
    self_cols = np.array([col.get(g, -1) for g in target_genes])
    P, cnt = combine(source_rows(npz, "D_", self_cols))
    known = {g: i for i, g in enumerate(targets)}
    is_real = np.array([g in known for g in target_genes])
    modelled = np.where(~is_real & (cnt > 0))[0]

    # global multiplier: median knockdown's top-K values at B6's detectable level
    b6_path = cfg.PROJECT_ROOT / "models" / "b6_profiles.npy"
    if b6_path.exists():
        b6 = np.load(b6_path).astype(np.float64)
        ref_level = median_topk_level(b6, modelled, min(top_k, 200))
    else:
        raise SystemExit("models/b6_profiles.npy needed for the reference effect level (run gene_transfer_b5/b6 first)")
    mult = ref_level / median_topk_level(P, modelled, min(top_k, 200))
    print(f"[B7] global multiplier {mult:.2f} (= {mult:.2f} x LS scale); reference |z| level {ref_level:.3f}")

    profiles = np.zeros((len(target_genes), R.shape[1]))
    source = []
    for qi, gene in enumerate(target_genes):
        if is_real[qi]:
            profiles[qi] = R[known[gene]]
            source.append("real")
            continue
        profiles[qi] = mult * P[qi]
        source.append("external" if cnt[qi] > 0 else "none")
    profiles = sparsify(profiles, top_k)
    sizes = np.linalg.norm(profiles[modelled], axis=1)
    print(f"[B7] profiles: {pd.Series(source).value_counts().to_dict()}; modelled effect-size spread "
          f"(10/50/90th pct of row norms): {np.percentile(sizes, [10, 50, 90]).round(2).tolist()}")

    out = cfg.PROJECT_ROOT / "models"
    np.save(out / "b7_profiles.npy", profiles.astype(np.float32))
    (out / "b7_meta.json").write_text(json.dumps({
        "target_genes": target_genes, "is_real": is_real.tolist(), "source": source, "top_k": top_k,
        "ls_scales": a, "global_multiplier": mult, "reference_level": ref_level, "transform": "t4_sparse_thinning",
        "vp_check": {"disc_topk": disc, "r2_ls": r2}}, indent=2))
    print("[B7] saved models/b7_profiles.npy and models/b7_meta.json")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--top-k", type=int, default=400)
    p.add_argument("--npz", default=None)
    args = p.parse_args()
    npz = args.npz or next((str(x) for root in (Path("/kaggle/input"), Path.cwd()) if root.exists()
                            for x in root.rglob("external_profiles.npz")), None)
    if npz is None:
        sys.exit("external_profiles.npz not found; pass --npz")
    main(args.top_k, npz)
