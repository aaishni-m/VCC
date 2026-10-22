"""
Extra H1 benchmark variants (VCC_NEXT_MODEL_DECISION.md next action 2), built WITHOUT any H1 response.

1. B5like_T2 -- B5's structure as submitted, emitted with B5's T2 emitter by kaggle/localbench_h1.py:
     direction d = unit(K562 z) + unit(Zhu z), scaled so its median top-200 |z| = 0.94 (B5's measured level)
     mean term  m = the TRAINING lines' average response (direction only), scaled to 0.576 x median ||d|| (B5's
                measured ratio of its 0.5 x H1-mean component to its transferred component)
     profile in z units (T2: log1p(CP10K) shift = 1.5 x z x sigma_g, sigma_g from H1 released-control pseudobulks).
   Exact B5 cannot be scored on H1 without leakage: its mean term IS the 2025 H1 mean response and its scale a = 2.51
   was least-squares fitted on the same H1 knockdowns.
2. Average-response ablation, T5 emitter, log2 units:
     eq_plus_mean    = alpha * equal_source_average + beta * M
     b5dir_plus_mean = alpha * b5_direction        + beta * M
   M for H1 = mean over the 10 training lines of each line's average response (genes present in >= 1 line).
   (alpha, beta) fitted by least squares on the training lines, lines weighted equally, where for training line L the
   mean term is built from the other 9 training lines (so L's own response never predicts L).
Output: results/h1_ablation_profiles.npz {genes, perts, V_<name>, kinds} merged by the bench with h1_profiles.
"""
import json
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, "C:/VCC")
from models.xline_models import combine, load  # noqa: E402

LINES = ["RPE1", "HepG2", "Jurkat", "HCT116", "HEK293T", "Jiang_a549", "Jiang_bxpc3", "Jiang_hap1", "Jiang_ht29", "Jiang_mcf7"]
EQ = {"k562": 1.0, "zhu": 1.0, "hct116": 1.0, "hek293t": 1.0}


def unit_rows(M):
    M = np.nan_to_num(M.astype(np.float64))
    return M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-12)


def b5dir(t):
    out = np.zeros(t["Y"].shape)
    for s in ("k562", "zhu"):
        z = t.get(f"Z_{s}")
        if z is not None:
            out += unit_rows(z)
    return out


def line_mean(t):
    """Average response of a line on its own genes (NaN where never tested)."""
    with np.errstate(invalid="ignore"):
        return np.nanmean(np.where(t["tested"], t["Y"], np.nan), axis=0)


def mean_term(tables, genes, exclude=()):
    """Mean over lines (excluding `exclude`) of each line's average response, mapped onto `genes`; 0 where absent."""
    acc = np.zeros(len(genes)); cnt = np.zeros(len(genes)); pos = {g: i for i, g in enumerate(genes)}
    for t in tables:
        if t["line"] in exclude:
            continue
        m = line_mean(t)
        for j, g in enumerate(t["genes"]):
            i = pos.get(g)
            if i is not None and np.isfinite(m[j]):
                acc[i] += m[j]; cnt[i] += 1
    return np.divide(acc, cnt, out=np.zeros_like(acc), where=cnt > 0)


def fit_two(tables, feat):
    """(alpha, beta) for y ~ alpha*feat + beta*M(-L), pooled normal equations with each line weighted equally."""
    G = np.zeros((2, 2)); b = np.zeros(2)
    for t in tables:
        m = t["tested"]; y = np.nan_to_num(t["Y"])[m]
        x1 = feat(t)[m]
        Mo = mean_term(tables, t["genes"], exclude=(t["line"],))
        x2 = np.broadcast_to(Mo, t["Y"].shape)[m]
        A = np.stack([x1, x2], 1); n = len(y)
        G += A.T @ A / n; b += A.T @ y / n
    return np.linalg.solve(G + 1e-9 * np.eye(2), b)


def topk_level(M, k=200):
    return float(np.median(np.sort(np.abs(M), axis=1)[:, -k:]))


def main():
    train = [load(L) for L in LINES]
    h1 = load("H1")
    M_h1 = mean_term(train, h1["genes"])
    out, kinds, info = {}, {}, {}
    # 1. B5-like as submitted (z units, T2)
    d = b5dir(h1)
    covered = np.abs(d).sum(1) > 0
    d = d / topk_level(d[covered]) * 0.94
    m = M_h1 / (np.linalg.norm(M_h1) + 1e-12) * np.median(np.linalg.norm(d[covered], axis=1)) * 0.576
    out["B5like_T2"] = (d + m[None, :]).astype(np.float32); kinds["B5like_T2"] = "t2_z"
    info["B5like_T2"] = {"direction_top200_level": 0.94, "mean_ratio": 0.576, "covered_kd": int(covered.sum())}
    # 2. ablation
    for name, feat in (("eq_plus_mean", lambda t: combine(t, EQ)), ("b5dir_plus_mean", b5dir)):
        a, bta = fit_two(train, feat)
        out[name] = (a * feat(h1) + bta * M_h1[None, :]).astype(np.float32); kinds[name] = "lfc"
        info[name] = {"alpha": round(float(a), 4), "beta_mean": round(float(bta), 4)}
    perts = np.array([q.split(":")[-1] for q in h1["perts"]])
    np.savez_compressed("C:/VCC/results/h1_ablation_profiles.npz", genes=h1["genes"], perts=perts,
                        kinds=json.dumps(kinds), **{f"V_{k}": v for k, v in out.items()})
    Path("C:/VCC/results/h1_ablation_config.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info, indent=1))
    print("mean term: genes with a value", int((M_h1 != 0).sum()), "of", len(M_h1), "| norm", round(float(np.linalg.norm(M_h1)), 3))


if __name__ == "__main__":
    main()
