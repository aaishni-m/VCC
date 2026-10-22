"""
Leave-one-line-out experiments for the next model (VCC_NEXT_MODEL_PLAN.md §3), on the table from xline_data.py.

Every candidate maps source log2FCs (k562, zhu, hct116, hek293t; plus their significance) to a held-out line's log2FC.
Anything fitted (scale alpha, source weights, ridge lambda, shrink threshold) is fitted on the OTHER lines only;
hyperparameters are chosen by an inner leave-one-line-out over those training lines. The held-out line's responses are
used only to score.

Pseudobulk metrics per held-out line (proxies of the official ones; the official scorer is run separately on H1/RPE1):
  pds_proxy  for each knockdown, rank of its own real response among all real responses of the line by cosine distance
             to the prediction; 1 - rank/(n-1), averaged (0.5 = chance). Genes that are targets in the line excluded.
  dir_sig    sign agreement on reference-significant genes (FDR < 0.05); a zero prediction counts as 0.5
  nmae_sig   mean|pred - y| / mean|y| on reference-significant genes (1.0 = predicting zero; lower is better)
  r2         1 - sum (pred - y)^2 / sum y^2 over tested genes (0 = predicting zero; higher is better)
Knockdowns with no source coverage get a zero prediction and stay in the averages (that is what a submission does).

Usage: python models/xline_models.py [--lines H1,RPE1,...] [--out results/VCC_NEXT_MODEL_PSEUDOBULK.csv]
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

TAB = Path("D:/VCC_outputs/xline")
SOURCES = ("k562", "zhu", "hct116", "hek293t")
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)


def load(line):
    d = np.load(TAB / f"{line}.npz", allow_pickle=False)
    t = {k: d[k] for k in d.files}
    t["line"] = line
    t["tested"] = np.isfinite(t["Y"])
    t["sig"] = t["tested"] & (np.nan_to_num(t["FDR"], nan=1.0) < 0.05)
    tg = {p.split(":")[-1] for p in t["perts"]}
    t["gene_keep"] = ~np.isin(t["genes"], list(tg))          # pds-style: line's own targets out of the feature space
    return t


# ---------------- source transforms ----------------
def src(t, s, shrink=None):
    """Source log2FC on the line's grid, NaN where missing; optional z-based soft shrinkage x * max(0, 1 - t/|z|)."""
    x = t.get(f"X_{s}")
    if x is None:
        return None
    x = x.astype(np.float64)
    if shrink:
        z = np.abs(np.nan_to_num(t[f"Z_{s}"].astype(np.float64)))
        x = x * np.clip(1 - shrink / np.maximum(z, 1e-9), 0, None)
    return x


def combine(t, weights, shrink=None):
    """Weighted mean over the sources a knockdown has (renormalised per knockdown); NaN-free (missing -> 0)."""
    num = np.zeros(t["Y"].shape); den = np.zeros((t["Y"].shape[0], 1))
    for s, w in weights.items():
        x = src(t, s, shrink.get(s) if isinstance(shrink, dict) else shrink)
        if x is None or w == 0:
            continue
        have = ~np.isnan(x).all(1, keepdims=True)
        num += w * np.nan_to_num(x); den += w * have
    return np.divide(num, den, out=np.zeros_like(num), where=den > 0)


# ---------------- fitting on training lines ----------------
def fit_scale(tables, predict):
    """alpha = sum x*y / sum x^2 over tested genes, each line weighted equally (so big lines don't dominate)."""
    num = den = 0.0
    for t in tables:
        p = predict(t); m = t["tested"]
        y = np.nan_to_num(t["Y"])
        sxy, sxx = float((p * y)[m].sum()), float((p * p)[m].sum())
        if sxx > 0:
            num += sxy / sxx * 1.0; den += 1.0
    return num / den if den else 0.0


def fit_ridge(tables, lam, shrink=None):
    """Shared weights over standardised source columns (+ an 'any source' indicator), ridge lambda; lines weighted equally.
    Returns {source: weight} in original units and the column scales."""
    G = np.zeros((len(SOURCES), len(SOURCES))); b = np.zeros(len(SOURCES)); scale = np.ones(len(SOURCES))
    cols = []
    for t in tables:
        m = t["tested"]
        Xs = [np.nan_to_num(src(t, s, shrink)) if src(t, s) is not None else np.zeros(t["Y"].shape) for s in SOURCES]
        cols.append((np.stack([x[m] for x in Xs], 1), np.nan_to_num(t["Y"])[m]))
    sd = np.sqrt(np.mean([np.mean(A ** 2, 0) for A, _ in cols], 0)); sd[sd == 0] = 1
    for A, y in cols:
        A = A / sd; n = len(y)
        G += A.T @ A / n; b += A.T @ y / n
    w = np.linalg.solve(G / len(cols) + lam * np.eye(len(SOURCES)), b / len(cols)) / sd
    return dict(zip(SOURCES, w))


def ridge_predict(t, w, shrink=None):
    out = np.zeros(t["Y"].shape)
    for s, ws in w.items():
        x = src(t, s, shrink)
        if x is not None:
            out += ws * np.nan_to_num(x)
    return out


# ---------------- metrics ----------------
def metrics(t, P):
    Y = np.nan_to_num(t["Y"]); m = t["tested"]; sig = t["sig"]; gk = t["gene_keep"]
    n = len(Y)
    Yp = np.where(m, Y, 0.0)[:, gk]; Pp = np.where(m, P, 0.0)[:, gk]
    Yn = Yp / (np.linalg.norm(Yp, axis=1, keepdims=True) + 1e-12); Pn = Pp / (np.linalg.norm(Pp, axis=1, keepdims=True) + 1e-12)
    S = Pn @ Yn.T                                               # cosine similarity pred_i vs real_j
    rank = (S > np.diag(S)[:, None]).sum(1) + 0.5 * ((S == np.diag(S)[:, None]).sum(1) - 1)
    pds = 1 - rank / max(n - 1, 1)
    pds[~np.any(Pp != 0, axis=1)] = 0.5                          # no prediction = chance
    agree = np.where(P == 0, 0.5, (np.sign(P) == np.sign(Y)).astype(float))
    dir_sig = np.array([agree[i, sig[i]].mean() if sig[i].any() else np.nan for i in range(n)])
    nmae = np.array([np.abs(P[i, sig[i]] - Y[i, sig[i]]).mean() / np.abs(Y[i, sig[i]]).mean() if sig[i].sum() >= 10
                     else np.nan for i in range(n)])
    r2_num = ((P - Y) ** 2 * m).sum(1); r2_den = (Y ** 2 * m).sum(1)
    covered = np.any(P != 0, axis=1)
    return {"pds_proxy": pds, "dir_sig": dir_sig, "nmae_sig": nmae, "r2_num": r2_num, "r2_den": r2_den,
            "covered": covered}


def summarise(line, name, cfg, M, rng):
    n = len(M["pds_proxy"])
    boot = lambda v: np.nanpercentile([np.nanmean(v[rng.integers(0, n, n)]) for _ in range(200)], [2.5, 97.5])  # noqa: E731
    r2 = 1 - M["r2_num"].sum() / M["r2_den"].sum()
    row = {"line": line, "model": name, "config": json.dumps(cfg), "n_knockdowns": n,
           "covered_frac": float(M["covered"].mean())}
    for k in ("pds_proxy", "dir_sig", "nmae_sig"):
        lo, hi = boot(M[k]); row[k] = float(np.nanmean(M[k])); row[k + "_ci"] = f"{lo:.3f}..{hi:.3f}"
    row["r2"] = float(r2)
    return row


# ---------------- candidates ----------------
def candidates(train, inner_lines):
    """name -> (config, predict(t)) fitted on `train` only."""
    out = {"zero": ({}, lambda t: np.zeros(t["Y"].shape))}
    eq = {s: 1.0 for s in SOURCES}
    for s in SOURCES:                                            # each source alone, scale fitted
        f = lambda t, s=s: combine(t, {s: 1.0})                  # noqa: E731
        a = fit_scale(train, f)
        out[f"src_{s}"] = ({"alpha": round(a, 4)}, lambda t, f=f, a=a: a * f(t))
    for c in (1.0, 2.0):                                         # B5v3-style: equal mean x fixed c (k562 + zhu only)
        out[f"b5v3_c{c:g}"] = ({"c": c}, lambda t, c=c: c * combine(t, {"k562": 1.0, "zhu": 1.0}))
    f = lambda t: combine(t, eq)                                 # noqa: E731
    a = fit_scale(train, f)
    out["B_equal_fitted"] = ({"alpha": round(a, 4)}, lambda t, a=a: a * combine(t, eq))
    # A: denoise by each source's own |z|, threshold chosen by inner LOLO on the training lines
    best = None
    for th in (0.5, 1.0, 2.0, 3.0):
        sc = []
        for v in inner_lines:
            tr = [t for t in train if t["line"] != v]; te = [t for t in train if t["line"] == v][0]
            ff = lambda t, th=th: combine(t, eq, shrink=th)      # noqa: E731
            aa = fit_scale(tr, ff)
            sc.append(np.nanmean(metrics(te, aa * ff(te))["pds_proxy"]))
        if best is None or np.mean(sc) > best[1]:
            best = (th, np.mean(sc))
    th = best[0]
    ff = lambda t, th=th: combine(t, eq, shrink=th)              # noqa: E731
    a = fit_scale(train, ff)
    out["A_denoise_equal"] = ({"shrink_z": th, "alpha": round(a, 4)}, lambda t, a=a, ff=ff: a * ff(t))
    # C: shared ridge over sources, lambda by inner LOLO (on r2, the magnitude-sensitive proxy)
    best = None
    for lam in (0.0, 0.01, 0.1, 1.0, 10.0):
        sc = []
        for v in inner_lines:
            tr = [t for t in train if t["line"] != v]; te = [t for t in train if t["line"] == v][0]
            w = fit_ridge(tr, lam)
            M = metrics(te, ridge_predict(te, w)); sc.append(1 - M["r2_num"].sum() / M["r2_den"].sum())
        if best is None or np.mean(sc) > best[1]:
            best = (lam, np.mean(sc))
    w = fit_ridge(train, best[0])
    out["C_ridge"] = ({"lambda": best[0], "w": {k: round(v, 4) for k, v in w.items()}}, lambda t, w=w: ridge_predict(t, w))
    w2 = fit_ridge(train, best[0], shrink=th)
    out["D_ridge_denoise"] = ({"lambda": best[0], "shrink_z": th, "w": {k: round(v, 4) for k, v in w2.items()}},
                              lambda t, w2=w2: ridge_predict(t, w2, shrink=th))
    return out


def main(lines, out_csv):
    tables = {L: load(L) for L in lines}
    log("loaded", {L: tables[L]["Y"].shape for L in lines})
    rng = np.random.default_rng(0)
    rows = []
    for held in lines:
        train = [tables[L] for L in lines if L != held]
        inner = [t["line"] for t in train]
        cands = candidates(train, inner)
        for name, (cfg, pred) in cands.items():
            rows.append(summarise(held, name, cfg, metrics(tables[held], pred(tables[held])), rng))
        log(held, "done:", {r["model"]: round(r["pds_proxy"], 3) for r in rows if r["line"] == held})
    df = pd.DataFrame(rows)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    summ = df.groupby("model")[["pds_proxy", "dir_sig", "nmae_sig", "r2"]].mean().sort_values("pds_proxy", ascending=False)
    print(summ.round(4).to_string())
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--lines", default="H1,RPE1,HepG2,Jurkat,HCT116,HEK293T,Jiang_a549,Jiang_bxpc3,Jiang_hap1,Jiang_ht29,Jiang_mcf7")
    ap.add_argument("--out", default="C:/VCC/results/VCC_NEXT_MODEL_PSEUDOBULK.csv")
    a = ap.parse_args()
    main(a.lines.split(","), a.out)
