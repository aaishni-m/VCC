"""
Semi-supervised context transfer: map a knockdown's measured effect in a SOURCE cell type into a TARGET cell type
using only the target's UNLABELLED control cells.

  unsupervised (target controls only): per-gene detection rate, and the gene programs of the target -- the PCA of
      its control cells' log-normalized, per-gene-standardized expression (V: genes x k, eigenvalues lam)
  supervised (source labels):          the source effect E_s(t) of each knockdown t (z vs source controls)

Methods (every one uses only information that also exists for contexts A/B/C, where no effects are measured):
  M0 raw       a * E_s
  M1 gated     a * E_s, zeroed on genes detected in < tau of target control cells
  M2 project   a * V_k V_k^T gate(E_s)                          (keep the part living in the target's top-k programs)
  M3 filter    a * V diag(lam / (lam + mu)) V^T gate(E_s)       (soft version: weight programs by target variance)

Validation (`python models/ssl_transfer.py DIR`): Replogle measured the same essential-gene knockdowns in K562 and RPE1,
with single-cell controls for both (kaggle/ssl_data.py extracts them into DIR). Each method predicts RPE1 from K562 and
K562 from RPE1; the TARGET's measured effects are only ever used to score, never to build the prediction -- except
the scalar scale `a`, which is fitted on one half of the knockdowns and scored on the other.

Scores per method (on held-out knockdowns):
  disc     pds-like: is the prediction closest (cosine) to its own true effect among all held-out knockdowns,
           with every perturbed gene's own column removed (0.5 chance, 1 perfect)
  r2       1 - ||a*pred - truth||^2 / ||truth||^2 summed over knockdowns (MSE-like; > 0 beats "no change")
  prec200  sign agreement with truth on each prediction's 200 largest genes
"""
import itertools
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp


def context_structure(ctrl: ad.AnnData, genes: np.ndarray, k_max: int = 400) -> dict:
    """Unsupervised description of a cell type from its control cells, on the given gene order."""
    X = ctrl[:, genes].X
    X = X.toarray() if sp.issparse(X) else np.asarray(X)
    X = X.astype(np.float64)
    det = (X > 0).mean(0)
    lib = ctrl.X.sum(1).A1 if sp.issparse(ctrl.X) else ctrl.X.sum(1)
    L = np.log1p(X / np.maximum(lib, 1)[:, None] * 1e4)
    sd = L.std(0)
    Zc = (L - L.mean(0)) / np.where(sd > 0, sd, 1)
    Zc[:, sd == 0] = 0
    _, s, Vt = np.linalg.svd(Zc - Zc.mean(0), full_matrices=False)
    k = min(k_max, Vt.shape[0])
    return {"det": det, "V": Vt[:k].T, "lam": (s[:k] ** 2) / (Zc.shape[0] - 1)}


def transfer(E: np.ndarray, st: dict, method: str, k: int = 50, tau: float = 0.0, mu: float = 1.0) -> np.ndarray:
    """E: (n_knockdowns, n_genes) source effects on the target's gene order. Returns unscaled predictions."""
    gate = (st["det"] >= tau).astype(float)
    Eg = E * gate
    if method == "raw":
        return E
    if method == "gated":
        return Eg
    V = st["V"][:, :k]
    if method == "project":
        return (Eg @ V) @ V.T * gate
    if method == "filter":
        w = st["lam"][:k] / (st["lam"][:k] + mu)
        return ((Eg @ V) * w) @ V.T * gate
    raise ValueError(method)


def score(P: np.ndarray, T: np.ndarray, drop_cols: np.ndarray, fit_rows: np.ndarray, test_rows: np.ndarray) -> dict:
    keep = np.setdiff1d(np.arange(T.shape[1]), drop_cols)
    Pk, Tk = P[:, keep], T[:, keep]
    a = float((Pk[fit_rows] * Tk[fit_rows]).sum() / max((Pk[fit_rows] ** 2).sum(), 1e-12))
    Pt, Tt = Pk[test_rows], Tk[test_rows]
    An = Pt / (np.linalg.norm(Pt, axis=1, keepdims=True) + 1e-12)
    Bn = Tt / (np.linalg.norm(Tt, axis=1, keepdims=True) + 1e-12)
    S = An @ Bn.T
    n = len(test_rows)
    rank = np.array([(S[i] > S[i, i]).sum() for i in range(n)])
    top = np.argsort(-np.abs(Pt), axis=1)[:, :200]
    prec = np.mean([np.mean(np.sign(Pt[i, top[i]]) == np.sign(Tt[i, top[i]])) for i in range(n)])
    return {"disc": float(np.mean(1 - rank / (n - 1))), "cos": float(np.mean(np.sum(An * Bn, 1))),
            "r2": float(1 - ((a * Pt - Tt) ** 2).sum() / (Tt ** 2).sum()), "prec200": float(prec), "a": a}


def evaluate(src: ad.AnnData, tgt: ad.AnnData, tgt_ctrl: ad.AnnData, label: str, seed: int = 0) -> pd.DataFrame:
    genes = np.array(sorted(set(src.var_names) & set(tgt.var_names) & set(tgt_ctrl.var_names)))
    targets = sorted(set(src.obs["target"]) & set(tgt.obs["target"]))
    def first_row(a):  # row position of each target's first profile
        t = a.obs["target"].to_numpy()
        return {g: i for i, g in reversed(list(enumerate(t)))}
    si, ti = first_row(src), first_row(tgt)
    E = src[[si[t] for t in targets], genes].X
    T = tgt[[ti[t] for t in targets], genes].X
    E, T = (np.asarray(m.toarray() if sp.issparse(m) else m, dtype=np.float64) for m in (E, T))
    strong = np.linalg.norm(T, axis=1) >= np.quantile(np.linalg.norm(T, axis=1), 0.5)   # knockdowns with a real effect
    gpos = {g: i for i, g in enumerate(genes)}
    drop = np.array([gpos[t] for t in targets if t in gpos])
    st = context_structure(tgt_ctrl, genes)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(targets))
    fit, test = order[: len(order) // 2], order[len(order) // 2:]
    test_strong = test[strong[test]]
    print(f"[{label}] {len(targets)} shared knockdowns, {len(genes)} shared genes, "
          f"{len(test)} held out ({len(test_strong)} with strong effects)", flush=True)
    configs = [("raw", {}), ("gated", {"tau": 0.05}), ("gated", {"tau": 0.2})]
    configs += [("project", {"k": k, "tau": t}) for k, t in itertools.product([10, 25, 50, 100, 200, 400], [0.0, 0.05])]
    configs += [("filter", {"k": 400, "mu": m, "tau": t}) for m, t in itertools.product([0.5, 2, 8, 32], [0.0, 0.05])]
    rows = []
    for method, kw in configs:
        P = transfer(E, st, method, **kw)
        for subset, rows_test in (("all", test), ("strong", test_strong)):
            rows.append({"pair": label, "method": method, **kw, "subset": subset, **score(P, T, drop, fit, rows_test)})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    d = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    eff = {c: ad.read_h5ad(d / f"effects_{c}.h5ad") for c in ("K562", "RPE1")}
    ctl = {c: ad.read_h5ad(d / f"controls_{c}.h5ad") for c in ("K562", "RPE1")}
    res = pd.concat([evaluate(eff["K562"], eff["RPE1"], ctl["RPE1"], "K562->RPE1"),
                     evaluate(eff["RPE1"], eff["K562"], ctl["K562"], "RPE1->K562")])
    res.to_csv(d / "ssl_transfer_results.csv", index=False)
    pd.set_option("display.width", 200)
    cols = ["method", "k", "tau", "mu"]
    summary = (res.assign(k=res.get("k"), tau=res.get("tau"), mu=res.get("mu"))
               .groupby(["subset"] + cols, dropna=False)[["disc", "cos", "r2", "prec200"]].mean()
               .reset_index().sort_values(["subset", "disc"], ascending=[True, False]))
    for subset in ("all", "strong"):
        print(f"\n=== averaged over both directions, held-out knockdowns: {subset} ===")
        print(summary[summary.subset == subset].drop(columns="subset").round(4).to_string(index=False))
