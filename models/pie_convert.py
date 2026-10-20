"""
Turn PIE predictions (Verma et al. 2026; `pie infer` parquet: per gene p_de, lfc_pred, delta_p_pred) into the log2 fold
profiles our count emitter (transform_candidates.t5_log2_fold / localbench emit_fold) applies to real control cells.

Modes (all return log2 folds; 0 = gene untouched):
  lfc_de    lfc_pred on genes PIE calls DE (p_de > thr), 0 elsewhere -- sparse, mirrors PIE's own DE calls
  lfc_soft  p_de * lfc_pred on every gene -- dense, uncertain genes shrunk
  delta     PIE's pseudobulk delta (mean log1p CP10K shift) read as a fold through the context's control mean m:
            fold = expm1(m + delta) / expm1(m)  (exact if every cell sat at the mean; genes with m ~ 0 are left alone)
"""
import json

import numpy as np
import pandas as pd

MODES = ("lfc_de", "lfc_soft", "delta")


def load_pie(parquet_paths, dataset):
    """{(context, perturbation): (p_de, lfc, delta)} and that dataset's gene axis, over one or more `pie infer`
    parquet files (pie_infer.py writes one per context)."""
    import pyarrow.parquet as pq

    paths = [parquet_paths] if isinstance(parquet_paths, (str, bytes)) or hasattr(parquet_paths, "suffix")         else list(parquet_paths)
    out, genes = {}, None
    for path in paths:
        tbl = pq.read_table(path)
        file_genes = json.loads(tbl.schema.metadata[b"pie"])["genes"].get(dataset)
        if file_genes is None:
            continue
        if genes is not None and list(file_genes) != genes:
            raise ValueError(f"{path}: {dataset} gene axis differs between files")
        genes = list(file_genes)
        df = tbl.to_pandas()
        df = df[df["dataset"] == dataset]
        out.update({(r.context, r.perturbation): (np.asarray(r.p_de, np.float64), np.asarray(r.lfc_pred, np.float64),
                                                  np.asarray(r.delta_p_pred, np.float64)) for r in df.itertuples()})
    if genes is None:
        raise ValueError(f"no PIE predictions for dataset {dataset!r} in {paths}")
    return out, genes


def to_log2fold(p_de, lfc, delta, mode, ctrl_log1p_mean=None, thr=0.5):
    if mode == "lfc_de":
        return np.where(p_de > thr, lfc, 0.0)
    if mode == "lfc_soft":
        return p_de * lfc
    if mode == "delta":
        m = np.asarray(ctrl_log1p_mean, np.float64)
        base = np.expm1(m)
        ok = base > 1e-3
        out = np.zeros_like(delta)
        out[ok] = np.log2(np.maximum(np.expm1(np.maximum(m[ok] + delta[ok], 0.0)), 1e-6) / base[ok])
        return out
    raise ValueError(f"mode must be one of {MODES}")


def align(vec, src_genes, dst_genes):
    """Vector on src_genes -> dst_genes order; genes PIE did not predict get 0 (no change)."""
    pos = {g: i for i, g in enumerate(src_genes)}
    out = np.zeros(len(dst_genes))
    idx = [(j, pos[g]) for j, g in enumerate(dst_genes) if g in pos]
    if idx:
        d, s = zip(*idx)
        out[list(d)] = vec[list(s)]
    return out


def profiles(pie, src_genes, dst_genes, context, perts, mode, ctrl_log1p_mean_dst=None, thr=0.5):
    """(len(perts), len(dst_genes)) log2 folds for one context; a pert PIE did not predict gets all zeros."""
    m_src = None if ctrl_log1p_mean_dst is None else align_back(ctrl_log1p_mean_dst, dst_genes, src_genes)
    rows = []
    for p in perts:
        if (context, p) not in pie:
            rows.append(np.zeros(len(dst_genes)))
            continue
        p_de, lfc, delta = pie[(context, p)]
        rows.append(align(to_log2fold(p_de, lfc, delta, mode, m_src, thr), src_genes, dst_genes))
    return np.stack(rows)


def align_back(vec_dst, dst_genes, src_genes):
    return pd.Series(vec_dst, index=dst_genes).reindex(src_genes).fillna(0.0).to_numpy()
