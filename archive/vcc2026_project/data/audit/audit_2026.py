import anndata as ad
import numpy as np
import scipy.sparse as sp
import json, gc

paths = {
    "A": "/mnt/user-data/uploads/controls/context_A.h5ad",
    "B": "/mnt/user-data/uploads/controls/context_B.h5ad",
    "C": "/mnt/user-data/uploads/controls/context_C.h5ad",
}

summary = {}

for ctx, p in paths.items():
    print(f"\n{'='*60}\nCONTEXT {ctx}\n{'='*60}")
    a = ad.read_h5ad(p)
    print("shape:", a.shape)
    print("X type:", type(a.X), "dtype:", a.X.dtype)
    is_sparse = sp.issparse(a.X)
    print("is sparse:", is_sparse)
    if is_sparse:
        nnz = a.X.nnz
        total = a.shape[0]*a.shape[1]
        print(f"nnz: {nnz} / {total} = {nnz/total:.4f} density ({100*(1-nnz/total):.2f}% sparse)")
        Xmax = a.X.max()
        Xmin = a.X.min()
        print("X min/max:", Xmin, Xmax)
        # check integer-ness
        data_sample = a.X.data[:1000]
        is_int = np.allclose(data_sample, np.round(data_sample))
        print("values look integral (raw counts?):", is_int)
    print("\nobs columns:", list(a.obs.columns))
    print(a.obs.head(3).to_string())
    print("\nvar columns:", list(a.var.columns))
    print(a.var.head(3).to_string())
    print("\nobsm keys:", list(a.obsm.keys()))
    print("obsp keys:", list(a.obsp.keys()))
    print("layers keys:", list(a.layers.keys()))
    print("uns keys:", list(a.uns.keys()))

    # target_gene distribution
    if "target_gene" in a.obs.columns:
        vc = a.obs["target_gene"].value_counts()
        print("\ntarget_gene unique values:", vc.index.tolist()[:5], "n_unique=", len(vc))
        print(vc.head())
    # ntc_id distribution
    ntc_col = None
    for c in a.obs.columns:
        if "ntc" in c.lower():
            ntc_col = c
    if ntc_col:
        vc2 = a.obs[ntc_col].value_counts()
        print(f"\n{ntc_col} distribution: n_unique={len(vc2)}")
        print(vc2.describe())

    # counts per cell (total UMI) and genes detected per cell
    counts_per_cell = np.asarray(a.X.sum(axis=1)).flatten()
    genes_per_cell = np.asarray((a.X > 0).sum(axis=1)).flatten()
    print("\ncounts_per_cell: mean=%.1f median=%.1f min=%.1f max=%.1f" % (
        counts_per_cell.mean(), np.median(counts_per_cell), counts_per_cell.min(), counts_per_cell.max()))
    print("genes_per_cell: mean=%.1f median=%.1f min=%.1f max=%.1f" % (
        genes_per_cell.mean(), np.median(genes_per_cell), genes_per_cell.min(), genes_per_cell.max()))

    # gene detection frequency (fraction of cells expressing each gene)
    cells_per_gene = np.asarray((a.X > 0).sum(axis=0)).flatten()
    gene_det_freq = cells_per_gene / a.shape[0]
    print("gene detection freq: mean=%.4f median=%.4f frac_genes_never_detected=%.4f" % (
        gene_det_freq.mean(), np.median(gene_det_freq), (cells_per_gene==0).mean()))

    # mean expression per gene (for later CPM filter check)
    mean_counts_per_cell_total = counts_per_cell.mean()

    summary[ctx] = {
        "shape": list(a.shape),
        "nnz": int(nnz) if is_sparse else None,
        "density": float(nnz/total) if is_sparse else None,
        "X_dtype": str(a.X.dtype),
        "X_min": float(Xmin), "X_max": float(Xmax),
        "obs_columns": list(a.obs.columns),
        "var_columns": list(a.var.columns),
        "counts_per_cell_mean": float(counts_per_cell.mean()),
        "counts_per_cell_median": float(np.median(counts_per_cell)),
        "counts_per_cell_min": float(counts_per_cell.min()),
        "counts_per_cell_max": float(counts_per_cell.max()),
        "genes_per_cell_mean": float(genes_per_cell.mean()),
        "genes_per_cell_median": float(np.median(genes_per_cell)),
        "n_genes_never_detected": int((cells_per_gene==0).sum()),
        "var_names_first5": list(a.var_names[:5]),
        "var_names_match_gene_names_csv": None,  # filled later
    }

    del a
    gc.collect()

print("\n\nSUMMARY JSON:")
print(json.dumps(summary, indent=2))
with open("/tmp/claude-0/-home-claude/40ffa64b-8854-5c6c-8471-0e27e926d30d/scratchpad/audit_2026_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
