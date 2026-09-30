"""
Kaggle job: build the small inputs for the semi-supervised context-transfer model (models/ssl_transfer.py).

For K562 and RPE1 (Replogle 2022, figshare, CC BY 4.0) it downloads the raw single-cell files (~10.7 / 8.7 GB),
keeps a random sample of NON-TARGETING control cells (the unlabelled context description), and the
essential-screen normalized bulk effect tables (the labels). Everything big stays on Kaggle's scratch disk;
only the small extracts are written to /kaggle/working:

    controls_K562.h5ad, controls_RPE1.h5ad    N_CTRL control cells x genes, raw counts
    effects_K562.h5ad,  effects_RPE1.h5ad     perturbations x genes, Replogle normalized-bulk z (NaN/inf -> 0)
"""
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path

pins = [f"{p}=={version(p)}" for p in ("numpy", "scipy", "pandas")]
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "anndata", "requests", *pins], check=True)

import anndata as ad  # noqa: E402
import h5py  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import requests  # noqa: E402
import scipy.sparse as sp  # noqa: E402

N_CTRL = 6000
FILES = {  # figshare file ids (article 20029387)
    "K562": {"single": 35773219, "bulk": 35780870},   # K562_essential_raw_singlecell_01 / K562_essential_normalized_bulk_01
    "RPE1": {"single": 35775606, "bulk": 35775512},   # rpe1_raw_singlecell_01 / rpe1_normalized_bulk_01
}
TMP, OUT = Path("/kaggle/temp/ssl"), Path("/kaggle/working")
TMP.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(0)


def fetch(fid, dest):
    if dest.exists():
        return dest
    with requests.get(f"https://ndownloader.figshare.com/files/{fid}", stream=True, timeout=900) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(1 << 24):
                f.write(chunk)
    with open(dest, "rb") as f:
        if f.read(4)[1:] != b"HDF":
            sys.exit(f"{dest.name} is not an HDF5 file")
    print(f"downloaded {dest.name}: {dest.stat().st_size / 1e9:.2f} GB", flush=True)
    return dest


def gene_col(df, candidates):
    return next(c for c in candidates if c in df.columns)


for line, ids in FILES.items():
    # labels: essential-screen effect table
    bulk = ad.read_h5ad(fetch(ids["bulk"], TMP / f"{line}_bulk.h5ad"))
    Xb = np.asarray(bulk.X.todense() if sp.issparse(bulk.X) else bulk.X, dtype=np.float32)
    Xb[~np.isfinite(Xb)] = 0
    tgt = (bulk.obs[gene_col(bulk.obs, ["gene", "gene_name"])].astype(str) if
           any(c in bulk.obs for c in ["gene", "gene_name"]) else pd.Series([s.split("_")[1] for s in bulk.obs_names]))
    var_names = bulk.var[gene_col(bulk.var, ["gene_name"])].astype(str).to_numpy()
    eff = ad.AnnData(X=Xb, obs=pd.DataFrame({"target": tgt.to_numpy()}, index=bulk.obs_names.astype(str)),
                     var=pd.DataFrame(index=pd.Index(var_names).astype(str)))
    eff = eff[~eff.obs["target"].str.contains("non-targeting")].copy()
    eff.var_names_make_unique()
    eff.write_h5ad(OUT / f"effects_{line}.h5ad")
    print(f"{line}: effects {eff.shape}", flush=True)

    # unlabelled context: control cells from the raw single-cell file (read only the chosen rows)
    path = fetch(ids["single"], TMP / f"{line}_single.h5ad")
    a = ad.read_h5ad(path, backed="r")
    col = gene_col(a.obs, ["gene", "gene_name", "target"])
    ctrl = np.where(a.obs[col].astype(str).str.contains("non-targeting").to_numpy())[0]
    pick = np.sort(rng.choice(ctrl, size=min(N_CTRL, len(ctrl)), replace=False))
    sc_var = a.var[gene_col(a.var, ["gene_name"])].astype(str).to_numpy() if "gene_name" in a.var else a.var_names.astype(str).to_numpy()
    print(f"{line}: {a.shape[0]} cells, {len(ctrl)} controls, X is {type(a.X).__name__}", flush=True)
    a.file.close()
    with h5py.File(path, "r") as f:
        x = f["X"]
        if isinstance(x, h5py.Dataset):                      # dense matrix: read the chosen rows in blocks
            rows = np.concatenate([x[pick[i:i + 500]] for i in range(0, len(pick), 500)]).astype(np.float32)
        else:                                                # CSR group
            indptr = x["indptr"][:]
            rows = np.zeros((len(pick), len(sc_var)), np.float32)
            for k, r in enumerate(pick):
                s, e = indptr[r], indptr[r + 1]
                rows[k, x["indices"][s:e]] = x["data"][s:e]
    c = ad.AnnData(X=sp.csr_matrix(rows), var=pd.DataFrame(index=pd.Index(sc_var).astype(str)))
    c.var_names_make_unique()
    c.write_h5ad(OUT / f"controls_{line}.h5ad")
    print(f"{line}: controls {c.shape}, median UMI/cell {np.median(rows.sum(1)):.0f}", flush=True)
    path.unlink()                                            # free scratch space before the next cell line
print("done", flush=True)
