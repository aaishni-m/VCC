"""
Check gene-symbol overlap between B3's 287 "unknown" VCC target genes and Replogle et al. 2022's
K562-essential and RPE1-essential Perturb-seq screens (each ~2-2.5k genes -- much smaller, faster
download than the genome-wide K562 screen's ~9,866 genes, which this deliberately skips for now).

Only reads each h5ad's var (gene) axis via h5py -- never loads the X expression matrix -- so this
stays cheap on RAM regardless of the file's size on disk. Files download to data/replogle_check/
and are kept there (not re-downloaded on a rerun).

Download source: pertpy's own dataset registry (github.com/scverse/pertpy), which points at
Zenodo record 10044268 -- the official processed-data mirror for Replogle et al., Cell 2022.

Usage:
    pip install requests h5py   # h5py likely already installed; requests probably too
    python models/check_replogle_overlap.py
"""
import json
from pathlib import Path

import h5py
import requests

DOWNLOAD_DIR = Path("data/replogle_check")
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

URLS = {
    "k562_essential": "https://zenodo.org/record/10044268/files/ReplogleWeissman2022_K562_essential.h5ad?download=1",
    "rpe1_essential":  "https://zenodo.org/record/10044268/files/ReplogleWeissman2022_rpe1.h5ad?download=1",
}


def download(url, dest):
    if dest.exists():
        print(f"  already downloaded: {dest} ({dest.stat().st_size / 1e9:.2f} GB)")
        return
    print(f"  downloading {url}")
    print(f"    -> {dest}")
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        done = 0
        tmp = dest.with_suffix(".part")
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024 * 8):
                f.write(chunk)
                done += len(chunk)
                if total:
                    print(f"\r    {done / 1e9:.2f} / {total / 1e9:.2f} GB", end="", flush=True)
        tmp.rename(dest)
    print()


def read_gene_symbols(h5ad_path):
    """Read ONLY the var (gene) index from an h5ad file via h5py -- never touches X, so this is
    fast and memory-cheap regardless of how many cells the file actually contains."""
    with h5py.File(h5ad_path, "r") as f:
        var = f["var"]
        if "_index" in var:
            idx = var["_index"][:]
        else:
            idx_name = var.attrs.get("_index", "gene_name")
            idx = var[idx_name][:]
        genes = [g.decode() if isinstance(g, bytes) else str(g) for g in idx]
    return genes


# ── Our 287 "unknown" genes (from b3_meta.json's gene_source labels) ───────
with open("models/b3_meta.json") as f:
    meta = json.load(f)
our_287 = [g for g, s in zip(meta["vcc_targets"], meta["gene_source"]) if s != "real"]
print(f"Our unmeasured target genes: {len(our_287)}\n")

results = {}
for name, url in URLS.items():
    print(f"=== {name} ===")
    dest = DOWNLOAD_DIR / f"{name}.h5ad"
    download(url, dest)
    genes = read_gene_symbols(dest)
    gene_set = {g.strip().upper() for g in genes}
    overlap = [g for g in our_287 if g.strip().upper() in gene_set]
    results[name] = {
        "total_genes_in_screen": len(genes),
        "overlap_count": len(overlap),
        "overlap_genes": overlap,
    }
    print(f"  total genes in screen: {len(genes)}")
    print(f"  overlap with our 287:  {len(overlap)}")
    print()

combined = sorted(set().union(*(set(r["overlap_genes"]) for r in results.values())))
print(f"Combined unique overlap (K562-essential U RPE1-essential): {len(combined)} / {len(our_287)}")
print(combined)

with open("models/replogle_overlap.json", "w") as f:
    json.dump(
        {"our_287_count": len(our_287), "per_dataset": results, "combined_overlap": combined},
        f, indent=2,
    )
print("\nSaved: models/replogle_overlap.json")
