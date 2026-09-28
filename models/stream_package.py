"""
Memory-light stand-in for `vcc prep`'s packaging step, for predictions too dense for prep on a 30 GB machine.

Why: B6's sparse T4 transform keeps every real control count, so the file has 2.09e9 stored entries (B5: 1.78e9,
whose multinomial redraw dropped many 1-counts). `vcc prep` holds the whole matrix in RAM (~22 GiB by its own
estimate) and was SIGKILLed on Kaggle's 30 GiB CPU box twice.

What it writes is exactly what vcc-cli 0.2.2's prep writes (vcc/prep.py run_prep + _write_vcc):
  tar( meta.json {schema, nnz, n_obs, n_vars, cli_version}, pred.h5ad.zst )
  pred.h5ad = AnnData(X=<float32 CSR>, obs={target_gene, context} indexed "0".."n-1", var=index only)
but X is copied into the file in chunks instead of being loaded. The same validations prep runs on a counts
submission are applied, streamed: gene order vs gene_names.csv, per-context target list and 400 cells each,
nnz cap, finite / non-negative / whole-number values, per-cell total <= 1e6.

`--selftest` packages a small subset both ways (real `vcc prep` and this) and requires the two pred.h5ad files to
be identical in structure, attributes and values before anything else runs.

Usage:
    python models/stream_package.py SRC.h5ad OUT.vcc --genes gene_names.csv --perts pert_counts.csv [--selftest]
"""
import argparse
import json
import os
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp
import zstandard as zstd
from vcc import __version__ as VCC_VERSION
from vcc import sizing
from vcc.prep import MAX_COUNTS_PER_CELL, read_gene_list
from vcc.vccfile import META_MEMBER, META_SCHEMA, PRED_MEMBER

PERT_COL, CONTEXT_COL, NTC = "target_gene", "context", "non-targeting"
CHUNK_ROWS = 6000


def write_minimal(src: Path, dst: Path) -> int:
    """pred.h5ad exactly as prep builds it, streaming X. Returns nnz."""
    a = ad.read_h5ad(src, backed="r")
    n_obs, n_vars = a.shape
    obs = pd.DataFrame({PERT_COL: a.obs[PERT_COL].to_numpy()}, index=np.arange(n_obs).astype(str))
    obs[CONTEXT_COL] = a.obs[CONTEXT_COL].astype(str).to_numpy()
    var = pd.DataFrame(index=a.var.index.values)
    a.file.close()
    ad.AnnData(X=sp.csr_matrix((n_obs, n_vars), dtype=np.float32), obs=obs, var=var).write_h5ad(dst)

    with h5py.File(src, "r") as fs, h5py.File(dst, "r+") as fd:
        sx, dx = fs["X"], fd["X"]
        nnz = int(sx["indptr"][-1])
        idx_dtype = np.int32 if nnz < 2**31 else np.int64   # what scipy (and so prep) holds after reading
        for name, dtype in (("data", np.float32), ("indices", idx_dtype), ("indptr", idx_dtype)):
            del dx[name]
            n = n_obs + 1 if name == "indptr" else nnz
            out = dx.create_dataset(name, shape=(n,), dtype=dtype)
            step = 200_000_000
            for s in range(0, n, step):
                out[s:s + step] = sx[name][s:s + step].astype(dtype, copy=False)
    return nnz


def validate(h5: Path, genes_csv: str, perts_csv: str, full_panel: bool = True) -> None:
    """prep's counts-submission checks, streamed. Raises SystemExit on the first failure."""
    a = ad.read_h5ad(h5, backed="r")
    if a.var_names.tolist() != read_gene_list(genes_csv):
        sys.exit("gene axis differs from gene_names.csv (order or content)")
    labels, contexts = a.obs[PERT_COL].astype(str).to_numpy(), a.obs[CONTEXT_COL].astype(str).to_numpy()
    a.file.close()
    if (labels == NTC).any():
        sys.exit("submission contains control cells")
    if full_panel:
        expected = set(pd.read_csv(perts_csv)[PERT_COL].astype(str))
        for c in ("A", "B", "C"):
            counts = pd.Series(labels[contexts == c]).value_counts()
            if set(counts.index) != expected or (counts != 400).any():
                sys.exit(f"context {c}: target list or 400-cells-per-perturbation check failed")
    with h5py.File(h5, "r") as f:
        x = f["X"]
        indptr = x["indptr"][:].astype(np.int64)
        nnz = int(indptr[-1])
        if nnz > sizing.HARD_CAP_NNZ:
            sys.exit(f"nnz {nnz} above the {sizing.HARD_CAP_NNZ} cap")
        max_total = 0.0
        for r in range(0, len(indptr) - 1, CHUNK_ROWS):
            lo, hi = indptr[r], indptr[min(r + CHUNK_ROWS, len(indptr) - 1)]
            d = x["data"][lo:hi]
            if not np.isfinite(d).all() or (d < 0).any() or (d != np.floor(d)).any():
                sys.exit(f"non-finite, negative or fractional values in rows {r}..")
            rows = indptr[r:r + CHUNK_ROWS + 1] - lo
            sums = np.add.reduceat(d.astype(np.float64), rows[:-1]) if len(d) else np.zeros(len(rows) - 1)
            sums[rows[:-1] == rows[1:]] = 0     # reduceat quirk on empty rows
            max_total = max(max_total, float(sums.max(initial=0)))
        if max_total > MAX_COUNTS_PER_CELL:
            sys.exit(f"a cell totals {max_total:.0f} counts, above {MAX_COUNTS_PER_CELL}")
    print(f"[package] validated: nnz={nnz:,}, max cell total={max_total:.0f}", flush=True)


def write_vcc(pred_h5ad: Path, out: Path, nnz: int, n_obs: int, n_vars: int) -> None:
    with tempfile.TemporaryDirectory(dir=pred_h5ad.parent) as tmp:
        zst, meta = Path(tmp) / PRED_MEMBER, Path(tmp) / META_MEMBER
        meta.write_text(json.dumps({"schema": META_SCHEMA, "nnz": nnz, "n_obs": n_obs, "n_vars": n_vars,
                                    "cli_version": VCC_VERSION}, sort_keys=True), encoding="utf-8")
        cctx = zstd.ZstdCompressor(level=3, threads=min(os.cpu_count() or 1, 8))
        with open(pred_h5ad, "rb") as src, open(zst, "wb") as dst:
            cctx.copy_stream(src, dst)

        def normalize(ti):
            ti.uid = ti.gid = 0
            ti.uname = ti.gname = ""
            ti.mtime = 0
            return ti

        with tarfile.open(out, "w") as tar:
            tar.add(meta, arcname=META_MEMBER, filter=normalize)
            tar.add(zst, arcname=PRED_MEMBER, filter=normalize)


def package(src: Path, out: Path, genes: str, perts: str, full_panel: bool = True) -> None:
    pred = out.with_suffix(".pred.h5ad")
    nnz = write_minimal(src, pred)
    validate(pred, genes, perts, full_panel)
    with h5py.File(pred, "r") as f:
        n_obs, n_vars = (int(v) for v in f["X"].attrs["shape"])
    write_vcc(pred, out, nnz, n_obs, n_vars)
    pred.unlink()
    print(f"[package] wrote {out} ({out.stat().st_size / 1e9:.2f} GB)", flush=True)


def _extract_pred(vcc_path: Path, dst: Path) -> None:
    with tarfile.open(vcc_path) as tar:
        blob = tar.extractfile(PRED_MEMBER)
        with open(dst, "wb") as fh:
            zstd.ZstdDecompressor().copy_stream(blob, fh)


def _h5_items(path: Path) -> dict:
    items = {}
    with h5py.File(path, "r") as f:
        def visit(name, obj):
            attrs = {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in obj.attrs.items()}
            if isinstance(obj, h5py.Dataset):
                v = obj[()]
                # object (variable-length string) arrays: compare the values, not their pointer bytes
                content = tuple(np.ravel(v).tolist()) if getattr(v, "dtype", None) == object else np.asarray(v).tobytes()
                items[name] = (str(obj.dtype), obj.shape, attrs, content)
            else:
                items[name] = ("group", None, attrs, None)
        f.visititems(visit)
        items["/"] = ("root", None, {k: str(v) for k, v in f.attrs.items()}, None)
    return items


def selftest(src: Path, genes: str, perts: str, workdir: Path) -> None:
    """Package a 2-perturbation-per-context subset with real `vcc prep` and with this module; require identity."""
    a = ad.read_h5ad(src, backed="r")
    lab, ctx = a.obs[PERT_COL].astype(str).to_numpy(), a.obs[CONTEXT_COL].astype(str).to_numpy()
    rows = np.concatenate([np.where((ctx == c) & np.isin(lab, np.unique(lab[ctx == c])[:2]))[0] for c in ("A", "B", "C")])
    sub = workdir / "selftest_src.h5ad"
    a[rows].to_memory().write_h5ad(sub)
    a.file.close()
    ref = workdir / "selftest_prep.vcc"
    subprocess.run([sys.executable, "-m", "vcc", "prep", str(sub), "-g", genes, "--perts", perts, "--no-verify-targets",
                    "--no-check-cell-counts", "-o", str(ref), "-f"], check=True)
    mine = workdir / "selftest_stream.vcc"
    package(sub, mine, genes, perts, full_panel=False)
    for p in (ref, mine):
        _extract_pred(p, p.with_suffix(".h5ad"))
    A, B = _h5_items(ref.with_suffix(".h5ad")), _h5_items(mine.with_suffix(".h5ad"))
    diffs = [k for k in sorted(set(A) | set(B)) if A.get(k) != B.get(k)]
    with tarfile.open(ref) as t1, tarfile.open(mine) as t2:
        m1 = json.loads(t1.extractfile(META_MEMBER).read())
        m2 = json.loads(t2.extractfile(META_MEMBER).read())
        names = (t1.getnames(), t2.getnames())
    if diffs or m1 != m2 or names[0] != names[1]:
        for k in diffs:
            print(f"  DIFF {k}: prep={A.get(k)[:3] if A.get(k) else None} stream={B.get(k)[:3] if B.get(k) else None}")
        print(f"  meta prep={m1} stream={m2}; members {names}")
        sys.exit("[selftest] streamed package differs from vcc prep -- not packaging the real submission")
    print(f"[selftest] identical to vcc prep: {len(A)} HDF5 objects, meta {m1}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("src", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--genes", required=True)
    p.add_argument("--perts", required=True)
    p.add_argument("--selftest", action="store_true")
    args = p.parse_args()
    if args.selftest:
        selftest(args.src, args.genes, args.perts, args.src.parent)
    package(args.src, args.out, args.genes, args.perts)
