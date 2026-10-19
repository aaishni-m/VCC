"""
Kaggle GPU job: PIE (Verma et al. 2026, "PIE: Generalizing perturbation effects across unseen perturbations,
contexts and datasets", bioRxiv 10.64898/2026.10.02.756297; Arc Institute, model under the Arc PIE Model
Non-Commercial License -- used here for non-commercial VCC participation) zero-shot on

  * vcc26.A / vcc26.B / vcc26.C -- the challenge contexts (controls only), our 300 targets
  * replogle.rpe1               -- the local benchmark's 300-knockdown RPE1 panel (Replogle is held out of the
                                   cross-dataset checkpoint's training, so this is an honest check)

with the released cross-dataset checkpoint (arcinstitute/PIE_replogle_xdataset, trained on Tahoe-100M, Jiang et al.,
X-Atlas/Orion and Arc VCC 2025). Everything follows the released code; two practical changes, both inputs only:
  * the 43 GB per-token ncbi_text source is cut down to the perturbations we query (same rows, same bytes);
  * contexts A/B/C are anonymous, so they have no Context Description (the source is simply absent for them, which
    PIE supports); genes without a gene_text embedding (271 of 18,533) are left off vcc26's gene axis.

Output in /kaggle/working: pie_predictions.parquet (p_de, lfc_pred, delta_p_pred per row x gene, PIE's own format)
plus pie_query.json and the vcc26 gene list.
"""
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

T0 = time.time()
TMP, OUT = Path("/kaggle/temp"), Path("/kaggle/working")
TMP.mkdir(parents=True, exist_ok=True)


def log(*a):
    print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)


def run(cmd, **kw):
    log("+", " ".join(map(str, cmd))[:300])
    subprocess.run(cmd, check=True, **kw)


log(subprocess.run(["df", "-h", "/kaggle/temp", "/kaggle/working"], capture_output=True, text=True).stdout)
log(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv"], capture_output=True,
                   text=True).stdout)

# ---- PIE needs Python 3.12: its own venv ----
run([sys.executable, "-m", "pip", "install", "-q", "uv"])
VENV = TMP / "pie-venv"
run(["uv", "venv", str(VENV), "--python", "3.12", "-q"])
PY = str(VENV / "bin" / "python")
run(["uv", "pip", "install", "--python", PY, "-q", "arc-pie", "requests", "scanpy"])

controls = next(Path("/kaggle/input").rglob("pert_counts.csv")).parent
panel_meta = next(Path("/kaggle/input").rglob("localbench_meta.json"))
inner = TMP / "pie_run.py"
inner.write_text(r'''
import json, os, sys, time, logging
from pathlib import Path
import numpy as np, pandas as pd, requests
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
T0 = time.time()
def log(*a): print(f"[pie {time.time()-T0:6.0f}s]", *a, flush=True)

TMP, OUT = Path("/kaggle/temp"), Path("/kaggle/working")
CONTROLS, PANEL_META = Path(sys.argv[1]), Path(sys.argv[2])
for k, v in {"PIE_DATA_ROOT": TMP / "pie_data", "PIE_RUNS_ROOT": TMP / "pie_runs", "PIE_CACHE_DIR": TMP / "pie_cache",
             "WANDB_MODE": "disabled"}.items():
    os.environ[k] = str(v); Path(v).mkdir(parents=True, exist_ok=True)

from huggingface_hub import snapshot_download, hf_hub_download, hf_hub_url
SRC_REV = "fb624a5158a7f37dfa13a015cdacadb985549d26"           # PIE_sources commit pinned by the checkpoint

run_dir = TMP / "pie_runs" / "replogle_xdataset"
snapshot_download("arcinstitute/PIE_replogle_xdataset", local_dir=run_dir, allow_patterns=["*.ckpt", "*.yaml", "*.json", "*.md"])
log("checkpoint", sorted(p.name for p in run_dir.iterdir()))

from pie.predict import load_checkpoint, predict_loaded, write_predictions_parquet, RowSource
loaded = load_checkpoint(run_dir / "best_auprc.ckpt")
data = loaded.config.data
log("checkpoint sources:", {k: str(v)[:90] for k, v in data.source_dirs.items()})

# ---- query perturbations ----
targets = pd.read_csv(CONTROLS / "pert_counts.csv")["target_gene"].str.strip().tolist()
panel = json.load(open(PANEL_META))["panel"]
wanted = sorted(set(targets) | set(panel))

# ---- knowledge sources: small ones whole, ncbi_text cut to our perturbations ----
src_root = TMP / "pie_sources"
small = [n for n in data.source_dirs if n != "ncbi_text"] + ["gene_text"]
snapshot_download("arcinstitute/PIE_sources", repo_type="dataset", revision=SRC_REV, local_dir=src_root,
                  allow_patterns=[f"{n}/*" for n in small])
nd = src_root / "ncbi_text"; nd.mkdir(parents=True, exist_ok=True)
for f in ("meta.json", "offsets.npy", "descriptions.json", "aliases.yaml"):
    try:
        hf_hub_download("arcinstitute/PIE_sources", f"ncbi_text/{f}", repo_type="dataset", revision=SRC_REV,
                        local_dir=TMP / "ncbi_full")
    except Exception as e:
        log("ncbi_text", f, "not downloaded:", e)
full = TMP / "ncbi_full" / "ncbi_text"
meta = json.load(open(full / "meta.json")); offsets = np.load(full / "offsets.npy")
aliases = {}
if (full / "aliases.yaml").exists():
    import yaml; aliases = yaml.safe_load(open(full / "aliases.yaml")) or {}
key_row = {k: i for i, k in enumerate(meta["keys"])}
keep = [k for k in meta["keys"] if k in set(wanted) or k in {aliases.get(w) for w in wanted}]
log(f"ncbi_text: {len(meta['keys'])} keys, keeping {len(keep)} for {len(wanted)} queried perturbations")
url = hf_hub_url("arcinstitute/PIE_sources", "ncbi_text/embeddings.npy", repo_type="dataset", revision=SRC_REV)
sess = requests.Session()
hdr = sess.get(url, headers={"Range": "bytes=0-4095"}, timeout=120).content
import ast
hlen = int.from_bytes(hdr[8:10], "little") if hdr[6] == 1 else int.from_bytes(hdr[8:12], "little")
start = (10 if hdr[6] == 1 else 12) + hlen
info = ast.literal_eval(hdr[(10 if hdr[6] == 1 else 12):start].decode("latin1").strip())
dt = np.dtype(info["descr"]); D = info["shape"][1]; row_bytes = D * dt.itemsize
chunks, new_off = [], [0]
for k in keep:
    i = key_row[k]; a, b = int(offsets[i]), int(offsets[i + 1])
    lo, hi = start + a * row_bytes, start + b * row_bytes - 1
    for attempt in range(5):
        try:
            r = sess.get(url, headers={"Range": f"bytes={lo}-{hi}"}, timeout=600); r.raise_for_status()
            buf = r.content; assert len(buf) == hi - lo + 1; break
        except Exception as e:
            log("retry", k, e); time.sleep(5)
    chunks.append(np.frombuffer(buf, dtype=dt).reshape(b - a, D)); new_off.append(new_off[-1] + (b - a))
np.save(nd / "embeddings.npy", np.concatenate(chunks)); np.save(nd / "offsets.npy", np.array(new_off, dtype=np.int64))
sub = dict(meta); sub["keys"] = keep
sub["provenance"] = dict(meta.get("provenance", {}), subset_of=f"PIE_sources@{SRC_REV}/ncbi_text")
json.dump(sub, open(nd / "meta.json", "w"))
if (full / "descriptions.json").exists():
    d = json.load(open(full / "descriptions.json")); json.dump({k: d[k] for k in keep if k in d}, open(nd / "descriptions.json", "w"))
if (full / "aliases.yaml").exists():
    import shutil; shutil.copy(full / "aliases.yaml", nd / "aliases.yaml")
log("ncbi_text subset written:", sum(c.shape[0] for c in chunks), "tokens")

local = {n: str(src_root / n) for n in data.source_dirs}
loaded.config = loaded.config.model_copy(update={"data": data.model_copy(
    update={"source_dirs": local, "gene_text_dir": str(src_root / "gene_text")})})

# ---- vcc26 controls-only dir: log1p(CP10K) control means of A/B/C on the genes gene_text covers ----
import anndata as ad, scipy.sparse as sp
gt = set(json.load(open(src_root / "gene_text" / "meta.json"))["keys"])
genes = pd.read_csv(CONTROLS / "gene_names.csv")["gene_name"].astype(str).tolist()
axis = [g for g in genes if g in gt]
(TMP / "vcc26_genes.txt").write_text("\n".join(axis) + "\n"); (OUT / "vcc26_genes.txt").write_text("\n".join(axis) + "\n")
log(f"vcc26 gene axis: {len(axis)} of {len(genes)} (gene_text coverage)")
h5dir = TMP / "vcc26_h5ad"; h5dir.mkdir(exist_ok=True)
for c in ("A", "B", "C"):
    a = ad.read_h5ad(CONTROLS / f"context_{c}.h5ad")
    X = a.X.tocsr().astype(np.float32)
    lib = np.asarray(X.sum(1)).ravel(); X = sp.diags(1e4 / np.maximum(lib, 1)) @ X; X.data = np.log1p(X.data)
    b = ad.AnnData(X=X.tocsr(), obs=pd.DataFrame({"context": c, "target_gene": "non-targeting"}, index=a.obs_names),
                   var=pd.DataFrame(index=a.var_names))
    b.write_h5ad(h5dir / f"vcc26_{c}.h5ad"); log("normalized controls", c, b.shape)
import subprocess
vdir = TMP / "pie_data" / "vcc26"
subprocess.run([str(Path(sys.executable).parent / "pie"), "prep", "dataset=arc_vcc_25", "name=vcc26",
                f"h5ad={h5dir}/*.h5ad", f"output_dir={vdir}", "controls_only=true",
                f"genes={TMP / 'vcc26_genes.txt'}", "overwrite=true"], check=True)

# ---- query: A/B/C x 300 targets, and the RPE1 panel in Replogle's own dir ----
rep = [p for p in data.preprocessed_dirs if "replogle" in str(p)][0]
query = {f"vcc26.{c}": targets for c in ("A", "B", "C")}
query["replogle.rpe1"] = panel
qp = TMP / "pie_query.json"; json.dump(query, open(qp, "w")); json.dump(query, open(OUT / "pie_query.json", "w"))
from pie.assets import resolve_asset
dirs = [resolve_asset(rep, kind="preprocessed"), vdir]
log("predicting", {k: len(v) for k, v in query.items()})
preds = predict_loaded(loaded, RowSource("query", qp), dirs, "cuda", 8)
out = write_predictions_parquet(preds, OUT / "pie_predictions.parquet")
log("wrote", out, [(b.dataset, len(b.contexts), len(b.genes)) for b in preds.blocks])
''')
run([PY, str(inner), str(controls), str(panel_meta)])
log("done")
