"""
Kaggle-side runner, pushed by .github/workflows/kaggle-run.yml. Runs on Kaggle's free CPU box
(~30 GB RAM), not on GitHub or the laptop.

Inputs (attached as Kaggle datasets by the workflow):
  <user>/vcc-code  -- this repo, uploaded fresh by the workflow on every run
  <user>/vcc-data  -- controls/ bundle + vcc-perturb.csv (uploaded once, by hand)
Output: /kaggle/working/<model>_submission.vcc (+ run log). Submitting stays a manual step.
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

MODEL = "__MODEL__"   # replaced by the workflow (b2 | b4 | b5 | b6)
PREP = "__PREP__"     # "dry-run" or "full"


def find_one(name):
    hits = list(Path("/kaggle/input").rglob(name))
    if len(hits) != 1:
        sys.exit(f"expected exactly one {name} under /kaggle/input, found {hits}")
    return hits[0]


def run(cmd, **kw):
    print("+", " ".join(map(str, cmd)), flush=True)
    subprocess.run(cmd, check=True, **kw)


code_dir = Path("/kaggle/temp/vcc")
shutil.copytree(find_one("config.py").parent, code_dir, dirs_exist_ok=True)
# Kaggle may unpack .npz (zip) files into folders; drop those so load_or_build rebuilds the cache.
for p in (code_dir / "data" / "audit" / "pseudobulk_cache").glob("*.npz"):
    if p.is_dir():
        print(f"warning: {p.name} was unpacked by Kaggle; rebuilding it", flush=True)
        shutil.rmtree(p)
controls = find_one("context_A.h5ad").parent

env = dict(os.environ,
           VCC_CONTROLS_DIR=str(controls),
           VCC_PERTURB_CSV=str(find_one("vcc-perturb.csv")),
           VCC_OUTPUTS_DIR="/kaggle/temp/outputs")   # 14 GB .h5ad stays off the 20 GB working dir

run([sys.executable, "-m", "pip", "install", "-q", "anndata", "uv"])
for step in {"b4": ["b4"], "b5": ["b5"], "b6": ["b5", "b6"]}.get(MODEL, []):
    run([sys.executable, f"models/gene_transfer_{step}.py"], cwd=code_dir, env=env)   # b5 finds external_profiles.npz under /kaggle/input
run([sys.executable, "models/generate_b2_submission.py", "--model", MODEL], cwd=code_dir, env=env)

VCC_CLI = "vcc-cli==0.2.2"   # stream_package.py mirrors this version's prep; bump both together
run(["uv", "tool", "install", VCC_CLI], env=env)
vcc = str(Path.home() / ".local" / "bin" / "vcc")
src = f"/kaggle/temp/outputs/{MODEL}_submission.h5ad"
genes, perts = str(controls / "gene_names.csv"), str(controls / "pert_counts.csv")
out = f"/kaggle/working/{MODEL}_submission.vcc"
if MODEL == "b6":
    # too dense for `vcc prep` on 30 GiB (see models/stream_package.py): package in a streaming way, after a
    # self-test that requires byte-level agreement with real `vcc prep` on a subset
    run([sys.executable, "-m", "pip", "install", "-q", VCC_CLI, "zstandard"])
    run([sys.executable, "models/stream_package.py", src, out, "--genes", genes, "--perts", perts, "--selftest"],
        cwd=code_dir, env=env)
else:
    prep = [vcc, "prep", src, "-g", genes, "--perts", perts, "-o", out]
    run(prep + ["--dry-run"], env=env)
    if PREP == "full":
        run(prep, env=env)
print("done")
