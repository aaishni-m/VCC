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

MODEL = "__MODEL__"   # replaced by the workflow (b2 | b4 | b5)
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
if MODEL in ("b4", "b5"):
    run([sys.executable, f"models/gene_transfer_{MODEL}.py"], cwd=code_dir, env=env)   # b5 finds external_profiles.npz under /kaggle/input
run([sys.executable, "models/generate_b2_submission.py", "--model", MODEL], cwd=code_dir, env=env)

run(["uv", "tool", "install", "vcc-cli"], env=env)
vcc = str(Path.home() / ".local" / "bin" / "vcc")
prep = [vcc, "prep", f"/kaggle/temp/outputs/{MODEL}_submission.h5ad",
        "-g", str(controls / "gene_names.csv"), "--perts", str(controls / "pert_counts.csv"),
        "-o", f"/kaggle/working/{MODEL}_submission.vcc"]
run(prep + ["--dry-run"], env=env)
if PREP == "full":
    run(prep, env=env)
print("done")
