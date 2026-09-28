"""
B6 -- B5's profiles restricted to each perturbation's top-K genes, generated with the sparse T4 transform.

Diagnosis behind it (predicted-side DE simulation on context A, official DE procedure):
  - B5's count generator (T2 = full multinomial redraw) makes the official Wilcoxon test call ~5,600 genes per
    perturbation, and ~7,200 even with zero predicted effect; untouched resampled controls give 0. The
    submission's significant sets were therefore almost all artifacts, which is why direction fidelity and
    significance overlap scored at baseline (-0.003, -0.010).
  - On vcc-perturb (H1), keeping only B5's top-200 genes raises direction precision 0.779 -> 0.821 and the
    pds proxy 0.632 -> 0.732; ~200 is also the reference significant-set size implied by the spec's chance
    overlap figures.

Usage:
    python models/gene_transfer_b6.py [--top-k 200]      # needs models/b5_profiles.npy (gene_transfer_b5.py)
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

import config as cfg


def sparsify(profiles, top_k):
    out = np.zeros_like(profiles)
    for i, row in enumerate(profiles):
        top = np.argsort(-np.abs(row))[:top_k]
        out[i, top] = row[top]
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--top-k", type=int, default=200)
    top_k = p.parse_args().top_k
    models = cfg.PROJECT_ROOT / "models"
    b5 = np.load(models / "b5_profiles.npy")
    meta = json.loads((models / "b5_meta.json").read_text())
    np.save(models / "b6_profiles.npy", sparsify(b5, top_k).astype(np.float32))
    meta.update({"top_k": top_k, "transform": "t4_sparse_thinning", "built_from": "b5"})
    (models / "b6_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[B6] kept top-{top_k} genes of each of {b5.shape[0]} B5 profiles; saved models/b6_profiles.npy")
