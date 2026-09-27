"""Diagnose gene name format mismatch between vcc-perturb.csv and gene_names.csv."""
import os
from pathlib import Path
import pandas as pd

CONTROLS_DIR = Path(os.environ.get("VCC_CONTROLS_DIR", "/mnt/c/Users/User/Downloads/controls"))
VCC_PERTURB  = Path(os.environ.get("VCC_PERTURB_CSV", "/mnt/c/VCC/vcc-perturb.csv"))

# gene_names.csv
gn = pd.read_csv(CONTROLS_DIR / "gene_names.csv", header=None)
gene_names = list(gn.iloc[:, 0])
print(f"gene_names.csv — {len(gene_names)} entries")
print(f"  first 10: {gene_names[:10]}")
print(f"  type of first entry: {type(gene_names[0])}")
print(f"  repr of first: {repr(gene_names[0])}")

# vcc-perturb.csv raw (no index assumption)
print()
raw = pd.read_csv(VCC_PERTURB, nrows=5)
print(f"vcc-perturb.csv — first 5 rows (no index_col):")
print(raw.to_string())
print(f"  columns: {list(raw.columns[:6])}")

# With index_col=0
print()
vp = pd.read_csv(VCC_PERTURB, index_col=0, nrows=10)
print(f"vcc-perturb.csv — index when loaded with index_col=0:")
print(f"  {list(vp.index)}")
print(f"  repr of first index: {repr(vp.index[0])}")
print(f"  columns (first 5): {list(vp.columns[:5])}")

# Cross-check
vcc_targets = list(pd.read_csv(CONTROLS_DIR / "pert_counts.csv")["target_gene"].str.strip())
gene_idx = {g.strip(): i for i, g in enumerate(gene_names)}
known_in_genome = [g for g in vp.index if g.strip() in gene_idx]
known_in_vcc    = [g for g in vp.index if g.strip() in vcc_targets]
print(f"\nvcc-perturb index rows in genome: {len(known_in_genome)}")
print(f"vcc-perturb index rows in VCC targets: {len(known_in_vcc)}  ({known_in_vcc})")
