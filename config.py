"""
Central configuration for the vcc2026 project.

Source of truth for the constants below is the released 2026 controls bundle
(manifest.json / gene_names.csv / pert_counts.csv) and the official cell_eval2
`vcc2026` scoring spec (docs/vcc2026_metrics/vcc2026-metrics.md, retrieved from
github.com/ArcInstitute/cell-eval2 on 2026-09-07). Do not hand-edit these
without re-checking against the manifest shipped with a given phase's controls.
"""
import os
from pathlib import Path

# ---- Filesystem ----
PROJECT_ROOT = Path(__file__).resolve().parent
# Override with the VCC_CONTROLS_DIR env var when running outside the original sandbox (e.g. on a
# machine where the controls bundle lives at a different path).
CONTROLS_DIR = Path(os.environ.get("VCC_CONTROLS_DIR", "/mnt/user-data/uploads/controls"))
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_AUDIT = PROJECT_ROOT / "data" / "audit"
OUTPUTS = Path(os.environ.get("VCC_OUTPUTS_DIR", PROJECT_ROOT / "outputs"))  # override on cloud boxes with a bigger scratch disk
EXPERIMENTS = PROJECT_ROOT / "experiments"

CONTEXT_FILES = {
    "A": CONTROLS_DIR / "context_A.h5ad",
    "B": CONTROLS_DIR / "context_B.h5ad",
    "C": CONTROLS_DIR / "context_C.h5ad",
}
GENE_NAMES_CSV = CONTROLS_DIR / "gene_names.csv"
PERT_COUNTS_CSV = CONTROLS_DIR / "pert_counts.csv"
MANIFEST_JSON = CONTROLS_DIR / "manifest.json"

# ---- Schema (validation-phase panel, panel_id vcc2026-val-1) ----
PERT_COL = "target_gene"
CONTEXT_COL = "context"
CONTROL_LABEL = "non-targeting"
N_GENES = 18533
N_TARGET_GENES = 300           # per context, same 300 gene symbols in every context
CELLS_PER_PERT = 400           # exact requirement for platform upload ("vcc prep")
N_NTC_CONSTRUCTS = 46          # distinct non-targeting guide constructs per context
CONTROL_CELLS_PER_CONTEXT = 18400
VAL_CONTEXTS = ["A", "B", "C"]     # validation-phase cell lines
FINAL_CONTEXTS = ["D", "E", "F"]   # final-phase cell lines (held out, released ~Oct 22 2026)

# ---- Submission constraints (cell_eval2 vcc2026 profile, section 7) ----
MAX_STORED_ENTRIES = 4_750_000_000   # sparse nnz cap across the whole submission (3 contexts)
MAX_COUNTS_PER_CELL = 1_000_000.0    # a cell's total count must not exceed this
REQUIRE_RAW_INTEGER_COUNTS = True    # no normalize_total / log1p — scorer needs raw counts

# ---- Scoring (informational; cannot be run locally without Arc's held-out reference) ----
VCC2026_METRICS = [
    "pds_cosine",
    "expr_mse_unbiased_capped_norm",
    "de_wilcoxon_direction_fidelity_yield_raw",
    "de_wilcoxon_direction_reach_raw",
    "de_wilcoxon_sig_jaccard",
    "de_wilcoxon_lfc_nmae",
]
DE_FDR_THRESHOLD = 0.05
DE_MIN_CPM_IN_CONTROL = 5.0
PDS_TARGET_SUM = 5.0e4
DE_TARGET_SUM = 1.0e6

RANDOM_SEED = 0
