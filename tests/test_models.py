"""Fast checks on synthetic data -- the real controls / vcc-perturb files never live in the repo."""
import numpy as np
import scipy.sparse as sp

import config as cfg
from models.gene_transfer_b2 import discrimination_score
from models.gene_transfer_b4 import build_profiles, loo
from models.generate_b2_submission import align_z_matrix_to_context, build_combined_obs
from models.transform_candidates import t2_log1p_additive


def synthetic_response(n_known=12, n_genes=40, seed=0):
    rng = np.random.default_rng(seed)
    gene_cols = [f"G{i}" for i in range(n_genes)]
    targets = gene_cols[:n_known]
    shared = rng.normal(0.3, 0.2, n_genes)
    R = shared + rng.normal(0, 0.3, (n_known, n_genes))
    R[np.arange(n_known), np.arange(n_known)] = -3.0  # strong self-knockdown, as in vcc-perturb
    return targets, gene_cols, R


def test_discrimination_perfect_when_prediction_equals_truth():
    _, _, R = synthetic_response()
    assert discrimination_score(R, R) == 1.0


def test_b4_profiles_real_passthrough_and_self_knockdown():
    targets, gene_cols, R = synthetic_response()
    query = [targets[0], "G20", "NOT_IN_VCC_PERTURB"]
    prof, is_real = build_profiles(query, targets, gene_cols, R, scale=1.0)
    assert is_real.tolist() == [True, False, False]
    np.testing.assert_allclose(prof[0], R[0])
    assert prof[1, 20] == -3.0                                     # own gene knocked down
    np.testing.assert_allclose(np.delete(prof[1], 20), np.delete(R.mean(0), 20))
    np.testing.assert_allclose(prof[2], R.mean(0))                 # no own column -> plain mean


def test_b4_loo_beats_null():
    targets, gene_cols, R = synthetic_response()
    r = loo(targets, gene_cols, R)
    assert r["mean+self"]["best_mse"] < r["null_mse"]
    assert r["mean+self"]["best_mse"] <= r["mean"]["best_mse"]


def test_t2_counts_valid_and_negative_z_lowers_gene():
    rng = np.random.default_rng(0)
    base = sp.csr_matrix(rng.poisson(5, (200, 30)))
    z = np.zeros(30)
    z[3] = -3.0
    ref = {"sigma_log1p_cp10k_floored": np.full(30, 0.3)}
    out = t2_log1p_additive(base, z, ref, rng, beta=1.5)
    assert np.issubdtype(out.dtype, np.integer) and out.min() >= 0
    np.testing.assert_array_equal(out.sum(1), np.asarray(base.sum(1)).ravel())
    assert out[:, 3].mean() < 0.6 * base[:, 3].mean()


def test_align_fills_missing_genes_with_zero():
    z = np.array([[1.0, 2.0]])
    out = align_z_matrix_to_context(z, ["A", "B"], np.array(["B", "X", "A"]))
    np.testing.assert_array_equal(out, [[2.0, 0.0, 1.0]])


def test_combined_obs_shape_and_unique_names():
    genes = [f"T{i}" for i in range(cfg.N_TARGET_GENES)]
    obs = build_combined_obs(genes)
    assert obs.shape[0] == len(cfg.VAL_CONTEXTS) * cfg.N_TARGET_GENES * cfg.CELLS_PER_PERT
    assert obs.index.is_unique
    assert (obs[cfg.PERT_COL].value_counts() == cfg.CELLS_PER_PERT * len(cfg.VAL_CONTEXTS)).all()
