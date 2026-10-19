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


def _fake_external(n=6, g=30, seed=1):
    rng = np.random.default_rng(seed)
    K, T = rng.normal(0, 2, (n, g)), rng.normal(0, 2, (n, g))
    K[0] = np.nan                       # a target K562 doesn't cover
    return {"D_K562": K, "D_T_mean": T}, np.array([3, -1, 5, 7, -1, 2])


def test_b5v2_with_changes_off_matches_b5():
    from models.gene_transfer_b5 import combined_direction as v1
    from models.gene_transfer_b5v2 import combined_direction as v2
    npz, self_cols = _fake_external()
    np.testing.assert_array_equal(v2(npz, "D_", 0, 0, self_cols), v1(npz, "D_", self_cols))


def test_b5v2_agreement_and_zhu_threshold():
    from models.gene_transfer_b5 import combined_direction as v1
    from models.gene_transfer_b5v2 import combined_direction as v2
    npz, self_cols = _fake_external()
    D1, D2 = v1(npz, "D_", self_cols), v2(npz, "D_", 0, 1, self_cols)
    assert np.all(np.abs(D2) <= np.abs(D1) + 1e-12)              # agreement only shrinks
    np.testing.assert_allclose(D2[0], D1[0])                       # single-source target untouched
    # Zhu soft-threshold at 1: |z| <= 1 drops out, the rest shrink by 1 before unit-normalising
    U = v2({"D_K562": np.zeros((1, 4)), "D_T_mean": np.array([[0.5, -0.9, 3.0, -2.0]])}, "D_", 1, 0)
    np.testing.assert_allclose(U, [[0, 0, 2 / np.sqrt(5), -1 / np.sqrt(5)]])


def test_b5v3_profiles_average_available_sources_and_zero_self():
    from models.gene_transfer_b5v3 import build_profiles
    nan = np.nan
    K = np.array([[1.0, 2.0, 0.0], [nan, nan, nan], [nan, nan, nan], [4.0, 0.0, 2.0]])
    T = np.array([[3.0, 0.0, 2.0], [1.0, -1.0, 0.5], [nan, nan, nan], [nan, nan, nan]])
    prof, src = build_profiles(K, T, 2.0, np.array([2, -1, -1, 0]))
    np.testing.assert_allclose(prof, [[4, 2, 0], [2, -2, 1], [0, 0, 0], [0, 0, 4]])
    assert src == ["K562+T", "T", "none", "K562"]


def test_t5_log2_fold_edits_only_changed_genes():
    from models.transform_candidates import t5_log2_fold
    rng = np.random.default_rng(0)
    X = sp.csr_matrix(rng.poisson(5, (50, 6)).astype(np.float32))
    ref = {"raw_mean_count": np.full(6, 5.0)}
    lfc = np.array([0.0, -1.0, 2.0, 0.0, -10.0, 0.5])
    out = t5_log2_fold(X, lfc, ref, rng)
    base = X.toarray()
    np.testing.assert_array_equal(out[:, [0, 3]], base[:, [0, 3]])       # fold 1: untouched
    assert np.all(out[:, 1] <= base[:, 1]) and np.all(out[:, 2] >= base[:, 2])
    assert out[:, 4].sum() < 0.1 * base[:, 4].sum()                       # clipped at 1/20
    np.testing.assert_array_equal(t5_log2_fold(X, np.zeros(6), ref, rng), base)


def test_pie_convert_modes_and_alignment():
    from models.pie_convert import align, to_log2fold
    p = np.array([0.9, 0.2, 0.6]); lfc = np.array([1.0, -2.0, -0.5]); d = np.array([0.0, 0.0, np.log(2)])
    np.testing.assert_allclose(to_log2fold(p, lfc, d, "lfc_de"), [1.0, 0.0, -0.5])
    np.testing.assert_allclose(to_log2fold(p, lfc, d, "lfc_soft"), [0.9, -0.4, -0.3])
    m = np.log1p(np.array([10.0, 0.0, 10.0]))          # control mean log1p; gene 2 shifts by log 2
    out = to_log2fold(p, lfc, d, "delta", m)
    assert out[0] == 0 and out[1] == 0 and 1.0 < out[2] < 1.1   # 10 -> 21: a bit over a doubling, as log1p implies
    np.testing.assert_allclose(align(np.array([1.0, 2.0]), ["a", "b"], ["b", "x", "a"]), [2.0, 0.0, 1.0])
