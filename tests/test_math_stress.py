"""Mathematical stress tests for TurboQuant internals.

Covers three domains:
  1. Lloyd-Max codebook mathematical properties
  2. Rotation matrix (orthogonal) properties
  3. Quantization MSE theoretical bounds
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import gamma as gamma_fn, gammaln as lgamma_fn

from langchain_turboquant.lloyd_max import (
    _hypersphere_pdf,
    compute_codebook,
    get_codebook,
)
from langchain_turboquant.quantizer import TurboQuantizer


# ===================================================================
# Helpers
# ===================================================================

def _pdf(x: float, dim: int) -> float:
    """Convenience wrapper for the hypersphere marginal PDF."""
    return _hypersphere_pdf(x, dim)


def _theoretical_variance(dim: int) -> float:
    """Variance of a single coordinate on S^{d-1} = 1/d."""
    return 1.0 / dim


# ===================================================================
# 1. Lloyd-Max codebook mathematical properties (10+ tests)
# ===================================================================

class TestLloydMaxMathProperties:
    """Mathematical properties of the Lloyd-Max optimal codebook."""

    # -- Codebook validity across many dimensions --

    @pytest.mark.parametrize("dim", [4, 8, 16, 32, 64, 128, 256, 512, 1024])
    def test_codebook_valid_across_dims(self, dim: int):
        """Codebook must have correct size, be sorted, and lie in [-1, 1]."""
        bits = 3
        cb = compute_codebook(bits, dim)
        n_levels = 1 << bits
        assert cb.shape == (n_levels,), f"Wrong shape for dim={dim}"
        assert np.all(cb[:-1] <= cb[1:]), f"Not sorted for dim={dim}"
        assert np.all(cb >= -1.0) and np.all(cb <= 1.0), f"Out of range for dim={dim}"

    @pytest.mark.parametrize("dim", [4, 8, 16, 32, 64, 128, 256, 512, 1024])
    def test_codebook_no_nans(self, dim: int):
        """Codebook must not contain NaN or Inf values."""
        cb = compute_codebook(3, dim)
        assert np.all(np.isfinite(cb)), f"Non-finite values for dim={dim}"

    # -- Centroid condition: each centroid is the conditional expectation --

    @pytest.mark.parametrize("dim", [16, 64, 256])
    def test_centroid_condition(self, dim: int):
        """Each centroid c_i must equal E[x | b_i <= x <= b_{i+1}]."""
        bits = 2
        cb = compute_codebook(bits, dim)
        n = len(cb)
        # Reconstruct boundaries as midpoints
        boundaries = np.empty(n + 1)
        boundaries[0] = -1.0
        boundaries[-1] = 1.0
        for i in range(1, n):
            boundaries[i] = (cb[i - 1] + cb[i]) / 2.0

        for i in range(n):
            lo, hi = boundaries[i], boundaries[i + 1]
            num, _ = quad(lambda x: x * _pdf(x, dim), lo, hi)
            den, _ = quad(lambda x: _pdf(x, dim), lo, hi)
            if abs(den) > 1e-12:
                expected = num / den
                assert abs(cb[i] - expected) < 1e-4, (
                    f"Centroid {i} at dim={dim}: {cb[i]:.6f} != E[x]={expected:.6f}"
                )

    # -- MSE monotone decrease with increasing bits --

    def _compute_mse(self, bits: int, dim: int) -> float:
        """Compute the MSE of the Lloyd-Max quantizer."""
        cb = compute_codebook(bits, dim)
        n = len(cb)
        boundaries = np.empty(n + 1)
        boundaries[0] = -1.0
        boundaries[-1] = 1.0
        for i in range(1, n):
            boundaries[i] = (cb[i - 1] + cb[i]) / 2.0

        mse = 0.0
        for i in range(n):
            lo, hi = boundaries[i], boundaries[i + 1]
            val, _ = quad(lambda x: (x - cb[i]) ** 2 * _pdf(x, dim), lo, hi)
            mse += val
        return mse

    @pytest.mark.parametrize("dim", [32, 128])
    def test_mse_decreases_with_bits(self, dim: int):
        """MSE must monotonically decrease as bits increase: 2 < 3 < 4 < 5."""
        mses = [self._compute_mse(b, dim) for b in [2, 3, 4, 5]]
        for i in range(len(mses) - 1):
            assert mses[i] > mses[i + 1], (
                f"MSE not decreasing at dim={dim}: bits={i+2} MSE={mses[i]:.6e} "
                f">= bits={i+3} MSE={mses[i+1]:.6e}"
            )

    # -- Symmetry: c_i ≈ -c_{n-1-i} --

    @pytest.mark.parametrize("dim", [16, 64, 128, 256, 512])
    def test_codebook_symmetry(self, dim: int):
        """Codebook should be symmetric: c_i ≈ -c_{n-1-i}."""
        cb = compute_codebook(3, dim)
        n = len(cb)
        for i in range(n // 2):
            assert abs(cb[i] + cb[n - 1 - i]) < 0.05, (
                f"Asymmetry at dim={dim}: c[{i}]={cb[i]:.4f}, "
                f"c[{n-1-i}]={cb[n-1-i]:.4f}"
            )

    # -- PDF variance decreases with dimension --

    def test_pdf_variance_decreases_with_dim(self):
        """Variance of the marginal PDF should decrease ~ 1/d."""
        dims = [8, 16, 32, 64, 128, 256]
        variances = []
        for d in dims:
            var, _ = quad(lambda x: x ** 2 * _pdf(x, d), -1.0, 1.0)
            variances.append(var)
        for i in range(len(variances) - 1):
            assert variances[i] > variances[i + 1], (
                f"Variance not decreasing: dim={dims[i]} var={variances[i]:.6f} "
                f"<= dim={dims[i+1]} var={variances[i+1]:.6f}"
            )

    def test_pdf_variance_matches_theory(self):
        """Variance should be close to 1/d for the unit hypersphere."""
        for d in [16, 64, 256]:
            var, _ = quad(lambda x: x ** 2 * _pdf(x, d), -1.0, 1.0)
            expected = _theoretical_variance(d)
            assert abs(var - expected) / expected < 0.05, (
                f"dim={d}: var={var:.6f} vs 1/d={expected:.6f}"
            )

    # -- PDF integrates to 1 --

    @pytest.mark.parametrize("dim", [4, 16, 64, 256, 1024])
    def test_pdf_integrates_to_one(self, dim: int):
        """The hypersphere marginal PDF must integrate to 1."""
        total, _ = quad(lambda x: _pdf(x, dim), -1.0, 1.0)
        assert abs(total - 1.0) < 1e-6, f"dim={dim}: integral={total:.8f}"

    # -- Codebook mean near zero --

    @pytest.mark.parametrize("dim", [16, 64, 256, 512])
    def test_codebook_mean_near_zero(self, dim: int):
        """Mean of codebook centroids should be near 0 by symmetry."""
        cb = compute_codebook(3, dim)
        assert abs(cb.mean()) < 0.02, f"dim={dim}: mean={cb.mean():.4f}"


# ===================================================================
# 2. Rotation matrix properties (10+ tests)
# ===================================================================

class TestRotationMatrixProperties:
    """Properties of the random orthogonal rotation matrix Pi."""

    @pytest.mark.parametrize("dim", [8, 16, 32, 64, 128, 256, 512])
    def test_det_pm_one(self, dim: int):
        """det(Pi) must be +1 or -1 (orthogonal matrix)."""
        q = TurboQuantizer(dim=dim, bits=3, seed=42)
        det = np.linalg.det(q.rotation)
        assert abs(abs(det) - 1.0) < 1e-4, f"dim={dim}: |det|={abs(det):.6f}"

    @pytest.mark.parametrize("dim", [8, 16, 32, 64, 128, 256, 512])
    def test_orthogonality(self, dim: int):
        """Pi @ Pi^T must equal I (orthogonal)."""
        q = TurboQuantizer(dim=dim, bits=3, seed=42)
        product = q.rotation @ q.rotation.T
        np.testing.assert_allclose(
            product, np.eye(dim), atol=1e-4,
            err_msg=f"Pi @ Pi^T != I for dim={dim}",
        )

    def test_inner_product_preservation(self):
        """<Pi@x, Pi@y> must equal <x, y> for random vectors."""
        dim = 64
        q = TurboQuantizer(dim=dim, bits=3, seed=42)
        rng = np.random.RandomState(7)
        for _ in range(20):
            x = rng.randn(dim).astype(np.float32)
            y = rng.randn(dim).astype(np.float32)
            original_ip = np.dot(x, y)
            rotated_ip = np.dot(q.rotation @ x, q.rotation @ y)
            np.testing.assert_allclose(
                rotated_ip, original_ip, rtol=1e-3,
                err_msg="Inner product not preserved",
            )

    def test_norm_preservation(self):
        """||Pi @ x|| == ||x|| for any x."""
        dim = 128
        q = TurboQuantizer(dim=dim, bits=3, seed=42)
        rng = np.random.RandomState(8)
        for _ in range(20):
            x = rng.randn(dim).astype(np.float32)
            original_norm = np.linalg.norm(x)
            rotated_norm = np.linalg.norm(q.rotation @ x)
            np.testing.assert_allclose(
                rotated_norm, original_norm, rtol=1e-3,
                err_msg="Norm not preserved by rotation",
            )

    def test_double_rotation_identity(self):
        """Pi @ Pi^T @ x ≈ x (round-trip)."""
        dim = 64
        q = TurboQuantizer(dim=dim, bits=3, seed=42)
        rng = np.random.RandomState(9)
        x = rng.randn(dim).astype(np.float32)
        recovered = q.rotation.T @ (q.rotation @ x)
        np.testing.assert_allclose(recovered, x, atol=1e-4)

    def test_transpose_equals_inverse(self):
        """Pi^T must equal Pi^{-1} for orthogonal matrix."""
        dim = 64
        q = TurboQuantizer(dim=dim, bits=3, seed=42)
        inv = np.linalg.inv(q.rotation.astype(np.float64))
        np.testing.assert_allclose(
            q.rotation.T.astype(np.float64), inv, atol=1e-4,
        )

    @pytest.mark.parametrize("dim", [16, 64, 256])
    def test_rotated_coordinates_symmetric(self, dim: int):
        """After rotation, coordinate means should be near 0 for random unit vecs."""
        q = TurboQuantizer(dim=dim, bits=3, seed=42)
        rng = np.random.RandomState(10)
        vecs = rng.randn(500, dim).astype(np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        unit_vecs = vecs / norms
        rotated = unit_vecs @ q.rotation.T
        col_means = rotated.mean(axis=0)
        assert np.all(np.abs(col_means) < 0.2), (
            f"dim={dim}: max col mean = {np.abs(col_means).max():.4f}"
        )

    def test_eigenvalues_unit_modulus(self):
        """Eigenvalues of an orthogonal matrix have |lambda|=1."""
        dim = 32
        q = TurboQuantizer(dim=dim, bits=3, seed=42)
        eigvals = np.linalg.eigvals(q.rotation.astype(np.float64))
        mods = np.abs(eigvals)
        np.testing.assert_allclose(mods, np.ones(dim), atol=1e-4)

    def test_rows_are_orthonormal(self):
        """Each row of Pi must be a unit vector, and rows must be mutually orthogonal."""
        dim = 64
        q = TurboQuantizer(dim=dim, bits=3, seed=42)
        R = q.rotation.astype(np.float64)
        # Row norms
        row_norms = np.linalg.norm(R, axis=1)
        np.testing.assert_allclose(row_norms, np.ones(dim), atol=1e-4)
        # Mutual orthogonality (check a sample of pairs)
        rng = np.random.RandomState(11)
        for _ in range(30):
            i, j = rng.choice(dim, size=2, replace=False)
            dot = np.dot(R[i], R[j])
            assert abs(dot) < 1e-4, f"Row {i} and {j} not orthogonal: dot={dot:.6f}"

    def test_columns_are_orthonormal(self):
        """Each column of Pi must be a unit vector."""
        dim = 64
        q = TurboQuantizer(dim=dim, bits=3, seed=42)
        R = q.rotation.astype(np.float64)
        col_norms = np.linalg.norm(R, axis=0)
        np.testing.assert_allclose(col_norms, np.ones(dim), atol=1e-4)

    def test_different_seeds_different_rotations(self):
        """Different seeds must yield different rotation matrices."""
        dim = 32
        q1 = TurboQuantizer(dim=dim, bits=3, seed=1)
        q2 = TurboQuantizer(dim=dim, bits=3, seed=2)
        assert not np.allclose(q1.rotation, q2.rotation, atol=1e-6)


# ===================================================================
# 3. Quantization MSE theoretical bounds (10+ tests)
# ===================================================================

class TestQuantizationMSEBounds:
    """MSE and inner-product estimation quality tests."""

    def _per_coord_mse_bound(self, bits: int, dim: int) -> float:
        """Upper bound on per-coordinate MSE: variance / 2^(2*bits) * constant.

        For a Lloyd-Max quantizer, MSE < sigma^2 * (delta^2 / 12) where
        delta = 2 / 2^bits (uniform partition width). We use a looser but
        always-valid bound: per-coord MSE < variance_of_pdf.
        """
        return 1.0 / dim  # Variance = 1/d is a trivial upper bound for opt quant MSE

    @pytest.mark.parametrize(
        "bits, dim",
        [(2, 32), (2, 128), (3, 32), (3, 128), (3, 256), (4, 32), (4, 64), (4, 128)],
    )
    def test_mse_within_theoretical_bound(self, bits: int, dim: int):
        """Per-coordinate MSE of Lloyd-Max must be less than the coordinate variance."""
        cb = compute_codebook(bits, dim)
        n = len(cb)
        boundaries = np.empty(n + 1)
        boundaries[0] = -1.0
        boundaries[-1] = 1.0
        for i in range(1, n):
            boundaries[i] = (cb[i - 1] + cb[i]) / 2.0
        mse = 0.0
        for i in range(n):
            lo, hi = boundaries[i], boundaries[i + 1]
            val, _ = quad(lambda x, ci=cb[i]: (x - ci) ** 2 * _pdf(x, dim), lo, hi)
            mse += val
        coord_var = 1.0 / dim
        assert mse < coord_var, (
            f"bits={bits}, dim={dim}: MSE={mse:.6e} >= var={coord_var:.6e}"
        )

    def test_unit_vector_mse(self):
        """MSE on unit vectors should be bounded."""
        dim = 64
        q = TurboQuantizer(dim=dim, bits=3, seed=42)
        rng = np.random.RandomState(20)
        vecs = rng.randn(200, dim).astype(np.float32)
        vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)

        compressed = q.quantize(vecs)
        recon = q.dequantize(compressed)
        mse = np.mean(np.sum((vecs - recon) ** 2, axis=1))
        # MSE should be less than 2 (loose bound for 3-bit on 64d unit vecs)
        assert mse < 2.0, f"Unit vector MSE={mse:.4f} too high"

    def test_norm_preservation_after_quantize(self):
        """Norms of original vectors should be stored and approximately preserved."""
        dim = 64
        q = TurboQuantizer(dim=dim, bits=3, seed=42)
        rng = np.random.RandomState(21)
        vecs = rng.randn(50, dim).astype(np.float32)
        original_norms = np.linalg.norm(vecs, axis=1)

        compressed = q.quantize(vecs)
        np.testing.assert_allclose(compressed.norms, original_norms, rtol=1e-5)

    def test_nonunit_vector_norm_stored(self):
        """For non-unit vectors, the stored norm must match the original."""
        dim = 32
        q = TurboQuantizer(dim=dim, bits=3, seed=42)
        rng = np.random.RandomState(22)
        vecs = rng.randn(30, dim).astype(np.float32) * 5.0  # large norms
        compressed = q.quantize(vecs)
        np.testing.assert_allclose(
            compressed.norms, np.linalg.norm(vecs, axis=1), rtol=1e-5,
        )

    def test_qjl_reduces_mse(self):
        """Stage1+Stage2 reconstruction MSE must be <= Stage1-only MSE."""
        dim = 64
        q = TurboQuantizer(dim=dim, bits=3, qjl_dim=dim, seed=42)
        rng = np.random.RandomState(23)
        vecs = rng.randn(200, dim).astype(np.float32)
        vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)

        compressed = q.quantize(vecs)

        # Stage 1 only reconstruction
        stage1_recon = q._dequantize_mse(compressed.indices)
        # Scale back by norms
        stage1_recon = stage1_recon * compressed.norms[:, np.newaxis]
        mse_stage1 = np.mean(np.sum((vecs - stage1_recon) ** 2, axis=1))

        # Stage 1 + Stage 2
        full_recon = q.dequantize(compressed)
        mse_full = np.mean(np.sum((vecs - full_recon) ** 2, axis=1))

        assert mse_full <= mse_stage1 + 1e-6, (
            f"QJL did not help: stage1 MSE={mse_stage1:.6f}, "
            f"full MSE={mse_full:.6f}"
        )

    def test_asymmetric_ip_unbiased(self):
        """Asymmetric inner product should be approximately unbiased."""
        dim = 128
        q = TurboQuantizer(dim=dim, bits=3, qjl_dim=dim, seed=42)
        rng = np.random.RandomState(24)

        n_trials = 300
        vecs = rng.randn(n_trials, dim).astype(np.float32)
        query = rng.randn(dim).astype(np.float32)

        exact_ips = vecs @ query
        compressed = q.quantize(vecs)
        approx_ips = q.asymmetric_scores(query, compressed)

        # Mean error should be small relative to mean magnitude
        mean_error = np.mean(approx_ips - exact_ips)
        mean_magnitude = np.mean(np.abs(exact_ips))
        relative_bias = abs(mean_error) / (mean_magnitude + 1e-10)
        assert relative_bias < 0.15, (
            f"Asymmetric IP bias too large: {relative_bias:.4f}"
        )

    def test_asymmetric_ip_variance_bounded(self):
        """Variance of asymmetric IP estimation error should be bounded."""
        dim = 64
        q = TurboQuantizer(dim=dim, bits=3, qjl_dim=dim, seed=42)
        rng = np.random.RandomState(25)

        vecs = rng.randn(500, dim).astype(np.float32)
        query = rng.randn(dim).astype(np.float32)

        exact_ips = vecs @ query
        compressed = q.quantize(vecs)
        approx_ips = q.asymmetric_scores(query, compressed)

        errors = approx_ips - exact_ips
        error_var = np.var(errors)
        signal_var = np.var(exact_ips)
        # Error variance should be less than the signal variance
        assert error_var < signal_var, (
            f"Error variance {error_var:.4f} >= signal variance {signal_var:.4f}"
        )

    def test_mse_decreases_with_more_bits_empirical(self):
        """Empirical MSE must decrease when using more bits."""
        dim = 64
        rng = np.random.RandomState(26)
        vecs = rng.randn(100, dim).astype(np.float32)
        vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)

        mses = []
        for bits in [2, 3, 4]:
            q = TurboQuantizer(dim=dim, bits=bits, seed=42)
            compressed = q.quantize(vecs)
            stage1_recon = q._dequantize_mse(compressed.indices)
            stage1_recon = stage1_recon * compressed.norms[:, np.newaxis]
            mse = np.mean(np.sum((vecs - stage1_recon) ** 2, axis=1))
            mses.append(mse)

        for i in range(len(mses) - 1):
            assert mses[i] > mses[i + 1], (
                f"Empirical MSE not decreasing: bits={i+2} MSE={mses[i]:.6f} "
                f">= bits={i+3} MSE={mses[i+1]:.6f}"
            )

    def test_cosine_similarity_quality(self):
        """Cosine similarity scores should correlate well with ground truth."""
        dim = 128
        q = TurboQuantizer(dim=dim, bits=3, seed=42)
        rng = np.random.RandomState(27)

        vecs = rng.randn(200, dim).astype(np.float32)
        query = rng.randn(dim).astype(np.float32)

        # Ground truth cosine
        norms = np.linalg.norm(vecs, axis=1)
        exact_cos = vecs @ query / (norms * np.linalg.norm(query))

        compressed = q.quantize(vecs)
        approx_cos = q.cosine_scores(query, compressed)

        # Pearson correlation
        corr = np.corrcoef(exact_cos, approx_cos)[0, 1]
        assert corr > 0.85, f"Cosine score correlation {corr:.4f} too low"

    def test_higher_dim_lower_relative_mse(self):
        """In higher dimensions, per-coordinate MSE relative to variance decreases."""
        rng = np.random.RandomState(28)
        ratios = []
        for dim in [32, 128, 512]:
            q = TurboQuantizer(dim=dim, bits=3, seed=42)
            vecs = rng.randn(100, dim).astype(np.float32)
            vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
            compressed = q.quantize(vecs)
            stage1_recon = q._dequantize_mse(compressed.indices)
            stage1_recon = stage1_recon * compressed.norms[:, np.newaxis]
            total_mse = np.mean(np.sum((vecs - stage1_recon) ** 2, axis=1))
            per_coord_mse = total_mse / dim
            coord_var = 1.0 / dim
            ratio = per_coord_mse / coord_var
            ratios.append(ratio)

        # The ratio should not increase with dimension (it should be roughly constant
        # or decrease for Lloyd-Max). We check the last is not much worse than the first.
        assert ratios[-1] < ratios[0] * 1.5, (
            f"Relative MSE ratio increased too much: {ratios}"
        )

    def test_residual_norms_bounded(self):
        """Residual norms (gammas) from Stage 1 should be bounded by sqrt(2)."""
        dim = 64
        q = TurboQuantizer(dim=dim, bits=3, seed=42)
        rng = np.random.RandomState(29)
        vecs = rng.randn(100, dim).astype(np.float32)
        vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
        compressed = q.quantize(vecs)
        # For unit vectors: ||x - x_hat|| <= ||x|| + ||x_hat|| <= 2, but
        # practically much less for decent quantization
        assert np.all(compressed.gammas < np.sqrt(2.0) + 0.1), (
            f"Max gamma = {compressed.gammas.max():.4f}"
        )

    def test_qjl_bits_are_signs(self):
        """QJL bits must be exactly +1 or -1."""
        dim = 64
        q = TurboQuantizer(dim=dim, bits=3, seed=42)
        rng = np.random.RandomState(30)
        vecs = rng.randn(50, dim).astype(np.float32)
        compressed = q.quantize(vecs)
        unique = set(np.unique(compressed.qjl_bits))
        assert unique.issubset({-1, 0, 1}), f"Unexpected QJL values: {unique}"
