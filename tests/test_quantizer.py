"""Tests for the TurboQuant quantizer engine."""

import numpy as np
import pytest

from langchain_turboquant.lloyd_max import compute_codebook, get_codebook
from langchain_turboquant.quantizer import TurboQuantizer


# ---------------------------------------------------------------------------
# Lloyd-Max codebook tests
# ---------------------------------------------------------------------------

class TestLloydMax:
    def test_codebook_shape(self):
        cb = compute_codebook(bits=3, dim=128)
        assert cb.shape == (8,)  # 2^3 = 8

    def test_codebook_sorted(self):
        cb = compute_codebook(bits=3, dim=128)
        assert np.all(cb[:-1] <= cb[1:])

    def test_codebook_range(self):
        cb = compute_codebook(bits=3, dim=128)
        assert np.all(cb >= -1.0)
        assert np.all(cb <= 1.0)

    def test_codebook_cache(self):
        cb1 = get_codebook(bits=3, dim=64)
        cb2 = get_codebook(bits=3, dim=64)
        assert cb1 is cb2  # Same object from cache

    def test_codebook_symmetry(self):
        """Codebook should be roughly symmetric around 0."""
        cb = compute_codebook(bits=3, dim=256)
        assert abs(cb.mean()) < 0.1

    def test_different_bits(self):
        cb2 = compute_codebook(bits=2, dim=128)
        cb3 = compute_codebook(bits=3, dim=128)
        assert len(cb2) == 4
        assert len(cb3) == 8


# ---------------------------------------------------------------------------
# TurboQuantizer tests
# ---------------------------------------------------------------------------

class TestTurboQuantizer:
    @pytest.fixture
    def quantizer(self):
        return TurboQuantizer(dim=64, bits=3, seed=42)

    @pytest.fixture
    def random_vectors(self):
        rng = np.random.RandomState(123)
        return rng.randn(100, 64).astype(np.float32)

    def test_quantize_shape(self, quantizer, random_vectors):
        compressed = quantizer.quantize(random_vectors)
        assert compressed.indices.shape == (100, 64)
        assert compressed.qjl_bits.shape == (100, 64)
        assert compressed.gammas.shape == (100,)
        assert compressed.norms.shape == (100,)

    def test_quantize_single_vector(self, quantizer):
        vec = np.random.randn(64).astype(np.float32)
        compressed = quantizer.quantize(vec)
        assert compressed.indices.shape == (1, 64)

    def test_dequantize_shape(self, quantizer, random_vectors):
        compressed = quantizer.quantize(random_vectors)
        reconstructed = quantizer.dequantize(compressed)
        assert reconstructed.shape == (100, 64)

    def test_reconstruction_quality(self, quantizer, random_vectors):
        """Reconstructed vectors should be correlated with originals."""
        compressed = quantizer.quantize(random_vectors)
        reconstructed = quantizer.dequantize(compressed)

        # Cosine similarity between each original and reconstruction
        cos_sims = []
        for orig, recon in zip(random_vectors, reconstructed):
            cos = np.dot(orig, recon) / (np.linalg.norm(orig) * np.linalg.norm(recon) + 1e-10)
            cos_sims.append(cos)
        mean_cos = np.mean(cos_sims)
        assert mean_cos > 0.8, f"Mean cosine similarity {mean_cos:.3f} too low"

    def test_asymmetric_scores_shape(self, quantizer, random_vectors):
        compressed = quantizer.quantize(random_vectors)
        query = np.random.randn(64).astype(np.float32)
        scores = quantizer.asymmetric_scores(query, compressed)
        assert scores.shape == (100,)

    def test_cosine_scores_range(self, quantizer, random_vectors):
        compressed = quantizer.quantize(random_vectors)
        query = np.random.randn(64).astype(np.float32)
        scores = quantizer.cosine_scores(query, compressed)
        # Cosine similarity should be in [-1, 1] (with some numerical slack)
        assert np.all(scores >= -1.5)
        assert np.all(scores <= 1.5)

    def test_self_similarity_highest(self, quantizer):
        """A vector should be most similar to itself."""
        rng = np.random.RandomState(999)
        vectors = rng.randn(10, 64).astype(np.float32)
        compressed = quantizer.quantize(vectors)

        query = vectors[0]
        scores = quantizer.cosine_scores(query, compressed)
        assert np.argmax(scores) == 0

    def test_compression_ratio(self, quantizer):
        ratio = quantizer.compression_ratio()
        assert ratio > 3.0, f"Compression ratio {ratio:.1f}x is too low"

    def test_compressed_bytes(self, quantizer):
        orig = quantizer.original_bytes_per_vector()
        comp = quantizer.compressed_bytes_per_vector()
        assert comp < orig

    def test_reproducibility(self):
        """Same seed → same results."""
        q1 = TurboQuantizer(dim=32, bits=3, seed=42)
        q2 = TurboQuantizer(dim=32, bits=3, seed=42)
        vec = np.random.randn(5, 32).astype(np.float32)
        c1 = q1.quantize(vec)
        c2 = q2.quantize(vec)
        np.testing.assert_array_equal(c1.indices, c2.indices)
        np.testing.assert_array_equal(c1.qjl_bits, c2.qjl_bits)

    def test_high_dim(self):
        """Should work with high-dimensional vectors (e.g. 1536d)."""
        q = TurboQuantizer(dim=1536, bits=3, seed=42)
        vec = np.random.randn(5, 1536).astype(np.float32)
        compressed = q.quantize(vec)
        assert compressed.indices.shape == (5, 1536)
        assert q.compression_ratio() > 4.0


# ---------------------------------------------------------------------------
# Top-k recall test
# ---------------------------------------------------------------------------

class TestRecall:
    def test_top10_recall(self):
        """TurboQuant top-10 recall should be >= 70% vs brute-force."""
        rng = np.random.RandomState(42)
        dim = 128
        n = 500
        vectors = rng.randn(n, dim).astype(np.float32)
        query = rng.randn(dim).astype(np.float32)

        # Ground truth: exact cosine similarity
        norms = np.linalg.norm(vectors, axis=1)
        exact_scores = vectors @ query / (norms * np.linalg.norm(query))
        exact_top10 = set(np.argsort(exact_scores)[::-1][:10])

        # TurboQuant approximate
        quantizer = TurboQuantizer(dim=dim, bits=3, seed=42)
        compressed = quantizer.quantize(vectors)
        approx_scores = quantizer.cosine_scores(query, compressed)
        approx_top10 = set(np.argsort(approx_scores)[::-1][:10])

        recall = len(exact_top10 & approx_top10) / 10.0
        assert recall >= 0.7, f"Top-10 recall {recall:.1%} is too low"
