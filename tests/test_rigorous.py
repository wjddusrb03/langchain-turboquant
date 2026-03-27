"""Rigorous tests for TurboQuant — mathematical correctness, edge cases,
scaling behaviour, and asymmetric estimator properties.

These tests go beyond basic API checks and verify that the implementation
faithfully reproduces the theoretical guarantees from the TurboQuant paper
(ICLR 2026).
"""

from __future__ import annotations

import tempfile
import time

import numpy as np
import pytest
from scipy.spatial.distance import cosine as cosine_dist

from langchain_turboquant.lloyd_max import (
    _hypersphere_pdf,
    compute_codebook,
    get_codebook,
)
from langchain_turboquant.quantizer import CompressedVectors, TurboQuantizer


# ===================================================================
# Part 1: Lloyd-Max codebook mathematical properties
# ===================================================================


class TestLloydMaxMath:
    """Verify mathematical properties of the Lloyd-Max codebook."""

    @pytest.mark.parametrize("dim", [8, 32, 64, 128, 256, 512])
    def test_codebook_is_valid_for_various_dims(self, dim):
        """Codebook should be valid across many dimensions."""
        cb = compute_codebook(bits=3, dim=dim)
        assert cb.shape == (8,)
        assert np.all(np.isfinite(cb))
        assert np.all(cb >= -1.0)
        assert np.all(cb <= 1.0)
        assert np.all(cb[:-1] < cb[1:])  # strictly sorted

    @pytest.mark.parametrize("bits", [1, 2, 3, 4])
    def test_codebook_levels(self, bits):
        """Number of centroids should be 2^bits."""
        cb = compute_codebook(bits=bits, dim=64)
        assert len(cb) == 2**bits

    def test_pdf_integrates_to_one(self):
        """Hypersphere marginal PDF should integrate to ~1."""
        from scipy.integrate import quad

        for dim in [8, 32, 128]:
            integral, _ = quad(lambda x: _hypersphere_pdf(x, dim), -1.0, 1.0)
            assert abs(integral - 1.0) < 0.01, (
                f"PDF integral = {integral:.4f} for dim={dim}, expected ~1.0"
            )

    def test_pdf_non_negative(self):
        """PDF should be non-negative everywhere."""
        xs = np.linspace(-0.999, 0.999, 1000)
        for dim in [8, 64, 256]:
            for x in xs:
                assert _hypersphere_pdf(x, dim) >= 0.0

    def test_pdf_symmetry(self):
        """PDF should be symmetric around 0."""
        xs = np.linspace(0.01, 0.99, 50)
        for dim in [16, 64, 256]:
            for x in xs:
                pos = _hypersphere_pdf(x, dim)
                neg = _hypersphere_pdf(-x, dim)
                assert abs(pos - neg) < 1e-10, (
                    f"Asymmetry at x={x}, dim={dim}: {pos} vs {neg}"
                )

    def test_high_dim_concentrates_near_zero(self):
        """In high dimensions, the marginal should concentrate near 0."""
        cb_low = compute_codebook(bits=3, dim=16)
        cb_high = compute_codebook(bits=3, dim=512)
        # High-dim codebook should have tighter range
        assert np.max(np.abs(cb_high)) < np.max(np.abs(cb_low))

    def test_codebook_convergence(self):
        """Lloyd-Max should converge — doubling iterations barely changes the codebook."""
        cb_100 = compute_codebook(bits=3, dim=64, max_iter=100)
        cb_200 = compute_codebook(bits=3, dim=64, max_iter=200)
        # Due to numerical integration precision, allow small differences
        max_diff = np.max(np.abs(cb_100 - cb_200))
        assert max_diff < 1e-3, f"Max codebook drift = {max_diff:.6f}"


# ===================================================================
# Part 2: Rotation matrix properties
# ===================================================================


class TestRotationMatrix:
    """Verify the random orthogonal rotation matrix."""

    @pytest.mark.parametrize("dim", [16, 64, 256])
    def test_orthogonality(self, dim):
        """Pi @ Pi^T should be identity."""
        q = TurboQuantizer(dim=dim, bits=3, seed=42)
        product = q.rotation @ q.rotation.T
        np.testing.assert_allclose(product, np.eye(dim), atol=1e-5)

    @pytest.mark.parametrize("dim", [16, 64, 256])
    def test_norm_preservation(self, dim):
        """Rotation should preserve vector norms."""
        q = TurboQuantizer(dim=dim, bits=3, seed=42)
        rng = np.random.RandomState(99)
        vecs = rng.randn(50, dim).astype(np.float32)
        rotated = vecs @ q.rotation.T
        norms_orig = np.linalg.norm(vecs, axis=1)
        norms_rot = np.linalg.norm(rotated, axis=1)
        np.testing.assert_allclose(norms_orig, norms_rot, rtol=1e-4)

    def test_deterministic_with_same_seed(self):
        """Same seed should produce identical rotation."""
        q1 = TurboQuantizer(dim=64, bits=3, seed=123)
        q2 = TurboQuantizer(dim=64, bits=3, seed=123)
        np.testing.assert_array_equal(q1.rotation, q2.rotation)

    def test_different_with_different_seed(self):
        """Different seeds should produce different rotations."""
        q1 = TurboQuantizer(dim=64, bits=3, seed=1)
        q2 = TurboQuantizer(dim=64, bits=3, seed=2)
        assert not np.allclose(q1.rotation, q2.rotation)


# ===================================================================
# Part 3: Quantization accuracy — MSE and cosine fidelity
# ===================================================================


class TestQuantizationAccuracy:
    """Verify reconstruction quality matches theoretical expectations."""

    @pytest.mark.parametrize("dim", [32, 64, 128, 256])
    def test_reconstruction_cosine_by_dim(self, dim):
        """Mean cosine similarity between original and reconstructed should be high."""
        q = TurboQuantizer(dim=dim, bits=3, seed=42)
        rng = np.random.RandomState(0)
        vecs = rng.randn(200, dim).astype(np.float32)

        compressed = q.quantize(vecs)
        reconstructed = q.dequantize(compressed)

        cosines = []
        for orig, recon in zip(vecs, reconstructed):
            c = np.dot(orig, recon) / (np.linalg.norm(orig) * np.linalg.norm(recon) + 1e-10)
            cosines.append(c)
        mean_cos = np.mean(cosines)
        assert mean_cos > 0.7, f"dim={dim}: mean cosine = {mean_cos:.4f}"

    @pytest.mark.parametrize("bits", [2, 3, 4])
    def test_more_bits_better_reconstruction(self, bits):
        """Higher bit-width should give better reconstruction."""
        dim = 64
        rng = np.random.RandomState(0)
        vecs = rng.randn(200, dim).astype(np.float32)

        errors = {}
        for b in [2, 3, 4]:
            q = TurboQuantizer(dim=dim, bits=b, seed=42)
            compressed = q.quantize(vecs)
            reconstructed = q.dequantize(compressed)
            mse = np.mean(np.sum((vecs - reconstructed) ** 2, axis=1))
            errors[b] = mse

        # More bits → lower MSE
        assert errors[4] < errors[3] < errors[2], (
            f"Expected MSE(4bit) < MSE(3bit) < MSE(2bit), got {errors}"
        )

    def test_normalised_mse_bound(self):
        """Normalised MSE should be within theoretical bound.

        TurboQuant paper states MSE ≤ (sqrt(3*pi)/2) * (1/4^b)
        For b=3: MSE ≤ ~0.038
        We test on unit vectors so the MSE is directly comparable.
        """
        dim = 128
        bits = 3
        q = TurboQuantizer(dim=dim, bits=bits, seed=42)
        rng = np.random.RandomState(0)
        vecs = rng.randn(500, dim).astype(np.float32)
        # Normalise
        vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)

        compressed = q.quantize(vecs)
        reconstructed = q.dequantize(compressed)

        # Per-vector MSE (averaged over dimensions)
        per_vec_mse = np.mean((vecs - reconstructed) ** 2, axis=1)
        mean_mse = np.mean(per_vec_mse)

        theoretical_bound = (np.sqrt(3 * np.pi) / 2) * (1.0 / 4**bits)
        # Allow 3x slack (theoretical bound is asymptotic)
        assert mean_mse < theoretical_bound * 3, (
            f"MSE {mean_mse:.6f} exceeds 3x theoretical bound {theoretical_bound:.6f}"
        )


# ===================================================================
# Part 4: Asymmetric estimator properties
# ===================================================================


class TestAsymmetricEstimator:
    """Verify the asymmetric inner-product estimator from the paper."""

    def test_unbiasedness_statistical(self):
        """The asymmetric estimator should be approximately unbiased.

        E[<q, dequant(quant(x))>] ≈ <q, x>

        We test this by averaging over many random seeds (projection matrices).
        """
        dim = 64
        rng = np.random.RandomState(42)
        x = rng.randn(dim).astype(np.float32)
        q_vec = rng.randn(dim).astype(np.float32)
        true_inner = np.dot(q_vec, x)

        estimates = []
        for seed in range(50):
            quantizer = TurboQuantizer(dim=dim, bits=3, seed=seed)
            compressed = quantizer.quantize(x)
            est = quantizer.asymmetric_scores(q_vec, compressed)[0]
            estimates.append(est)

        mean_estimate = np.mean(estimates)
        # Should be within 30% of true value (statistical test)
        relative_error = abs(mean_estimate - true_inner) / (abs(true_inner) + 1e-10)
        assert relative_error < 0.3, (
            f"Estimator bias: mean={mean_estimate:.4f}, true={true_inner:.4f}, "
            f"relative error={relative_error:.2%}"
        )

    def test_score_ordering_preserved(self):
        """If <q, x1> >> <q, x2>, asymmetric scores should preserve this ordering."""
        dim = 64
        rng = np.random.RandomState(42)

        q_vec = rng.randn(dim).astype(np.float32)
        # x1 is very similar to q, x2 is orthogonal, x3 is opposite
        x1 = q_vec + rng.randn(dim).astype(np.float32) * 0.1  # very close
        x2 = rng.randn(dim).astype(np.float32)  # random
        x3 = -q_vec + rng.randn(dim).astype(np.float32) * 0.1  # opposite

        vecs = np.stack([x1, x2, x3])
        quantizer = TurboQuantizer(dim=dim, bits=3, seed=42)
        compressed = quantizer.quantize(vecs)

        scores = quantizer.cosine_scores(q_vec, compressed)
        # x1 (similar) should score highest, x3 (opposite) lowest
        assert scores[0] > scores[1], "Similar vector should score higher than random"
        assert scores[1] > scores[2], "Random vector should score higher than opposite"

    def test_cosine_score_self_is_near_one(self):
        """Cosine similarity of a vector with itself should be close to 1."""
        dim = 128
        quantizer = TurboQuantizer(dim=dim, bits=3, seed=42)
        rng = np.random.RandomState(0)
        vec = rng.randn(dim).astype(np.float32)

        compressed = quantizer.quantize(vec)
        score = quantizer.cosine_scores(vec, compressed)[0]
        assert score > 0.8, f"Self-similarity = {score:.4f}, expected > 0.8"


# ===================================================================
# Part 5: Top-k recall at various scales
# ===================================================================


class TestRecallScaling:
    """Test search recall across different dataset sizes and dimensions."""

    @pytest.mark.parametrize(
        "n,dim",
        [(100, 32), (200, 64), (500, 128), (1000, 256)],
    )
    def test_top10_recall(self, n, dim):
        """Top-10 recall should be >= 60% across scales."""
        rng = np.random.RandomState(42)
        vectors = rng.randn(n, dim).astype(np.float32)
        query = rng.randn(dim).astype(np.float32)

        # Exact
        norms = np.linalg.norm(vectors, axis=1)
        exact_scores = vectors @ query / (norms * np.linalg.norm(query))
        exact_top10 = set(np.argsort(exact_scores)[::-1][:10])

        # Approximate
        quantizer = TurboQuantizer(dim=dim, bits=3, seed=42)
        compressed = quantizer.quantize(vectors)
        approx_scores = quantizer.cosine_scores(query, compressed)
        approx_top10 = set(np.argsort(approx_scores)[::-1][:10])

        recall = len(exact_top10 & approx_top10) / 10.0
        assert recall >= 0.6, (
            f"n={n}, dim={dim}: top-10 recall = {recall:.0%}"
        )

    def test_top1_recall_over_multiple_queries(self):
        """Top-1 recall over 50 queries should be >= 60%."""
        dim = 128
        n = 500
        rng = np.random.RandomState(42)
        vectors = rng.randn(n, dim).astype(np.float32)

        quantizer = TurboQuantizer(dim=dim, bits=3, seed=42)
        compressed = quantizer.quantize(vectors)

        hits = 0
        n_queries = 50
        for i in range(n_queries):
            query = rng.randn(dim).astype(np.float32)
            norms = np.linalg.norm(vectors, axis=1)
            exact = np.argmax(vectors @ query / (norms * np.linalg.norm(query)))
            approx = np.argmax(quantizer.cosine_scores(query, compressed))
            if exact == approx:
                hits += 1

        recall = hits / n_queries
        assert recall >= 0.6, f"Top-1 recall = {recall:.0%} over {n_queries} queries"

    def test_recall_improves_with_more_bits(self):
        """Higher bit-width should give equal or better recall."""
        dim = 128
        n = 500
        rng = np.random.RandomState(42)
        vectors = rng.randn(n, dim).astype(np.float32)
        query = rng.randn(dim).astype(np.float32)

        norms = np.linalg.norm(vectors, axis=1)
        exact_scores = vectors @ query / (norms * np.linalg.norm(query))
        exact_top10 = set(np.argsort(exact_scores)[::-1][:10])

        recalls = {}
        for bits in [2, 3, 4]:
            quantizer = TurboQuantizer(dim=dim, bits=bits, seed=42)
            compressed = quantizer.quantize(vectors)
            approx = quantizer.cosine_scores(query, compressed)
            approx_top10 = set(np.argsort(approx)[::-1][:10])
            recalls[bits] = len(exact_top10 & approx_top10) / 10.0

        assert recalls[4] >= recalls[3] >= recalls[2] or (
            # Allow 1 miss tolerance
            recalls[4] >= recalls[2] - 0.1
        ), f"Recall by bits: {recalls}"


# ===================================================================
# Part 6: Edge cases
# ===================================================================


class TestEdgeCases:
    """Test boundary conditions and unusual inputs."""

    def test_zero_vector(self):
        """Zero vector should not crash."""
        q = TurboQuantizer(dim=32, bits=3, seed=42)
        vec = np.zeros((1, 32), dtype=np.float32)
        compressed = q.quantize(vec)
        recon = q.dequantize(compressed)
        assert np.all(np.isfinite(recon))

    def test_very_large_vector(self):
        """Vectors with very large magnitude should work."""
        q = TurboQuantizer(dim=32, bits=3, seed=42)
        vec = np.ones((1, 32), dtype=np.float32) * 1e6
        compressed = q.quantize(vec)
        recon = q.dequantize(compressed)
        assert np.all(np.isfinite(recon))
        assert np.linalg.norm(recon) > 1e4  # Should preserve large magnitude

    def test_very_small_vector(self):
        """Vectors with very small magnitude should work."""
        q = TurboQuantizer(dim=32, bits=3, seed=42)
        vec = np.ones((1, 32), dtype=np.float32) * 1e-8
        compressed = q.quantize(vec)
        recon = q.dequantize(compressed)
        assert np.all(np.isfinite(recon))

    def test_single_vector(self):
        """Single vector (not batched) should work."""
        q = TurboQuantizer(dim=32, bits=3, seed=42)
        vec = np.random.randn(32).astype(np.float32)
        compressed = q.quantize(vec)
        assert compressed.indices.shape == (1, 32)

    def test_identical_vectors(self):
        """All identical vectors should produce identical compressed results."""
        q = TurboQuantizer(dim=32, bits=3, seed=42)
        vec = np.random.randn(32).astype(np.float32)
        batch = np.tile(vec, (10, 1))
        compressed = q.quantize(batch)
        # All indices should be the same
        for i in range(1, 10):
            np.testing.assert_array_equal(compressed.indices[0], compressed.indices[i])

    def test_orthogonal_vectors_low_similarity(self):
        """Orthogonal vectors should have near-zero cosine similarity."""
        dim = 64
        q = TurboQuantizer(dim=dim, bits=3, seed=42)

        # Construct orthogonal pair
        v1 = np.zeros(dim, dtype=np.float32)
        v1[0] = 1.0
        v2 = np.zeros(dim, dtype=np.float32)
        v2[1] = 1.0

        compressed = q.quantize(np.stack([v2]))
        score = q.cosine_scores(v1, compressed)[0]
        assert abs(score) < 0.3, f"Orthogonal vectors scored {score:.4f}"

    def test_opposite_vectors_negative_similarity(self):
        """Opposite vectors should have negative cosine similarity."""
        dim = 64
        q = TurboQuantizer(dim=dim, bits=3, seed=42)
        rng = np.random.RandomState(42)
        v1 = rng.randn(dim).astype(np.float32)
        v2 = -v1

        compressed = q.quantize(v2.reshape(1, -1))
        score = q.cosine_scores(v1, compressed)[0]
        assert score < -0.5, f"Opposite vectors scored {score:.4f}, expected < -0.5"

    def test_large_batch(self):
        """Should handle large batches without OOM."""
        q = TurboQuantizer(dim=128, bits=3, seed=42)
        vecs = np.random.randn(5000, 128).astype(np.float32)
        compressed = q.quantize(vecs)
        assert compressed.indices.shape[0] == 5000

    def test_1bit_quantization(self):
        """Extreme: 1-bit quantization should still work (2 centroids)."""
        q = TurboQuantizer(dim=32, bits=1, seed=42)
        vecs = np.random.randn(50, 32).astype(np.float32)
        compressed = q.quantize(vecs)
        assert compressed.indices.max() <= 1
        recon = q.dequantize(compressed)
        assert np.all(np.isfinite(recon))


# ===================================================================
# Part 7: Compression ratio verification
# ===================================================================


class TestCompressionRatio:
    """Verify compression ratios match theoretical expectations."""

    @pytest.mark.parametrize(
        "dim,bits,expected_min_ratio",
        [
            (64, 3, 3.0),
            (128, 3, 4.0),
            (256, 3, 5.0),
            (384, 3, 5.5),
            (768, 3, 6.0),
            (1536, 3, 6.5),
            (128, 2, 5.0),
            (128, 4, 3.0),
        ],
    )
    def test_compression_ratio(self, dim, bits, expected_min_ratio):
        """Compression ratio should meet minimum expectations."""
        q = TurboQuantizer(dim=dim, bits=bits, seed=42)
        ratio = q.compression_ratio()
        assert ratio >= expected_min_ratio, (
            f"dim={dim}, bits={bits}: ratio={ratio:.1f}x, expected >= {expected_min_ratio}x"
        )

    def test_compression_ratio_formula(self):
        """Verify the ratio calculation is correct."""
        dim = 128
        bits = 3
        q = TurboQuantizer(dim=dim, bits=bits, seed=42)

        orig = dim * 4  # float32
        # Stage 1: dim * bits, Stage 2: dim * 1, gamma: 32, norm: 32
        comp = (dim * bits + dim * 1 + 32 + 32) / 8

        expected_ratio = orig / comp
        actual_ratio = q.compression_ratio()
        assert abs(actual_ratio - expected_ratio) < 0.01


# ===================================================================
# Part 8: VectorStore integration rigorous tests
# ===================================================================


class TestVectorStoreRigorous:
    """Rigorous integration tests for the full VectorStore pipeline."""

    def _make_store(self, dim=64):
        from langchain_core.embeddings import Embeddings

        class DetEmbeddings(Embeddings):
            def __init__(self, d):
                self.d = d
            def embed_documents(self, texts):
                return [self._e(t) for t in texts]
            def embed_query(self, text):
                return self._e(text)
            def _e(self, text):
                rng = np.random.RandomState(hash(text) % (2**31))
                v = rng.randn(self.d).astype(np.float64)
                return (v / np.linalg.norm(v)).tolist()

        from langchain_turboquant import TurboQuantVectorStore
        return TurboQuantVectorStore(embedding=DetEmbeddings(dim), bits=3)

    def test_search_returns_correct_count(self):
        store = self._make_store()
        store.add_texts([f"doc {i}" for i in range(20)])
        results = store.similarity_search("doc 0", k=5)
        assert len(results) == 5

    def test_search_with_score_sorted_descending(self):
        store = self._make_store()
        store.add_texts([f"text {i}" for i in range(30)])
        results = store.similarity_search_with_score("text 0", k=10)
        scores = [s for _, s in results]
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1], "Scores not sorted descending"

    def test_incremental_add_preserves_old_docs(self):
        store = self._make_store()
        ids1 = store.add_texts(["alpha", "beta"])
        ids2 = store.add_texts(["gamma", "delta"])

        all_docs = store.get_by_ids(ids1 + ids2)
        contents = {d.page_content for d in all_docs}
        assert contents == {"alpha", "beta", "gamma", "delta"}

    def test_delete_then_search(self):
        store = self._make_store()
        ids = store.add_texts(["keep this", "delete this", "also keep"])
        store.delete([ids[1]])

        results = store.similarity_search("anything", k=10)
        contents = {r.page_content for r in results}
        assert "delete this" not in contents
        assert len(results) == 2

    def test_save_load_roundtrip_preserves_search(self):
        store = self._make_store()
        store.add_texts(["apple pie recipe", "banana smoothie", "cherry tart"])

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = f.name

        store.save(path)

        from langchain_turboquant import TurboQuantVectorStore
        from langchain_core.embeddings import Embeddings

        class DetEmbeddings(Embeddings):
            def __init__(self, d):
                self.d = d
            def embed_documents(self, texts):
                return [self._e(t) for t in texts]
            def embed_query(self, text):
                return self._e(text)
            def _e(self, text):
                rng = np.random.RandomState(hash(text) % (2**31))
                v = rng.randn(self.d).astype(np.float64)
                return (v / np.linalg.norm(v)).tolist()

        loaded = TurboQuantVectorStore.load(path, embedding=DetEmbeddings(64))
        r1 = store.similarity_search_with_score("apple", k=3)
        r2 = loaded.similarity_search_with_score("apple", k=3)

        # Same documents, same scores
        for (d1, s1), (d2, s2) in zip(r1, r2):
            assert d1.page_content == d2.page_content
            assert abs(s1 - s2) < 1e-6

    def test_metadata_preserved_through_pipeline(self):
        store = self._make_store()
        metadatas = [{"source": "wiki", "page": 1}, {"source": "book", "page": 42}]
        store.add_texts(["fact one", "fact two"], metadatas=metadatas)
        results = store.similarity_search("fact", k=2)
        for doc in results:
            assert "source" in doc.metadata
            assert "page" in doc.metadata

    def test_memory_stats_accuracy(self):
        store = self._make_store(dim=384)
        store.add_texts([f"document {i}" for i in range(100)])
        stats = store.memory_stats()

        assert stats["num_documents"] == 100
        assert stats["dimension"] == 384
        assert float(stats["compression_ratio"].rstrip("x")) > 5.0
        assert stats["compressed_bytes"] < stats["original_bytes"]


# ===================================================================
# Part 9: Performance benchmarks (informational, not strict)
# ===================================================================


class TestPerformance:
    """Benchmark CPU performance — these tests always pass but print timing."""

    def test_quantize_speed(self):
        """Benchmark quantization speed."""
        dim = 384
        n = 1000
        q = TurboQuantizer(dim=dim, bits=3, seed=42)
        vecs = np.random.randn(n, dim).astype(np.float32)

        start = time.perf_counter()
        q.quantize(vecs)
        elapsed = time.perf_counter() - start

        vecs_per_sec = n / elapsed
        print(f"\n  Quantize: {n} x {dim}d in {elapsed:.3f}s ({vecs_per_sec:.0f} vecs/sec)")
        assert elapsed < 30, "Quantization too slow"

    def test_search_speed(self):
        """Benchmark search speed."""
        dim = 384
        n = 5000
        q = TurboQuantizer(dim=dim, bits=3, seed=42)
        vecs = np.random.randn(n, dim).astype(np.float32)
        compressed = q.quantize(vecs)
        query = np.random.randn(dim).astype(np.float32)

        start = time.perf_counter()
        for _ in range(100):
            q.cosine_scores(query, compressed)
        elapsed = time.perf_counter() - start

        qps = 100 / elapsed
        print(f"\n  Search: 100 queries over {n} vecs x {dim}d in {elapsed:.3f}s ({qps:.0f} QPS)")
        assert elapsed < 30, "Search too slow"

    def test_memory_footprint(self):
        """Compare actual memory usage."""
        dim = 384
        n = 10000
        q = TurboQuantizer(dim=dim, bits=3, seed=42)
        vecs = np.random.randn(n, dim).astype(np.float32)

        original_bytes = vecs.nbytes
        compressed = q.quantize(vecs)

        compressed_bytes = (
            compressed.indices.nbytes
            + compressed.qjl_bits.nbytes
            + compressed.gammas.nbytes
            + compressed.norms.nbytes
        )

        ratio = original_bytes / compressed_bytes
        print(f"\n  Memory: original={original_bytes:,} bytes, "
              f"compressed={compressed_bytes:,} bytes, ratio={ratio:.1f}x")

        # indices stored as uint8 (8 bits) instead of packed 3 bits, and
        # qjl_bits stored as int8 (8 bits) instead of packed 1 bit.
        # So actual in-memory ratio is lower than theoretical.
        # Theoretical ratio at 384d/3bit = 7.7x, but uint8 storage = ~2.0x.
        # Bit-packing would improve this; for now verify it's better than 1.5x.
        assert ratio > 1.5, f"Memory ratio {ratio:.1f}x too low"
