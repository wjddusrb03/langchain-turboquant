"""Extensive recall and accuracy tests for TurboQuant.

Covers:
1. Top-k recall at scale across various k, dataset sizes, dimensions, and bits
2. Search ordering correctness and determinism
3. Asymmetric estimator statistical validation
"""

from __future__ import annotations

import numpy as np
import pytest

from langchain_turboquant.quantizer import TurboQuantizer


# ===================================================================
# Helpers
# ===================================================================


def _exact_cosine_scores(vectors: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Compute exact cosine similarities between query and all vectors."""
    norms = np.linalg.norm(vectors, axis=1)
    q_norm = np.linalg.norm(query)
    return (vectors @ query) / (norms * q_norm + 1e-30)


def _recall_at_k(
    exact_scores: np.ndarray,
    approx_scores: np.ndarray,
    k: int,
) -> float:
    """Compute recall@k: fraction of true top-k found in approximate top-k."""
    k = min(k, len(exact_scores))
    exact_topk = set(np.argsort(exact_scores)[::-1][:k])
    approx_topk = set(np.argsort(approx_scores)[::-1][:k])
    return len(exact_topk & approx_topk) / k


def _avg_recall(
    vectors: np.ndarray,
    quantizer: TurboQuantizer,
    compressed,
    k: int,
    n_queries: int = 100,
    seed: int = 7777,
) -> float:
    """Average recall@k over n_queries random queries."""
    rng = np.random.RandomState(seed)
    recalls = []
    for _ in range(n_queries):
        query = rng.randn(vectors.shape[1]).astype(np.float32)
        exact = _exact_cosine_scores(vectors, query)
        approx = quantizer.cosine_scores(query, compressed)
        recalls.append(_recall_at_k(exact, approx, k))
    return float(np.mean(recalls))


# ===================================================================
# Part 1: Top-k Recall at scale (10+ tests)
# ===================================================================


class TestTopKRecallExtensive:
    """Large-scale recall tests across k, n, dim, and bits."""

    # -- 1.1  Various k values --
    @pytest.mark.parametrize("k", [1, 5, 10, 20, 50])
    def test_recall_at_various_k(self, k):
        """Recall should be reasonable across different k values."""
        dim, n = 64, 500
        rng = np.random.RandomState(42)
        vectors = rng.randn(n, dim).astype(np.float32)

        quantizer = TurboQuantizer(dim=dim, bits=3, seed=42)
        compressed = quantizer.quantize(vectors)

        recall = _avg_recall(vectors, quantizer, compressed, k=k, n_queries=50)
        # Larger k is generally easier; all should be above 0.4
        assert recall >= 0.4, f"k={k}: avg recall = {recall:.2%}"

    # -- 1.2  Various dataset sizes --
    @pytest.mark.parametrize("n", [50, 100, 500, 1000, 2000])
    def test_recall_at_various_n(self, n):
        """Recall should hold across dataset sizes."""
        dim = 64
        rng = np.random.RandomState(42)
        vectors = rng.randn(n, dim).astype(np.float32)

        quantizer = TurboQuantizer(dim=dim, bits=3, seed=42)
        compressed = quantizer.quantize(vectors)

        recall = _avg_recall(vectors, quantizer, compressed, k=10, n_queries=30)
        assert recall >= 0.3, f"n={n}: avg recall@10 = {recall:.2%}"

    # -- 1.3  Various dimensions --
    @pytest.mark.parametrize("dim", [16, 32, 64, 128, 256, 384, 512, 768])
    def test_recall_at_various_dim(self, dim):
        """Recall should be positive across dimensions."""
        n = min(200, max(50, dim * 2))  # scale n modestly with dim
        rng = np.random.RandomState(42)
        vectors = rng.randn(n, dim).astype(np.float32)

        quantizer = TurboQuantizer(dim=dim, bits=3, seed=42)
        compressed = quantizer.quantize(vectors)

        recall = _avg_recall(vectors, quantizer, compressed, k=10, n_queries=20)
        assert recall >= 0.2, f"dim={dim}: avg recall@10 = {recall:.2%}"

    # -- 1.4  Various bit widths --
    @pytest.mark.parametrize("bits", [1, 2, 3, 4])
    def test_recall_at_various_bits(self, bits):
        """Even 1-bit should produce some recall; more bits should be better."""
        dim, n = 64, 300
        rng = np.random.RandomState(42)
        vectors = rng.randn(n, dim).astype(np.float32)

        quantizer = TurboQuantizer(dim=dim, bits=bits, seed=42)
        compressed = quantizer.quantize(vectors)

        recall = _avg_recall(vectors, quantizer, compressed, k=10, n_queries=30)
        assert recall >= 0.1, f"bits={bits}: avg recall@10 = {recall:.2%}"

    # -- 1.5  Recall monotonically improves with bits (100 queries) --
    def test_recall_improves_with_bits_100_queries(self):
        """Recall averaged over 100 queries should generally increase with bits."""
        dim, n = 64, 300
        rng = np.random.RandomState(42)
        vectors = rng.randn(n, dim).astype(np.float32)

        recalls = {}
        for bits in [1, 2, 3, 4]:
            quantizer = TurboQuantizer(dim=dim, bits=bits, seed=42)
            compressed = quantizer.quantize(vectors)
            recalls[bits] = _avg_recall(
                vectors, quantizer, compressed, k=10, n_queries=100,
            )

        # Allow small slack: 4-bit >= 2-bit - 0.05, and 3-bit >= 1-bit - 0.05
        assert recalls[4] >= recalls[2] - 0.05, f"recalls: {recalls}"
        assert recalls[3] >= recalls[1] - 0.05, f"recalls: {recalls}"

    # -- 1.6  Large k relative to n --
    def test_recall_k_equals_n(self):
        """When k == n, recall should be exactly 1.0."""
        dim, n = 32, 50
        rng = np.random.RandomState(42)
        vectors = rng.randn(n, dim).astype(np.float32)
        query = rng.randn(dim).astype(np.float32)

        quantizer = TurboQuantizer(dim=dim, bits=3, seed=42)
        compressed = quantizer.quantize(vectors)

        exact = _exact_cosine_scores(vectors, query)
        approx = quantizer.cosine_scores(query, compressed)
        recall = _recall_at_k(exact, approx, k=n)
        assert recall == 1.0, f"k=n: recall should be 1.0, got {recall}"

    # -- 1.7  100-query average recall at moderate scale --
    def test_100_query_recall_moderate_scale(self):
        """100-query avg recall@10 on 500 vectors of dim=128."""
        dim, n = 128, 500
        rng = np.random.RandomState(42)
        vectors = rng.randn(n, dim).astype(np.float32)

        quantizer = TurboQuantizer(dim=dim, bits=3, seed=42)
        compressed = quantizer.quantize(vectors)

        recall = _avg_recall(vectors, quantizer, compressed, k=10, n_queries=100)
        assert recall >= 0.3, f"100-query avg recall@10 = {recall:.2%}"

    # -- 1.8  Higher k gives higher recall --
    def test_higher_k_gives_higher_or_equal_recall(self):
        """recall@50 >= recall@10 >= recall@1 on average."""
        dim, n = 64, 300
        rng = np.random.RandomState(42)
        vectors = rng.randn(n, dim).astype(np.float32)

        quantizer = TurboQuantizer(dim=dim, bits=3, seed=42)
        compressed = quantizer.quantize(vectors)

        r1 = _avg_recall(vectors, quantizer, compressed, k=1, n_queries=50)
        r10 = _avg_recall(vectors, quantizer, compressed, k=10, n_queries=50)
        r50 = _avg_recall(vectors, quantizer, compressed, k=50, n_queries=50)

        # Allow small slack for stochastic variation
        assert r50 >= r10 - 0.05, f"r50={r50:.2%}, r10={r10:.2%}"
        assert r10 >= r1 - 0.10, f"r10={r10:.2%}, r1={r1:.2%}"


# ===================================================================
# Part 2: Search ordering correctness (10+ tests)
# ===================================================================


class TestSearchOrdering:
    """Verify search result ordering and determinism."""

    # -- 2.1  Self-search ranks first --
    def test_self_search_ranks_first(self):
        """Searching a vector against a DB that contains it should rank it #1."""
        dim = 128
        rng = np.random.RandomState(42)
        vectors = rng.randn(100, dim).astype(np.float32)

        quantizer = TurboQuantizer(dim=dim, bits=3, seed=42)
        compressed = quantizer.quantize(vectors)

        hits = 0
        for idx in range(0, 100, 5):  # test 20 vectors
            scores = quantizer.cosine_scores(vectors[idx], compressed)
            if np.argmax(scores) == idx:
                hits += 1
        # At least 50% should rank themselves first
        assert hits >= 10, f"Self-search top-1 hits: {hits}/20"

    # -- 2.2  Cosine similarity ordering preserved for well-separated vectors --
    def test_cosine_order_preserved_well_separated(self):
        """For very similar vs very dissimilar vectors, order should be preserved."""
        dim = 128
        rng = np.random.RandomState(42)
        query = rng.randn(dim).astype(np.float32)

        # v_close is close to query, v_far is far
        v_close = query + rng.randn(dim).astype(np.float32) * 0.05
        v_far = -query + rng.randn(dim).astype(np.float32) * 0.05

        quantizer = TurboQuantizer(dim=dim, bits=3, seed=42)
        compressed = quantizer.quantize(np.stack([v_close, v_far]))

        scores = quantizer.cosine_scores(query, compressed)
        assert scores[0] > scores[1], (
            f"Close vector score {scores[0]:.4f} <= far vector score {scores[1]:.4f}"
        )

    # -- 2.3  Scores are descending after argsort --
    def test_scores_descending_after_sort(self):
        """Sorted scores should be in non-increasing order."""
        dim = 64
        n = 200
        rng = np.random.RandomState(42)
        vectors = rng.randn(n, dim).astype(np.float32)
        query = rng.randn(dim).astype(np.float32)

        quantizer = TurboQuantizer(dim=dim, bits=3, seed=42)
        compressed = quantizer.quantize(vectors)
        scores = quantizer.cosine_scores(query, compressed)

        sorted_scores = scores[np.argsort(scores)[::-1]]
        for i in range(len(sorted_scores) - 1):
            assert sorted_scores[i] >= sorted_scores[i + 1], (
                f"Score at position {i} ({sorted_scores[i]:.6f}) < "
                f"position {i+1} ({sorted_scores[i+1]:.6f})"
            )

    # -- 2.4  Deterministic results (same query, same result) --
    def test_deterministic_results(self):
        """Same query should always produce the same scores."""
        dim = 64
        rng = np.random.RandomState(42)
        vectors = rng.randn(100, dim).astype(np.float32)
        query = rng.randn(dim).astype(np.float32)

        quantizer = TurboQuantizer(dim=dim, bits=3, seed=42)
        compressed = quantizer.quantize(vectors)

        scores1 = quantizer.cosine_scores(query, compressed)
        scores2 = quantizer.cosine_scores(query, compressed)
        scores3 = quantizer.cosine_scores(query, compressed)

        np.testing.assert_array_equal(scores1, scores2)
        np.testing.assert_array_equal(scores2, scores3)

    # -- 2.5  Deterministic top-k ranking --
    def test_deterministic_topk_ranking(self):
        """Top-k indices should be identical across repeated calls."""
        dim = 64
        rng = np.random.RandomState(42)
        vectors = rng.randn(200, dim).astype(np.float32)
        query = rng.randn(dim).astype(np.float32)

        quantizer = TurboQuantizer(dim=dim, bits=3, seed=42)
        compressed = quantizer.quantize(vectors)

        topk1 = np.argsort(quantizer.cosine_scores(query, compressed))[::-1][:10]
        topk2 = np.argsort(quantizer.cosine_scores(query, compressed))[::-1][:10]
        np.testing.assert_array_equal(topk1, topk2)

    # -- 2.6  Close cluster vs far cluster --
    def test_close_cluster_vs_far_cluster(self):
        """Vectors near the query should score higher than a distant cluster."""
        dim = 64
        rng = np.random.RandomState(42)
        query = rng.randn(dim).astype(np.float32)
        query = query / np.linalg.norm(query)

        # Close cluster: query + small noise
        close_cluster = query[np.newaxis, :] + rng.randn(20, dim).astype(np.float32) * 0.1
        # Far cluster: -query + small noise
        far_cluster = -query[np.newaxis, :] + rng.randn(20, dim).astype(np.float32) * 0.1

        all_vectors = np.concatenate([close_cluster, far_cluster], axis=0)
        quantizer = TurboQuantizer(dim=dim, bits=3, seed=42)
        compressed = quantizer.quantize(all_vectors)

        scores = quantizer.cosine_scores(query, compressed)
        close_scores = scores[:20]
        far_scores = scores[20:]

        assert np.mean(close_scores) > np.mean(far_scores), (
            f"Close cluster mean {np.mean(close_scores):.4f} <= "
            f"far cluster mean {np.mean(far_scores):.4f}"
        )

    # -- 2.7  Random vs structured (clustered) vectors recall comparison --
    def test_random_vs_clustered_recall(self):
        """Clustered data should still produce reasonable recall."""
        dim = 64
        rng = np.random.RandomState(42)

        # Random vectors
        random_vecs = rng.randn(200, dim).astype(np.float32)

        # Clustered vectors: 4 clusters
        centers = rng.randn(4, dim).astype(np.float32)
        clustered_vecs = np.concatenate([
            c[np.newaxis, :] + rng.randn(50, dim).astype(np.float32) * 0.3
            for c in centers
        ], axis=0)

        for label, vecs in [("random", random_vecs), ("clustered", clustered_vecs)]:
            quantizer = TurboQuantizer(dim=dim, bits=3, seed=42)
            compressed = quantizer.quantize(vecs)
            recall = _avg_recall(vecs, quantizer, compressed, k=10, n_queries=30)
            assert recall >= 0.2, f"{label}: avg recall@10 = {recall:.2%}"

    # -- 2.8  Top-1 accuracy: similar vector beats random --
    def test_top1_similar_beats_random(self):
        """A vector very close to query should beat random vectors in top-1."""
        dim = 128
        rng = np.random.RandomState(42)
        query = rng.randn(dim).astype(np.float32)
        near = query + rng.randn(dim).astype(np.float32) * 0.01  # very close
        others = rng.randn(99, dim).astype(np.float32)

        all_vecs = np.concatenate([near[np.newaxis, :], others], axis=0)
        quantizer = TurboQuantizer(dim=dim, bits=3, seed=42)
        compressed = quantizer.quantize(all_vecs)

        scores = quantizer.cosine_scores(query, compressed)
        assert np.argmax(scores) == 0, "Nearest vector should rank first"

    # -- 2.9  Negative scores for opposite vectors --
    def test_opposite_vectors_have_negative_scores(self):
        """Vectors opposite to query should have negative cosine scores."""
        dim = 64
        rng = np.random.RandomState(42)
        query = rng.randn(dim).astype(np.float32)
        opposite = -query

        quantizer = TurboQuantizer(dim=dim, bits=3, seed=42)
        compressed = quantizer.quantize(opposite[np.newaxis, :])
        score = quantizer.cosine_scores(query, compressed)[0]
        assert score < -0.5, f"Opposite vector score = {score:.4f}, expected < -0.5"

    # -- 2.10  Multiple clusters ranking --
    def test_multiple_cluster_ranking(self):
        """Query belonging to cluster A should rank cluster A vectors higher."""
        dim = 64
        rng = np.random.RandomState(42)

        center_a = rng.randn(dim).astype(np.float32)
        center_b = -center_a  # opposite direction
        cluster_a = center_a[np.newaxis, :] + rng.randn(30, dim).astype(np.float32) * 0.15
        cluster_b = center_b[np.newaxis, :] + rng.randn(30, dim).astype(np.float32) * 0.15

        all_vecs = np.concatenate([cluster_a, cluster_b], axis=0)
        query = center_a + rng.randn(dim).astype(np.float32) * 0.05  # close to A

        quantizer = TurboQuantizer(dim=dim, bits=3, seed=42)
        compressed = quantizer.quantize(all_vecs)
        scores = quantizer.cosine_scores(query, compressed)

        top10 = set(np.argsort(scores)[::-1][:10])
        cluster_a_in_top10 = len(top10 & set(range(30)))
        assert cluster_a_in_top10 >= 7, (
            f"Only {cluster_a_in_top10}/10 from cluster A in top-10"
        )

    # -- 2.11  Asymmetric scores vs full dequantize ranking agreement --
    def test_asymmetric_vs_dequantize_ranking_agree(self):
        """Asymmetric scores and full-dequantize should mostly agree on top-k."""
        dim = 64
        n = 100
        rng = np.random.RandomState(42)
        vectors = rng.randn(n, dim).astype(np.float32)
        query = rng.randn(dim).astype(np.float32)

        quantizer = TurboQuantizer(dim=dim, bits=3, seed=42)
        compressed = quantizer.quantize(vectors)

        # Asymmetric scores
        asym_scores = quantizer.cosine_scores(query, compressed)
        asym_top10 = set(np.argsort(asym_scores)[::-1][:10])

        # Full dequantize scores
        recon = quantizer.dequantize(compressed)
        recon_scores = _exact_cosine_scores(recon, query)
        recon_top10 = set(np.argsort(recon_scores)[::-1][:10])

        overlap = len(asym_top10 & recon_top10)
        assert overlap >= 5, f"Asymmetric vs dequantize top-10 overlap: {overlap}/10"


# ===================================================================
# Part 3: Asymmetric estimator statistical validation (5+ tests)
# ===================================================================


class TestAsymmetricEstimatorStats:
    """Statistical validation of the asymmetric inner-product estimator."""

    # -- 3.1  Mean of estimates over 100 seeds converges to true value --
    def test_mean_converges_over_100_seeds(self):
        """Average inner-product estimate over 100 random seeds should
        converge to the true inner product."""
        dim = 64
        rng = np.random.RandomState(42)
        x = rng.randn(dim).astype(np.float32)
        q = rng.randn(dim).astype(np.float32)
        true_ip = float(np.dot(q, x))

        estimates = []
        for seed in range(100):
            quantizer = TurboQuantizer(dim=dim, bits=3, seed=seed)
            compressed = quantizer.quantize(x)
            est = float(quantizer.asymmetric_scores(q, compressed)[0])
            estimates.append(est)

        mean_est = np.mean(estimates)
        relative_error = abs(mean_est - true_ip) / (abs(true_ip) + 1e-10)
        assert relative_error < 0.25, (
            f"Mean estimate {mean_est:.4f} vs true {true_ip:.4f}, "
            f"relative error {relative_error:.2%}"
        )

    # -- 3.2  Variance is bounded --
    def test_variance_bounded(self):
        """Variance of the estimator should be bounded relative to signal."""
        dim = 64
        rng = np.random.RandomState(42)
        x = rng.randn(dim).astype(np.float32)
        q = rng.randn(dim).astype(np.float32)
        true_ip = float(np.dot(q, x))

        estimates = []
        for seed in range(100):
            quantizer = TurboQuantizer(dim=dim, bits=3, seed=seed)
            compressed = quantizer.quantize(x)
            est = float(quantizer.asymmetric_scores(q, compressed)[0])
            estimates.append(est)

        variance = np.var(estimates)
        # Variance should be less than the squared signal magnitude
        # (otherwise the estimator is useless)
        signal_sq = true_ip ** 2 + 1e-10
        assert variance < signal_sq * 5.0, (
            f"Variance {variance:.4f} exceeds 5x signal^2 {signal_sq:.4f}"
        )

    # -- 3.3  Correlation between true and estimated inner products >= 0.9 --
    def test_correlation_above_threshold(self):
        """Pearson correlation between true and estimated IPs should be >= 0.9."""
        dim = 128
        n = 200
        rng = np.random.RandomState(42)
        vectors = rng.randn(n, dim).astype(np.float32)
        query = rng.randn(dim).astype(np.float32)

        true_ips = vectors @ query
        quantizer = TurboQuantizer(dim=dim, bits=3, seed=42)
        compressed = quantizer.quantize(vectors)
        est_ips = quantizer.asymmetric_scores(query, compressed)

        corr = np.corrcoef(true_ips, est_ips)[0, 1]
        assert corr >= 0.9, f"Correlation = {corr:.4f}, expected >= 0.9"

    # -- 3.4  Correlation for cosine scores --
    def test_cosine_correlation_above_threshold(self):
        """Pearson correlation for cosine scores should be >= 0.9."""
        dim = 128
        n = 200
        rng = np.random.RandomState(42)
        vectors = rng.randn(n, dim).astype(np.float32)
        query = rng.randn(dim).astype(np.float32)

        true_cosines = _exact_cosine_scores(vectors, query)
        quantizer = TurboQuantizer(dim=dim, bits=3, seed=42)
        compressed = quantizer.quantize(vectors)
        est_cosines = quantizer.cosine_scores(query, compressed)

        corr = np.corrcoef(true_cosines, est_cosines)[0, 1]
        assert corr >= 0.9, f"Cosine correlation = {corr:.4f}, expected >= 0.9"

    # -- 3.5  Higher bits give lower estimation error --
    def test_higher_bits_lower_estimation_error(self):
        """Estimation MSE should decrease with more bits."""
        dim = 128
        n = 200
        rng = np.random.RandomState(42)
        vectors = rng.randn(n, dim).astype(np.float32)
        query = rng.randn(dim).astype(np.float32)
        true_ips = vectors @ query

        errors = {}
        for bits in [1, 2, 3, 4]:
            quantizer = TurboQuantizer(dim=dim, bits=bits, seed=42)
            compressed = quantizer.quantize(vectors)
            est_ips = quantizer.asymmetric_scores(query, compressed)
            errors[bits] = float(np.mean((true_ips - est_ips) ** 2))

        # 4-bit should have lower error than 1-bit
        assert errors[4] < errors[1], (
            f"4-bit error {errors[4]:.4f} >= 1-bit error {errors[1]:.4f}"
        )

    # -- 3.6  Estimator on unit vectors: mean error is small --
    def test_unit_vector_mean_error_small(self):
        """For unit vectors, mean absolute error of cosine estimate should be small."""
        dim = 128
        n = 300
        rng = np.random.RandomState(42)
        vectors = rng.randn(n, dim).astype(np.float32)
        vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        query = rng.randn(dim).astype(np.float32)
        query = query / np.linalg.norm(query)

        true_cosines = _exact_cosine_scores(vectors, query)
        quantizer = TurboQuantizer(dim=dim, bits=3, seed=42)
        compressed = quantizer.quantize(vectors)
        est_cosines = quantizer.cosine_scores(query, compressed)

        mae = float(np.mean(np.abs(true_cosines - est_cosines)))
        assert mae < 0.15, f"Mean absolute cosine error = {mae:.4f}, expected < 0.15"

    # -- 3.7  Multiple query average: estimator bias is small --
    def test_multi_query_estimator_bias(self):
        """Across 50 queries, the average bias should be near zero."""
        dim = 64
        n = 100
        rng = np.random.RandomState(42)
        vectors = rng.randn(n, dim).astype(np.float32)

        quantizer = TurboQuantizer(dim=dim, bits=3, seed=42)
        compressed = quantizer.quantize(vectors)

        biases = []
        for _ in range(50):
            query = rng.randn(dim).astype(np.float32)
            true_ips = vectors @ query
            est_ips = quantizer.asymmetric_scores(query, compressed)
            # Mean signed error per query
            biases.append(float(np.mean(est_ips - true_ips)))

        mean_bias = np.mean(biases)
        # Average bias across queries should be small relative to typical IP magnitude
        assert abs(mean_bias) < 5.0, f"Mean bias = {mean_bias:.4f}"
