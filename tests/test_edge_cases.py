"""Edge-case tests for TurboQuantizer and TurboQuantVectorStore."""

import numpy as np
import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from langchain_turboquant.quantizer import TurboQuantizer, CompressedVectors
from langchain_turboquant.vectorstore import TurboQuantVectorStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class FakeEmbeddings(Embeddings):
    """Deterministic fake embeddings for testing."""

    def __init__(self, dim: int = 64):
        self.dim = dim

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        embeddings = []
        for text in texts:
            seed = hash(text) % (2**31)
            rng = np.random.RandomState(seed)
            embeddings.append(rng.randn(self.dim).tolist())
        return embeddings

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


# ===========================================================================
# Part 1: Quantizer edge cases (15+)
# ===========================================================================

class TestQuantizerEdgeCases:
    """Edge cases for TurboQuantizer.quantize / dequantize / scores."""

    # --- all-ones vector ---
    def test_all_ones_vector(self):
        dim = 16
        q = TurboQuantizer(dim=dim, bits=2, seed=0)
        vecs = np.ones((1, dim), dtype=np.float32)
        c = q.quantize(vecs)
        assert c.indices.shape == (1, dim)
        recon = q.dequantize(c)
        assert recon.shape == (1, dim)

    # --- one-hot vectors ---
    def test_one_hot_vectors(self):
        dim = 16
        q = TurboQuantizer(dim=dim, bits=2, seed=0)
        for i in range(dim):
            vec = np.zeros((1, dim), dtype=np.float32)
            vec[0, i] = 1.0
            c = q.quantize(vec)
            assert c.norms[0] == pytest.approx(1.0, abs=1e-5)

    # --- NaN handling ---
    def test_nan_vector_does_not_crash(self):
        """NaN input may produce NaN output but must not raise."""
        dim = 16
        q = TurboQuantizer(dim=dim, bits=2, seed=0)
        vecs = np.array([[np.nan] * dim], dtype=np.float32)
        c = q.quantize(vecs)
        # We don't require meaningful results, just no exception
        assert c.indices.shape == (1, dim)

    # --- Inf handling ---
    def test_inf_vector_does_not_crash(self):
        """Inf input may produce Inf/NaN output but must not raise."""
        dim = 16
        q = TurboQuantizer(dim=dim, bits=2, seed=0)
        vecs = np.array([[np.inf] * dim], dtype=np.float32)
        c = q.quantize(vecs)
        assert c.indices.shape == (1, dim)

    # --- large batch ---
    def test_large_batch(self):
        dim = 16
        q = TurboQuantizer(dim=dim, bits=2, seed=0)
        rng = np.random.RandomState(99)
        vecs = rng.randn(10000, dim).astype(np.float32)
        c = q.quantize(vecs)
        assert c.indices.shape == (10000, dim)
        assert c.norms.shape == (10000,)

    # --- single-vector batch ---
    def test_single_vector_batch(self):
        dim = 16
        q = TurboQuantizer(dim=dim, bits=2, seed=0)
        vecs = np.random.randn(1, dim).astype(np.float32)
        c = q.quantize(vecs)
        assert c.indices.shape == (1, dim)

    # --- 1D input (auto-expand) ---
    def test_1d_input_auto_expand(self):
        dim = 16
        q = TurboQuantizer(dim=dim, bits=2, seed=0)
        vec = np.random.randn(dim).astype(np.float32)  # 1-D
        c = q.quantize(vec)
        assert c.indices.shape == (1, dim)

    # --- dim=1 ---
    def test_dim_1(self):
        q = TurboQuantizer(dim=1, bits=2, seed=0)
        vecs = np.array([[3.0]], dtype=np.float32)
        c = q.quantize(vecs)
        recon = q.dequantize(c)
        assert recon.shape == (1, 1)

    # --- dim=2 ---
    def test_dim_2(self):
        q = TurboQuantizer(dim=2, bits=2, seed=0)
        vecs = np.array([[1.0, 2.0]], dtype=np.float32)
        c = q.quantize(vecs)
        recon = q.dequantize(c)
        assert recon.shape == (1, 2)

    # --- dim=3 (minimum non-trivial) ---
    def test_dim_3(self):
        q = TurboQuantizer(dim=3, bits=2, seed=0)
        vecs = np.array([[1.0, 0.0, -1.0]], dtype=np.float32)
        c = q.quantize(vecs)
        recon = q.dequantize(c)
        assert recon.shape == (1, 3)

    # --- float64 input ---
    def test_float64_input(self):
        dim = 16
        q = TurboQuantizer(dim=dim, bits=2, seed=0)
        vecs = np.random.randn(5, dim).astype(np.float64)
        c = q.quantize(vecs)
        assert c.indices.dtype == np.uint8

    # --- integer input ---
    def test_integer_input(self):
        dim = 16
        q = TurboQuantizer(dim=dim, bits=2, seed=0)
        vecs = np.ones((3, dim), dtype=np.int32)
        c = q.quantize(vecs)
        assert c.indices.shape == (3, dim)

    # --- empty array ---
    def test_empty_array(self):
        dim = 16
        q = TurboQuantizer(dim=dim, bits=2, seed=0)
        vecs = np.empty((0, dim), dtype=np.float32)
        c = q.quantize(vecs)
        assert c.indices.shape[0] == 0

    # --- all same direction ---
    def test_all_same_direction(self):
        dim = 16
        q = TurboQuantizer(dim=dim, bits=2, seed=0)
        base = np.random.randn(1, dim).astype(np.float32)
        # All vectors same direction, different magnitudes
        scales = np.array([0.1, 1.0, 10.0, 100.0])[:, None]
        vecs = base * scales
        c = q.quantize(vecs)
        recon = q.dequantize(c)
        assert recon.shape == (4, dim)

    # --- zero vector ---
    def test_zero_vector(self):
        dim = 16
        q = TurboQuantizer(dim=dim, bits=2, seed=0)
        vecs = np.zeros((1, dim), dtype=np.float32)
        c = q.quantize(vecs)
        assert c.norms[0] == pytest.approx(0.0, abs=1e-8)

    # --- asymmetric_scores with zero query ---
    def test_asymmetric_scores_zero_query(self):
        dim = 16
        q = TurboQuantizer(dim=dim, bits=2, seed=0)
        vecs = np.random.randn(5, dim).astype(np.float32)
        c = q.quantize(vecs)
        query = np.zeros(dim, dtype=np.float32)
        scores = q.asymmetric_scores(query, c)
        np.testing.assert_allclose(scores, 0.0, atol=1e-6)

    # --- cosine_scores returns values in approx [-1, 1] ---
    def test_cosine_scores_range(self):
        dim = 16
        q = TurboQuantizer(dim=dim, bits=3, seed=0)
        rng = np.random.RandomState(7)
        vecs = rng.randn(50, dim).astype(np.float32)
        c = q.quantize(vecs)
        query = rng.randn(dim).astype(np.float32)
        scores = q.cosine_scores(query, c)
        # Approximate cosine similarity should be roughly in [-1.5, 1.5]
        assert np.all(scores > -2.0) and np.all(scores < 2.0)


# ===========================================================================
# Part 2: VectorStore edge cases (15+)
# ===========================================================================

class TestVectorStoreEdgeCases:
    """Edge cases for TurboQuantVectorStore."""

    @pytest.fixture
    def emb(self):
        return FakeEmbeddings(dim=32)

    @pytest.fixture
    def store(self, emb):
        return TurboQuantVectorStore(embedding=emb, bits=2)

    # --- empty string text ---
    def test_empty_string_text(self, store):
        ids = store.add_texts([""])
        assert len(ids) == 1
        assert store._documents[0].page_content == ""

    # --- very long string ---
    def test_very_long_string(self, store):
        long_text = "x" * 10000
        ids = store.add_texts([long_text])
        assert len(ids) == 1
        assert store._documents[0].page_content == long_text

    # --- unicode / Korean text ---
    def test_unicode_korean_text(self, store):
        texts = ["안녕하세요", "こんにちは", "你好世界", "🎉🚀"]
        ids = store.add_texts(texts)
        assert len(ids) == 4
        results = store.similarity_search("안녕", k=2)
        assert len(results) == 2

    # --- special characters only ---
    def test_special_characters_text(self, store):
        ids = store.add_texts(["!@#$%^&*()_+-={}[]|\\:;\"'<>?,./~`"])
        assert len(ids) == 1

    # --- duplicate texts ---
    def test_duplicate_texts(self, store):
        ids = store.add_texts(["same", "same", "same"])
        assert len(ids) == 3
        assert len(store._documents) == 3
        # All should get different auto-generated IDs
        assert len(set(ids)) == 3

    # --- duplicate IDs ---
    def test_duplicate_ids(self, store):
        ids = store.add_texts(["a", "b"], ids=["dup", "dup"])
        assert len(ids) == 2
        # Both stored (store does not enforce uniqueness)
        assert len(store._documents) == 2

    # --- get_by_ids with nonexistent ID ---
    def test_get_by_ids_nonexistent(self, store):
        store.add_texts(["hello"], ids=["real"])
        docs = store.get_by_ids(["nonexistent"])
        assert docs == []

    # --- get_by_ids mixed existing and nonexistent ---
    def test_get_by_ids_mixed(self, store):
        store.add_texts(["hello", "world"], ids=["a", "b"])
        docs = store.get_by_ids(["a", "missing", "b"])
        assert len(docs) == 2

    # --- search with k=0 ---
    def test_search_k_zero(self, store):
        store.add_texts(["hello", "world"])
        results = store.similarity_search("hello", k=0)
        assert results == []

    # --- search with k=1 ---
    def test_search_k_one(self, store):
        store.add_texts(["hello", "world"])
        results = store.similarity_search("hello", k=1)
        assert len(results) == 1

    # --- 1 doc, k=10 ---
    def test_search_k_larger_than_docs(self, store):
        store.add_texts(["only one"])
        results = store.similarity_search("one", k=10)
        assert len(results) == 1

    # --- no metadata ---
    def test_add_texts_no_metadata(self, store):
        ids = store.add_texts(["hello"])
        doc = store._documents[0]
        # Only _id should be in metadata
        assert "_id" in doc.metadata

    # --- empty metadata ---
    def test_add_texts_empty_metadata(self, store):
        ids = store.add_texts(["hello"], metadatas=[{}])
        doc = store._documents[0]
        assert "_id" in doc.metadata

    # --- nested metadata ---
    def test_nested_metadata(self, store):
        meta = {"info": {"nested": {"deep": True}}, "tags": [1, 2, 3]}
        ids = store.add_texts(["hello"], metadatas=[meta])
        doc = store._documents[0]
        assert doc.metadata["info"]["nested"]["deep"] is True
        assert doc.metadata["tags"] == [1, 2, 3]

    # --- delete then re-add ---
    def test_delete_then_readd(self, store):
        ids = store.add_texts(["first", "second"])
        store.delete([ids[0]])
        assert len(store._documents) == 1

        new_ids = store.add_texts(["third"])
        assert len(store._documents) == 2
        results = store.similarity_search("third", k=1)
        assert len(results) == 1

    # --- delete with None ids ---
    def test_delete_none_ids(self, store):
        store.add_texts(["a"])
        result = store.delete(ids=None)
        assert result is False
        assert len(store._documents) == 1

    # --- similarity_search_by_vector empty store ---
    def test_search_by_vector_empty(self, store):
        vec = [0.0] * 32
        results = store.similarity_search_by_vector(vec, k=5)
        assert results == []

    # --- from_texts with metadata and ids ---
    def test_from_texts_with_all_options(self, emb):
        store = TurboQuantVectorStore.from_texts(
            ["alpha", "beta"],
            embedding=emb,
            metadatas=[{"k": 1}, {"k": 2}],
            ids=["id-a", "id-b"],
            bits=2,
        )
        assert len(store._documents) == 2
        docs = store.get_by_ids(["id-b"])
        assert len(docs) == 1
        assert docs[0].metadata["k"] == 2
