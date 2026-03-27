"""Tests for the TurboQuantVectorStore LangChain integration."""

import tempfile

import numpy as np
import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from langchain_turboquant.vectorstore import TurboQuantVectorStore


# ---------------------------------------------------------------------------
# Fake deterministic embeddings (no API key needed)
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


# ---------------------------------------------------------------------------
# VectorStore tests
# ---------------------------------------------------------------------------

class TestTurboQuantVectorStore:
    @pytest.fixture
    def embedding(self):
        return FakeEmbeddings(dim=64)

    @pytest.fixture
    def store(self, embedding):
        return TurboQuantVectorStore(embedding=embedding, bits=3)

    @pytest.fixture
    def populated_store(self, embedding):
        texts = [
            "The cat sat on the mat",
            "Dogs are loyal companions",
            "Python is a programming language",
            "Machine learning is transforming AI",
            "The weather is sunny today",
        ]
        return TurboQuantVectorStore.from_texts(texts, embedding=embedding, bits=3)

    def test_add_texts(self, store):
        ids = store.add_texts(["hello", "world"])
        assert len(ids) == 2
        assert store._compressed is not None
        assert len(store._documents) == 2

    def test_add_texts_empty(self, store):
        ids = store.add_texts([])
        assert ids == []

    def test_add_texts_with_metadata(self, store):
        metadatas = [{"source": "test1"}, {"source": "test2"}]
        ids = store.add_texts(["hello", "world"], metadatas=metadatas)
        assert len(ids) == 2
        assert store._documents[0].metadata["source"] == "test1"

    def test_add_texts_with_ids(self, store):
        ids = store.add_texts(["hello", "world"], ids=["id1", "id2"])
        assert ids == ["id1", "id2"]

    def test_add_texts_incremental(self, store):
        store.add_texts(["first"])
        store.add_texts(["second", "third"])
        assert len(store._documents) == 3
        assert store._compressed.indices.shape[0] == 3

    def test_similarity_search(self, populated_store):
        results = populated_store.similarity_search("cat", k=2)
        assert len(results) == 2
        assert all(isinstance(r, Document) for r in results)

    def test_similarity_search_with_score(self, populated_store):
        results = populated_store.similarity_search_with_score("programming", k=3)
        assert len(results) == 3
        assert all(isinstance(r, tuple) for r in results)
        assert all(isinstance(r[0], Document) for r in results)
        assert all(isinstance(r[1], float) for r in results)
        # Scores should be sorted descending
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)

    def test_similarity_search_empty_store(self, store):
        results = store.similarity_search("hello", k=2)
        assert results == []

    def test_similarity_search_k_larger_than_store(self, populated_store):
        results = populated_store.similarity_search("test", k=100)
        assert len(results) == 5  # only 5 docs in store

    def test_from_texts(self, embedding):
        store = TurboQuantVectorStore.from_texts(
            ["hello", "world"],
            embedding=embedding,
            bits=3,
        )
        assert len(store._documents) == 2

    def test_similarity_search_by_vector(self, populated_store, embedding):
        vec = embedding.embed_query("cat")
        results = populated_store.similarity_search_by_vector(vec, k=2)
        assert len(results) == 2

    def test_delete(self, store):
        ids = store.add_texts(["a", "b", "c"])
        result = store.delete([ids[1]])
        assert result is True
        assert len(store._documents) == 2
        assert store._compressed.indices.shape[0] == 2

    def test_delete_nonexistent(self, store):
        store.add_texts(["a", "b"])
        result = store.delete(["nonexistent"])
        assert result is False

    def test_delete_all(self, store):
        ids = store.add_texts(["a", "b"])
        result = store.delete(ids)
        assert result is True
        assert len(store._documents) == 0
        assert store._compressed is None

    def test_get_by_ids(self, store):
        ids = store.add_texts(["alpha", "beta", "gamma"])
        docs = store.get_by_ids([ids[0], ids[2]])
        assert len(docs) == 2
        contents = {d.page_content for d in docs}
        assert "alpha" in contents
        assert "gamma" in contents

    def test_memory_stats(self, populated_store):
        stats = populated_store.memory_stats()
        assert stats["num_documents"] == 5
        assert stats["dimension"] == 64
        assert "compression_ratio" in stats
        assert "memory_saved_pct" in stats

    def test_memory_stats_empty(self, store):
        stats = store.memory_stats()
        assert stats["num_documents"] == 0

    def test_save_and_load(self, populated_store, embedding):
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = f.name

        populated_store.save(path)
        loaded = TurboQuantVectorStore.load(path, embedding=embedding)

        assert len(loaded._documents) == 5
        results = loaded.similarity_search("cat", k=2)
        assert len(results) == 2

    def test_as_retriever(self, populated_store):
        retriever = populated_store.as_retriever(search_kwargs={"k": 2})
        assert retriever is not None
