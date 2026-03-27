"""Persistence, serialization, and state-consistency tests for TurboQuantVectorStore."""

import os
import pickle
import tempfile

import numpy as np
import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from langchain_turboquant.quantizer import CompressedVectors, TurboQuantizer
from langchain_turboquant.vectorstore import TurboQuantVectorStore


# ---------------------------------------------------------------------------
# Deterministic fake embeddings (no API key needed)
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
# Helpers
# ---------------------------------------------------------------------------

def _tmp_path(suffix=".pkl"):
    """Return a temp file path (caller must clean up)."""
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    return path


def _make_texts(n: int) -> list[str]:
    """Generate *n* unique deterministic text strings."""
    return [f"document number {i}" for i in range(n)]


# ===========================================================================
# 1. Save / Load rigorous tests (>=10)
# ===========================================================================

class TestSaveLoad:
    """Save/Load round-trip tests."""

    @pytest.fixture
    def emb(self):
        return FakeEmbeddings(dim=64)

    # -- 1.1 empty store --
    def test_save_load_empty(self, emb):
        store = TurboQuantVectorStore(embedding=emb, bits=3)
        path = _tmp_path()
        try:
            store.save(path)
            loaded = TurboQuantVectorStore.load(path, embedding=emb)
            assert len(loaded._documents) == 0
            assert loaded._compressed is None
        finally:
            os.unlink(path)

    # -- 1.2 single document --
    def test_save_load_single_doc(self, emb):
        store = TurboQuantVectorStore(embedding=emb, bits=3)
        store.add_texts(["hello world"])
        path = _tmp_path()
        try:
            store.save(path)
            loaded = TurboQuantVectorStore.load(path, embedding=emb)
            assert len(loaded._documents) == 1
            assert loaded._documents[0].page_content == "hello world"
        finally:
            os.unlink(path)

    # -- 1.3 100 documents --
    def test_save_load_100_docs(self, emb):
        texts = _make_texts(100)
        store = TurboQuantVectorStore.from_texts(texts, embedding=emb, bits=3)
        path = _tmp_path()
        try:
            store.save(path)
            loaded = TurboQuantVectorStore.load(path, embedding=emb)
            assert len(loaded._documents) == 100
        finally:
            os.unlink(path)

    # -- 1.4 search results identical after round-trip --
    def test_save_load_search_results_match(self, emb):
        texts = _make_texts(20)
        store = TurboQuantVectorStore.from_texts(texts, embedding=emb, bits=3)
        original_results = store.similarity_search("document number 5", k=5)

        path = _tmp_path()
        try:
            store.save(path)
            loaded = TurboQuantVectorStore.load(path, embedding=emb)
            loaded_results = loaded.similarity_search("document number 5", k=5)

            assert len(original_results) == len(loaded_results)
            for orig, load in zip(original_results, loaded_results):
                assert orig.page_content == load.page_content
        finally:
            os.unlink(path)

    # -- 1.5 scores are exactly equal after round-trip --
    def test_save_load_scores_exact(self, emb):
        texts = _make_texts(10)
        store = TurboQuantVectorStore.from_texts(texts, embedding=emb, bits=3)
        orig_results = store.similarity_search_with_score("query text", k=10)

        path = _tmp_path()
        try:
            store.save(path)
            loaded = TurboQuantVectorStore.load(path, embedding=emb)
            loaded_results = loaded.similarity_search_with_score("query text", k=10)

            for (_, orig_score), (_, loaded_score) in zip(orig_results, loaded_results):
                assert orig_score == loaded_score, (
                    f"Scores differ: {orig_score} vs {loaded_score}"
                )
        finally:
            os.unlink(path)

    # -- 1.6 metadata preserved --
    def test_save_load_metadata_preserved(self, emb):
        metas = [{"source": "a", "page": 1}, {"source": "b", "page": 2}]
        store = TurboQuantVectorStore(embedding=emb, bits=3)
        store.add_texts(["doc1", "doc2"], metadatas=metas)
        path = _tmp_path()
        try:
            store.save(path)
            loaded = TurboQuantVectorStore.load(path, embedding=emb)
            for orig, loaded_doc in zip(store._documents, loaded._documents):
                assert orig.metadata["source"] == loaded_doc.metadata["source"]
                assert orig.metadata["page"] == loaded_doc.metadata["page"]
        finally:
            os.unlink(path)

    # -- 1.7 IDs preserved --
    def test_save_load_ids_preserved(self, emb):
        store = TurboQuantVectorStore(embedding=emb, bits=3)
        ids = store.add_texts(["a", "b", "c"], ids=["id-a", "id-b", "id-c"])
        path = _tmp_path()
        try:
            store.save(path)
            loaded = TurboQuantVectorStore.load(path, embedding=emb)
            assert loaded._ids == ids
        finally:
            os.unlink(path)

    # -- 1.8 add_texts works after load --
    def test_add_texts_after_load(self, emb):
        store = TurboQuantVectorStore.from_texts(["alpha"], embedding=emb, bits=3)
        path = _tmp_path()
        try:
            store.save(path)
            loaded = TurboQuantVectorStore.load(path, embedding=emb)
            loaded.add_texts(["beta", "gamma"])
            assert len(loaded._documents) == 3
            assert loaded._compressed.indices.shape[0] == 3
        finally:
            os.unlink(path)

    # -- 1.9 delete works after load --
    def test_delete_after_load(self, emb):
        store = TurboQuantVectorStore(embedding=emb, bits=3)
        ids = store.add_texts(["x", "y", "z"], ids=["1", "2", "3"])
        path = _tmp_path()
        try:
            store.save(path)
            loaded = TurboQuantVectorStore.load(path, embedding=emb)
            loaded.delete(["2"])
            assert len(loaded._documents) == 2
            assert "2" not in loaded._ids
        finally:
            os.unlink(path)

    # -- 1.10 load non-existent file raises error --
    def test_load_nonexistent_file(self, emb):
        with pytest.raises(FileNotFoundError):
            TurboQuantVectorStore.load("/tmp/does_not_exist_abc123.pkl", embedding=emb)

    # -- 1.11 load invalid file raises error --
    def test_load_invalid_format(self, emb):
        path = _tmp_path(suffix=".pkl")
        try:
            with open(path, "wb") as f:
                f.write(b"this is not a pickle file")
            with pytest.raises(Exception):
                TurboQuantVectorStore.load(path, embedding=emb)
        finally:
            os.unlink(path)


# ===========================================================================
# 2. CompressedVectors serialization (>=5)
# ===========================================================================

class TestCompressedVectorsSerialization:
    """Verify CompressedVectors can be serialized / deserialized."""

    @pytest.fixture
    def sample_cv(self):
        """Create a small CompressedVectors for testing."""
        rng = np.random.RandomState(0)
        n, dim, qjl_dim = 10, 64, 64
        return CompressedVectors(
            indices=rng.randint(0, 8, size=(n, dim)).astype(np.uint8),
            qjl_bits=rng.choice([-1, 1], size=(n, qjl_dim)).astype(np.int8),
            gammas=rng.rand(n).astype(np.float32),
            norms=rng.rand(n).astype(np.float32) + 0.5,
        )

    # -- 2.1 pickle round-trip --
    def test_pickle_roundtrip(self, sample_cv):
        data = pickle.dumps(sample_cv)
        loaded = pickle.loads(data)
        np.testing.assert_array_equal(loaded.indices, sample_cv.indices)
        np.testing.assert_array_equal(loaded.qjl_bits, sample_cv.qjl_bits)
        np.testing.assert_array_equal(loaded.gammas, sample_cv.gammas)
        np.testing.assert_array_equal(loaded.norms, sample_cv.norms)

    # -- 2.2 pickle to file --
    def test_pickle_file_roundtrip(self, sample_cv):
        path = _tmp_path()
        try:
            with open(path, "wb") as f:
                pickle.dump(sample_cv, f)
            with open(path, "rb") as f:
                loaded = pickle.load(f)
            np.testing.assert_array_equal(loaded.indices, sample_cv.indices)
            np.testing.assert_array_equal(loaded.norms, sample_cv.norms)
        finally:
            os.unlink(path)

    # -- 2.3 numpy savez round-trip --
    def test_numpy_savez_roundtrip(self, sample_cv):
        path = _tmp_path(suffix=".npz")
        try:
            np.savez(
                path,
                indices=sample_cv.indices,
                qjl_bits=sample_cv.qjl_bits,
                gammas=sample_cv.gammas,
                norms=sample_cv.norms,
            )
            data = np.load(path)
            loaded = CompressedVectors(
                indices=data["indices"].copy(),
                qjl_bits=data["qjl_bits"].copy(),
                gammas=data["gammas"].copy(),
                norms=data["norms"].copy(),
            )
            data.close()
            np.testing.assert_array_equal(loaded.indices, sample_cv.indices)
            np.testing.assert_array_equal(loaded.qjl_bits, sample_cv.qjl_bits)
            np.testing.assert_array_equal(loaded.gammas, sample_cv.gammas)
            np.testing.assert_array_equal(loaded.norms, sample_cv.norms)
        finally:
            os.unlink(path)

    # -- 2.4 data integrity: dtypes preserved --
    def test_serialization_dtypes(self, sample_cv):
        data = pickle.dumps(sample_cv)
        loaded = pickle.loads(data)
        assert loaded.indices.dtype == np.uint8
        assert loaded.qjl_bits.dtype == np.int8
        assert loaded.gammas.dtype == np.float32
        assert loaded.norms.dtype == np.float32

    # -- 2.5 data integrity: shapes preserved --
    def test_serialization_shapes(self, sample_cv):
        data = pickle.dumps(sample_cv)
        loaded = pickle.loads(data)
        assert loaded.indices.shape == sample_cv.indices.shape
        assert loaded.qjl_bits.shape == sample_cv.qjl_bits.shape
        assert loaded.gammas.shape == sample_cv.gammas.shape
        assert loaded.norms.shape == sample_cv.norms.shape

    # -- 2.6 numpy savez compressed --
    def test_numpy_savez_compressed(self, sample_cv):
        path = _tmp_path(suffix=".npz")
        try:
            np.savez_compressed(
                path,
                indices=sample_cv.indices,
                qjl_bits=sample_cv.qjl_bits,
                gammas=sample_cv.gammas,
                norms=sample_cv.norms,
            )
            data = np.load(path)
            np.testing.assert_array_equal(data["indices"], sample_cv.indices)
            np.testing.assert_array_equal(data["qjl_bits"], sample_cv.qjl_bits)
            np.testing.assert_array_equal(data["gammas"], sample_cv.gammas)
            np.testing.assert_array_equal(data["norms"], sample_cv.norms)
            data.close()
        finally:
            os.unlink(path)


# ===========================================================================
# 3. Memory / state consistency (>=10)
# ===========================================================================

class TestStateConsistency:
    """State consistency under add / delete / save / load sequences."""

    @pytest.fixture
    def emb(self):
        return FakeEmbeddings(dim=64)

    # -- 3.1 add -> delete -> add cycle --
    def test_add_delete_add_cycle(self, emb):
        store = TurboQuantVectorStore(embedding=emb, bits=3)

        ids1 = store.add_texts(["a", "b", "c"])
        assert len(store._documents) == 3

        store.delete([ids1[1]])
        assert len(store._documents) == 2

        ids2 = store.add_texts(["d", "e"])
        assert len(store._documents) == 4
        assert store._compressed.indices.shape[0] == 4

        # Verify search still works
        results = store.similarity_search("a", k=4)
        assert len(results) == 4

    # -- 3.2 repeated add/delete cycles --
    def test_repeated_add_delete_cycles(self, emb):
        store = TurboQuantVectorStore(embedding=emb, bits=3)

        for cycle in range(5):
            ids = store.add_texts([f"cycle-{cycle}-doc-{j}" for j in range(3)])
            store.delete([ids[0]])

        # 5 cycles * 3 docs - 5 deleted = 10
        assert len(store._documents) == 10
        assert store._compressed.indices.shape[0] == 10

    # -- 3.3 bulk add then bulk delete --
    def test_bulk_add_bulk_delete(self, emb):
        texts = _make_texts(50)
        store = TurboQuantVectorStore(embedding=emb, bits=3)
        ids = store.add_texts(texts)

        to_delete = ids[:25]
        store.delete(to_delete)

        assert len(store._documents) == 25
        remaining_contents = {d.page_content for d in store._documents}
        for i in range(25, 50):
            assert texts[i] in remaining_contents

    # -- 3.4 compressed array size shrinks after delete --
    def test_compressed_shrinks_after_delete(self, emb):
        store = TurboQuantVectorStore(embedding=emb, bits=3)
        ids = store.add_texts(_make_texts(10))
        assert store._compressed.indices.shape[0] == 10

        store.delete(ids[:5])
        assert store._compressed.indices.shape[0] == 5
        assert store._compressed.qjl_bits.shape[0] == 5
        assert store._compressed.gammas.shape[0] == 5
        assert store._compressed.norms.shape[0] == 5

    # -- 3.5 multiple save/load cycles produce identical data --
    def test_repeated_save_load(self, emb):
        texts = _make_texts(10)
        store = TurboQuantVectorStore.from_texts(texts, embedding=emb, bits=3)
        original_scores = store.similarity_search_with_score("query", k=10)

        path = _tmp_path()
        try:
            for _ in range(5):
                store.save(path)
                store = TurboQuantVectorStore.load(path, embedding=emb)

            reloaded_scores = store.similarity_search_with_score("query", k=10)
            assert len(original_scores) == len(reloaded_scores)
            for (_, s1), (_, s2) in zip(original_scores, reloaded_scores):
                assert s1 == s2
        finally:
            os.unlink(path)

    # -- 3.6 memory_stats after add --
    def test_memory_stats_after_add(self, emb):
        store = TurboQuantVectorStore(embedding=emb, bits=3)

        stats0 = store.memory_stats()
        assert stats0["num_documents"] == 0

        store.add_texts(_make_texts(5))
        stats1 = store.memory_stats()
        assert stats1["num_documents"] == 5

        store.add_texts(_make_texts(3))
        stats2 = store.memory_stats()
        assert stats2["num_documents"] == 8

    # -- 3.7 memory_stats after delete --
    def test_memory_stats_after_delete(self, emb):
        store = TurboQuantVectorStore(embedding=emb, bits=3)
        ids = store.add_texts(_make_texts(10))
        assert store.memory_stats()["num_documents"] == 10

        store.delete(ids[:3])
        assert store.memory_stats()["num_documents"] == 7

    # -- 3.8 from_texts -> save -> load -> similarity_search pipeline --
    def test_full_pipeline(self, emb):
        texts = [
            "quantum computing basics",
            "classical mechanics overview",
            "deep learning tutorial",
            "organic chemistry fundamentals",
            "linear algebra refresher",
        ]
        store = TurboQuantVectorStore.from_texts(texts, embedding=emb, bits=3)
        path = _tmp_path()
        try:
            store.save(path)
            loaded = TurboQuantVectorStore.load(path, embedding=emb)
            results = loaded.similarity_search("machine learning", k=3)
            assert len(results) == 3
            assert all(isinstance(r, Document) for r in results)
        finally:
            os.unlink(path)

    # -- 3.9 different bits settings are independent --
    def test_different_bits_independent(self, emb):
        texts = _make_texts(10)

        store_2bit = TurboQuantVectorStore.from_texts(texts, embedding=emb, bits=2)
        store_4bit = TurboQuantVectorStore.from_texts(texts, embedding=emb, bits=4)

        scores_2 = store_2bit.similarity_search_with_score("query", k=5)
        scores_4 = store_4bit.similarity_search_with_score("query", k=5)

        # Different bit configurations should typically produce different scores
        # (very unlikely to be all equal)
        any_different = any(
            s2 != s4 for (_, s2), (_, s4) in zip(scores_2, scores_4)
        )
        assert any_different, "2-bit and 4-bit stores should produce different scores"

    # -- 3.10 delete all then re-add --
    def test_delete_all_then_readd(self, emb):
        store = TurboQuantVectorStore(embedding=emb, bits=3)
        ids = store.add_texts(["a", "b"])
        store.delete(ids)
        assert len(store._documents) == 0
        assert store._compressed is None

        # Re-add should work from scratch
        new_ids = store.add_texts(["c", "d", "e"])
        assert len(store._documents) == 3
        assert store._compressed.indices.shape[0] == 3
        results = store.similarity_search("c", k=1)
        assert len(results) == 1

    # -- 3.11 compressed bytes stats change with add/delete --
    def test_memory_stats_bytes_change(self, emb):
        store = TurboQuantVectorStore(embedding=emb, bits=3)
        store.add_texts(_make_texts(10))
        stats_10 = store.memory_stats()

        store.add_texts(_make_texts(10))
        stats_20 = store.memory_stats()

        assert stats_20["compressed_bytes"] > stats_10["compressed_bytes"]
        assert stats_20["original_bytes"] > stats_10["original_bytes"]

    # -- 3.12 ids remain consistent through add/delete --
    def test_ids_consistent_through_operations(self, emb):
        store = TurboQuantVectorStore(embedding=emb, bits=3)
        ids1 = store.add_texts(["a", "b", "c"], ids=["id-a", "id-b", "id-c"])
        store.delete(["id-b"])
        ids2 = store.add_texts(["d"], ids=["id-d"])

        assert store._ids == ["id-a", "id-c", "id-d"]
        docs = store.get_by_ids(["id-a", "id-c", "id-d"])
        assert len(docs) == 3
