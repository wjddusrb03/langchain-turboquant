"""TurboQuant Vector Store for LangChain.

A memory-efficient vector store that compresses embedding vectors using the
TurboQuant algorithm (PolarQuant + QJL).  Achieves ~5-6x compression with
near-zero accuracy loss and NO codebook training.

Usage::

    from langchain_turboquant import TurboQuantVectorStore
    from langchain_openai import OpenAIEmbeddings

    store = TurboQuantVectorStore(embedding=OpenAIEmbeddings())
    store.add_texts(["Hello world", "LangChain is great"])
    results = store.similarity_search("greeting", k=1)
"""

from __future__ import annotations

import pickle
import uuid
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from langchain_turboquant.quantizer import CompressedVectors, TurboQuantizer


class TurboQuantVectorStore(VectorStore):
    """LangChain vector store backed by TurboQuant compression.

    Parameters
    ----------
    embedding : Embeddings
        Any LangChain-compatible embedding model.
    bits : int
        Quantization bits per coordinate (default 3 → ~6x compression).
    qjl_dim : int or None
        QJL projection dimensions (default = embedding dim).
    seed : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        embedding: Embeddings,
        bits: int = 3,
        qjl_dim: Optional[int] = None,
        seed: int = 42,
    ) -> None:
        self._embedding = embedding
        self._bits = bits
        self._qjl_dim = qjl_dim
        self._seed = seed

        # Lazily initialised after we know the embedding dimension
        self._quantizer: Optional[TurboQuantizer] = None
        self._dim: Optional[int] = None

        # In-memory storage
        self._ids: list[str] = []
        self._documents: list[Document] = []
        self._compressed: Optional[CompressedVectors] = None

    # ----- Properties -----

    @property
    def embeddings(self) -> Embeddings:
        return self._embedding

    # ----- Internal helpers -----

    def _ensure_quantizer(self, dim: int) -> TurboQuantizer:
        """Create the quantizer on first use (we need the dimension)."""
        if self._quantizer is None or self._dim != dim:
            self._dim = dim
            self._quantizer = TurboQuantizer(
                dim=dim,
                bits=self._bits,
                qjl_dim=self._qjl_dim,
                seed=self._seed,
            )
        return self._quantizer

    def _append_compressed(self, new_compressed: CompressedVectors) -> None:
        """Append new compressed vectors to the store."""
        if self._compressed is None:
            self._compressed = new_compressed
        else:
            self._compressed = CompressedVectors(
                indices=np.concatenate(
                    [self._compressed.indices, new_compressed.indices], axis=0
                ),
                qjl_bits=np.concatenate(
                    [self._compressed.qjl_bits, new_compressed.qjl_bits], axis=0
                ),
                gammas=np.concatenate(
                    [self._compressed.gammas, new_compressed.gammas]
                ),
                norms=np.concatenate(
                    [self._compressed.norms, new_compressed.norms]
                ),
            )

    # ----- VectorStore required methods -----

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[list[dict]] = None,
        *,
        ids: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> list[str]:
        """Embed texts, compress, and store.

        Parameters
        ----------
        texts : iterable of str
        metadatas : optional list of metadata dicts
        ids : optional list of document IDs

        Returns
        -------
        list of document IDs
        """
        texts_list = list(texts)
        if not texts_list:
            return []

        # Embed
        embeddings = self._embedding.embed_documents(texts_list)
        vectors = np.array(embeddings, dtype=np.float32)

        # Initialise quantizer with actual dimension
        quantizer = self._ensure_quantizer(vectors.shape[1])

        # Compress
        compressed = quantizer.quantize(vectors)
        self._append_compressed(compressed)

        # Store documents
        metadatas = metadatas or [{} for _ in texts_list]
        ids = ids or [str(uuid.uuid4()) for _ in texts_list]

        for text, meta, doc_id in zip(texts_list, metadatas, ids):
            self._ids.append(doc_id)
            self._documents.append(
                Document(page_content=text, metadata={**meta, "_id": doc_id})
            )

        return ids

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any,
    ) -> list[Document]:
        """Search for documents most similar to the query."""
        docs_and_scores = self.similarity_search_with_score(query, k=k, **kwargs)
        return [doc for doc, _ in docs_and_scores]

    @classmethod
    def from_texts(
        cls,
        texts: list[str],
        embedding: Embeddings,
        metadatas: Optional[list[dict]] = None,
        *,
        ids: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> TurboQuantVectorStore:
        """Create a TurboQuantVectorStore from a list of texts."""
        store = cls(embedding=embedding, **kwargs)
        store.add_texts(texts, metadatas=metadatas, ids=ids)
        return store

    # ----- Extended search methods -----

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any,
    ) -> list[tuple[Document, float]]:
        """Return top-k documents with cosine similarity scores."""
        if self._compressed is None or len(self._documents) == 0:
            return []

        query_embedding = self._embedding.embed_query(query)
        query_vec = np.array(query_embedding, dtype=np.float32)

        scores = self._quantizer.cosine_scores(query_vec, self._compressed)

        # Top-k (highest cosine similarity)
        k = min(k, len(self._documents))
        top_indices = np.argsort(scores)[::-1][:k]

        return [(self._documents[i], float(scores[i])) for i in top_indices]

    def similarity_search_by_vector(
        self,
        embedding: list[float],
        k: int = 4,
        **kwargs: Any,
    ) -> list[Document]:
        """Search by a pre-computed embedding vector."""
        if self._compressed is None or len(self._documents) == 0:
            return []

        query_vec = np.array(embedding, dtype=np.float32)
        scores = self._quantizer.cosine_scores(query_vec, self._compressed)

        k = min(k, len(self._documents))
        top_indices = np.argsort(scores)[::-1][:k]

        return [self._documents[i] for i in top_indices]

    def delete(
        self,
        ids: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> Optional[bool]:
        """Delete documents by IDs."""
        if ids is None:
            return False

        id_set = set(ids)
        keep_mask = np.array([doc_id not in id_set for doc_id in self._ids])

        if not keep_mask.any():
            self._ids = []
            self._documents = []
            self._compressed = None
            return True

        if keep_mask.all():
            return False

        # Filter
        self._ids = [self._ids[i] for i in range(len(self._ids)) if keep_mask[i]]
        self._documents = [
            self._documents[i] for i in range(len(self._documents)) if keep_mask[i]
        ]

        if self._compressed is not None:
            self._compressed = CompressedVectors(
                indices=self._compressed.indices[keep_mask],
                qjl_bits=self._compressed.qjl_bits[keep_mask],
                gammas=self._compressed.gammas[keep_mask],
                norms=self._compressed.norms[keep_mask],
            )

        return True

    def get_by_ids(self, ids: list[str]) -> list[Document]:
        """Retrieve documents by their IDs."""
        id_to_doc = {doc_id: doc for doc_id, doc in zip(self._ids, self._documents)}
        return [id_to_doc[doc_id] for doc_id in ids if doc_id in id_to_doc]

    def _select_relevance_score_fn(self) -> Callable[[float], float]:
        return self._cosine_relevance_score_fn

    # ----- Persistence -----

    def save(self, path: str) -> None:
        """Save the vector store to disk."""
        data = {
            "bits": self._bits,
            "qjl_dim": self._qjl_dim,
            "seed": self._seed,
            "dim": self._dim,
            "ids": self._ids,
            "documents": self._documents,
            "compressed": self._compressed,
            "quantizer_codebook": self._quantizer.codebook if self._quantizer else None,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(data, f)

    @classmethod
    def load(cls, path: str, embedding: Embeddings) -> TurboQuantVectorStore:
        """Load a saved vector store from disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)

        store = cls(
            embedding=embedding,
            bits=data["bits"],
            qjl_dim=data["qjl_dim"],
            seed=data["seed"],
        )
        store._dim = data["dim"]
        store._ids = data["ids"]
        store._documents = data["documents"]
        store._compressed = data["compressed"]

        if data["dim"] is not None:
            store._ensure_quantizer(data["dim"])

        return store

    # ----- Stats -----

    def memory_stats(self) -> dict[str, Any]:
        """Return memory usage statistics."""
        n = len(self._documents)
        if self._quantizer is None or n == 0:
            return {"num_documents": 0}

        compressed_bytes = self._quantizer.compressed_bytes_per_vector() * n
        original_bytes = self._quantizer.original_bytes_per_vector() * n
        ratio = self._quantizer.compression_ratio()

        return {
            "num_documents": n,
            "dimension": self._dim,
            "bits": self._bits,
            "compressed_bytes": compressed_bytes,
            "original_bytes": original_bytes,
            "compression_ratio": f"{ratio:.1f}x",
            "memory_saved_pct": f"{(1 - 1/ratio) * 100:.1f}%",
        }
