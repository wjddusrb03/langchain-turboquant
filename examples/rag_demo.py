"""RAG Demo — TurboQuant Vector Store for LangChain.

This demo shows how to use TurboQuantVectorStore as a drop-in replacement
for any LangChain vector store, with ~6x memory savings.

No API keys needed — uses a local fake embedding for demonstration.
To use real embeddings, replace FakeEmbeddings with OpenAIEmbeddings etc.
"""

import sys
import io

# Fix Windows console encoding
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np
from langchain_core.embeddings import Embeddings

from langchain_turboquant import TurboQuantVectorStore


# ---------------------------------------------------------------------------
# 1. Fake embeddings (no API key needed for demo)
# ---------------------------------------------------------------------------

class FakeEmbeddings(Embeddings):
    """Deterministic hash-based embeddings for offline demo."""

    def __init__(self, dim: int = 384):
        self.dim = dim

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        seed = hash(text) % (2**31)
        rng = np.random.RandomState(seed)
        vec = rng.randn(self.dim)
        # Add some semantic structure: similar words → closer vectors
        for i, word in enumerate(text.lower().split()):
            word_seed = hash(word) % (2**31)
            word_rng = np.random.RandomState(word_seed)
            vec += word_rng.randn(self.dim) * 0.3
        vec /= np.linalg.norm(vec)
        return vec.tolist()


# ---------------------------------------------------------------------------
# 2. Sample documents (TurboQuant knowledge base)
# ---------------------------------------------------------------------------

documents = [
    "TurboQuant is a compression algorithm by Google Research that reduces LLM KV cache memory by 6x.",
    "PolarQuant rotates vectors using a random orthogonal matrix and quantizes in polar coordinates.",
    "QJL uses the Johnson-Lindenstrauss transform to sketch residual errors with just 1 bit per dimension.",
    "TurboQuant achieves 3-bit quantization with zero accuracy loss on the Needle-in-a-Haystack benchmark.",
    "The algorithm requires no training or fine-tuning — it is applied post-training as a drop-in replacement.",
    "On NVIDIA H100 GPUs, 4-bit TurboQuant achieves up to 8x speedup over 32-bit uncompressed keys.",
    "TurboQuant was published at ICLR 2026 and will be presented in Rio de Janeiro, Brazil.",
    "The key innovation is the asymmetric estimator that computes inner products directly on compressed data.",
    "Samsung and SK Hynix stock prices fell after the announcement due to fears of reduced memory demand.",
    "Experts argue that TurboQuant will actually increase AI adoption, driving more memory demand long-term.",
    "LangChain is a framework for building applications with large language models using composable components.",
    "Vector stores in LangChain enable similarity search over document embeddings for RAG pipelines.",
    "Python is a popular programming language used extensively in machine learning and data science.",
    "The weather in Seoul today is partly cloudy with temperatures around 15 degrees Celsius.",
    "RAG (Retrieval-Augmented Generation) combines document retrieval with LLM generation for accurate answers.",
]

metadatas = [
    {"topic": "turboquant", "source": "google_research"},
    {"topic": "turboquant", "source": "algorithm"},
    {"topic": "turboquant", "source": "algorithm"},
    {"topic": "turboquant", "source": "benchmark"},
    {"topic": "turboquant", "source": "feature"},
    {"topic": "turboquant", "source": "benchmark"},
    {"topic": "turboquant", "source": "conference"},
    {"topic": "turboquant", "source": "algorithm"},
    {"topic": "market", "source": "news"},
    {"topic": "market", "source": "analysis"},
    {"topic": "langchain", "source": "docs"},
    {"topic": "langchain", "source": "docs"},
    {"topic": "general", "source": "misc"},
    {"topic": "general", "source": "misc"},
    {"topic": "rag", "source": "docs"},
]


# ---------------------------------------------------------------------------
# 3. Build the vector store
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("  TurboQuant Vector Store - RAG Demo")
    print("=" * 60)

    embedding = FakeEmbeddings(dim=384)

    # Create store with 3-bit quantization
    print("\n[1] Building vector store with 3-bit TurboQuant compression...")
    store = TurboQuantVectorStore.from_texts(
        documents,
        embedding=embedding,
        metadatas=metadatas,
        bits=3,
    )

    # Memory stats
    stats = store.memory_stats()
    print(f"    Documents:        {stats['num_documents']}")
    print(f"    Dimension:        {stats['dimension']}")
    print(f"    Compression:      {stats['compression_ratio']}")
    print(f"    Memory saved:     {stats['memory_saved_pct']}")
    print(f"    Original size:    {stats['original_bytes']:.0f} bytes")
    print(f"    Compressed size:  {stats['compressed_bytes']:.0f} bytes")

    # ---------------------------------------------------------------------------
    # 4. Similarity search queries
    # ---------------------------------------------------------------------------

    queries = [
        "How does TurboQuant compress vectors?",
        "What is the performance on GPU?",
        "How did the stock market react?",
        "What is RAG?",
        "What is the weather like?",
    ]

    print("\n[2] Running similarity searches...\n")

    for query in queries:
        print(f"  Q: {query}")
        results = store.similarity_search_with_score(query, k=3)
        for i, (doc, score) in enumerate(results):
            topic = doc.metadata.get("topic", "?")
            print(f"     {i+1}. [{topic}] (score={score:.3f}) {doc.page_content[:80]}...")
        print()

    # ---------------------------------------------------------------------------
    # 5. Save and reload
    # ---------------------------------------------------------------------------

    print("[3] Testing save/load persistence...")
    store.save("demo_store.pkl")
    loaded = TurboQuantVectorStore.load("demo_store.pkl", embedding=embedding)
    results = loaded.similarity_search("compression algorithm", k=1)
    print(f"    Loaded store, search result: {results[0].page_content[:60]}...")

    # ---------------------------------------------------------------------------
    # 6. Use as LangChain retriever
    # ---------------------------------------------------------------------------

    print("\n[4] Using as LangChain retriever...")
    retriever = store.as_retriever(search_kwargs={"k": 2})
    # retriever.invoke() would work in a full LangChain chain
    print("    Retriever created successfully!")
    print(f"    Type: {type(retriever).__name__}")

    print("\n" + "=" * 60)
    print("  Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
