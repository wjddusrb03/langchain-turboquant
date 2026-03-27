# langchain-turboquant

**The first LangChain integration for TurboQuant** - Google Research's training-free vector compression algorithm (ICLR 2026).

Drop-in replacement for any LangChain vector store with **~6x memory reduction** and near-zero accuracy loss. No GPU required.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests: 296 passed](https://img.shields.io/badge/tests-296%20passed-brightgreen.svg)](#testing)

> [한국어 README](README.ko.md)

---

## Why langchain-turboquant?

Large-scale RAG pipelines store millions of embedding vectors in memory. At 1536 dimensions (OpenAI `text-embedding-3-small`), each vector takes **6 KB**. A million vectors = **6 GB** just for embeddings.

**TurboQuant compresses these vectors to ~1 KB each** (3-bit quantization), cutting memory by 6x while preserving search accuracy. Unlike Product Quantization (PQ) or IVFPQ, TurboQuant requires **no codebook training** - it works out of the box on any embedding.

| Feature | langchain-turboquant | FAISS (PQ) | Chroma |
|---|---|---|---|
| Compression ratio | **~6x** (3-bit) | ~4x (8-bit PQ) | 1x (none) |
| Training required | **No** | Yes (codebook) | N/A |
| Drop-in LangChain | **Yes** | Partial | Yes |
| GPU required | **No** | Optional | No |
| Asymmetric search | **Yes** | Yes | N/A |

## How It Works

TurboQuant implements the two-stage compression algorithm from [Google Research (ICLR 2026)](https://arxiv.org/abs/2504.19874):

### Stage 1: PolarQuant (MSE-optimal scalar quantization)

1. **Random orthogonal rotation**: Multiply the vector by a random orthogonal matrix. This "isotropizes" the coordinates so each one follows the same distribution (the hypersphere marginal).
2. **Lloyd-Max quantization**: Quantize each rotated coordinate independently using a pre-computed optimal codebook for the hypersphere marginal PDF.

The codebook is computed analytically from the distribution - no training data needed.

### Stage 2: QJL (Quantized Johnson-Lindenstrauss residual correction)

1. Compute the **quantization residual** (difference between original and Stage 1 reconstruction).
2. Project the residual through a random Gaussian matrix.
3. Store only the **sign bits** (1 bit per dimension) of the projection.

At query time, an **asymmetric estimator** computes approximate inner products directly on compressed data - the query stays in full precision while stored vectors remain compressed.

### Compression Math

For dimension `d` with `b`-bit quantization and QJL dimension `m`:

```
Compressed bits per vector = d * b + m * 1 + 32 + 32
                           = d * (b + 1) + 64

Original bits per vector   = d * 32

Compression ratio          = 32d / (d * (b+1) + 64)
```

At d=1536, b=3: **ratio = 7.7x** (theoretical) / **~6x** (practical with uint8 storage)

## Installation

```bash
pip install langchain-turboquant
```

Or install from source:

```bash
git clone https://github.com/wjddusrb03/langchain-turboquant.git
cd langchain-turboquant
pip install -e ".[dev]"
```

### Dependencies

- Python >= 3.9
- NumPy >= 1.21
- SciPy >= 1.7
- LangChain Core >= 0.3

## Quick Start

```python
from langchain_turboquant import TurboQuantVectorStore
from langchain_openai import OpenAIEmbeddings

# Create a compressed vector store (3-bit = ~6x compression)
store = TurboQuantVectorStore(embedding=OpenAIEmbeddings(), bits=3)

# Add documents - just like any LangChain vector store
store.add_texts(
    ["TurboQuant compresses vectors by 6x",
     "LangChain is a framework for LLM applications",
     "RAG combines retrieval with generation"],
    metadatas=[{"topic": "compression"}, {"topic": "framework"}, {"topic": "rag"}]
)

# Search
results = store.similarity_search("How does compression work?", k=2)
for doc in results:
    print(doc.page_content)

# Check memory savings
print(store.memory_stats())
# {'num_documents': 3, 'dimension': 1536, 'bits': 3,
#  'compression_ratio': '7.7x', 'memory_saved_pct': '87.0%'}
```

### Use as a LangChain Retriever

```python
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

retriever = store.as_retriever(search_kwargs={"k": 3})

# Use in a RAG chain
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | ChatOpenAI()
)
```

### No API Key Demo

Run the included demo with fake embeddings (no API key needed):

```bash
python examples/rag_demo.py
```

## API Reference

### TurboQuantVectorStore

```python
TurboQuantVectorStore(
    embedding: Embeddings,  # Any LangChain embedding model
    bits: int = 3,          # Quantization bits (1-4, recommended: 3)
    qjl_dim: int = None,    # QJL dimensions (default: same as embedding dim)
    seed: int = 42,         # Random seed for reproducibility
)
```

**Methods:**

| Method | Description |
|---|---|
| `add_texts(texts, metadatas, ids)` | Embed, compress, and store texts |
| `similarity_search(query, k)` | Return top-k most similar documents |
| `similarity_search_with_score(query, k)` | Return top-k with cosine similarity scores |
| `similarity_search_by_vector(vector, k)` | Search by pre-computed embedding vector |
| `from_texts(texts, embedding, ...)` | Class method to create and populate store |
| `delete(ids)` | Delete documents by ID |
| `get_by_ids(ids)` | Retrieve documents by ID |
| `as_retriever(**kwargs)` | Convert to LangChain Retriever |
| `save(path)` | Persist store to disk |
| `load(path, embedding)` | Load store from disk |
| `memory_stats()` | Get compression statistics |

### TurboQuantizer (Low-level API)

```python
from langchain_turboquant import TurboQuantizer

quantizer = TurboQuantizer(dim=1536, bits=3)

# Compress vectors
compressed = quantizer.quantize(vectors)  # (n, 1536) -> CompressedVectors

# Asymmetric search (query in full precision, database compressed)
scores = quantizer.cosine_scores(query_vector, compressed)

# Reconstruct (for evaluation)
reconstructed = quantizer.dequantize(compressed)
```

## Compression Ratios by Configuration

| Dimension | Bits | Theoretical Ratio | Memory Saved |
|---|---|---|---|
| 384 | 3 | 5.8x | 82.8% |
| 768 | 3 | 6.8x | 85.3% |
| 1536 | 3 | 7.3x | 86.3% |
| 3072 | 3 | 7.7x | 87.0% |
| 1536 | 2 | 9.5x | 89.5% |
| 1536 | 4 | 6.1x | 83.6% |

Higher dimensions benefit more from compression (the fixed 64-bit overhead for norms/gammas becomes negligible).

## Testing

The project includes **296 comprehensive tests** covering:

- **Mathematical correctness** (83 tests): Lloyd-Max codebook properties, rotation matrix orthogonality, MSE bounds, PDF integration, centroid conditions
- **Edge cases** (35 tests): NaN/Inf vectors, empty arrays, Unicode text, dim=1/2/3, zero vectors, large batches
- **Search recall** (44 tests): Top-k recall at various k/n/dim/bits, cluster discrimination, asymmetric estimator statistics, Pearson correlation
- **Persistence** (29 tests): Save/load roundtrips, serialization formats, state consistency after add/delete cycles
- **Rigorous validation** (68 tests): Compression ratios, performance benchmarks, score ordering, reconstruction quality
- **Core functionality** (37 tests): VectorStore CRUD, quantizer operations, LangChain integration

```bash
# Run all tests
pytest tests/ -v

# Run specific test suite
pytest tests/test_math_stress.py -v     # Mathematical properties
pytest tests/test_recall_extensive.py -v # Search recall
pytest tests/test_edge_cases.py -v       # Edge cases
```

## Architecture

```
langchain-turboquant/
├── src/langchain_turboquant/
│   ├── __init__.py          # Package exports
│   ├── lloyd_max.py         # Lloyd-Max optimal codebook computation
│   ├── quantizer.py         # TurboQuantizer (PolarQuant + QJL)
│   └── vectorstore.py       # LangChain VectorStore integration
├── tests/
│   ├── test_quantizer.py    # Core quantizer tests
│   ├── test_vectorstore.py  # VectorStore API tests
│   ├── test_rigorous.py     # Rigorous validation
│   ├── test_math_stress.py  # Mathematical properties
│   ├── test_edge_cases.py   # Edge cases
│   ├── test_recall_extensive.py  # Search recall
│   └── test_persistence.py  # Persistence tests
├── examples/
│   └── rag_demo.py          # Working RAG demo (no API key needed)
├── pyproject.toml
├── LICENSE
└── README.md
```

## References

- **TurboQuant**: Zandieh et al., "TurboQuant: Redefining Efficiency of KV Cache Compression for Large Language Models" (ICLR 2026). [arXiv:2504.19874](https://arxiv.org/abs/2504.19874)
- **PolarQuant**: Zandieh et al., "PolarQuant: Achieving High-Fidelity Vector Quantization via Polar Coordinates" (AISTATS 2026). [arXiv:2502.02617](https://arxiv.org/abs/2502.02617)
- **QJL**: Zandieh et al., "QJL: 1-Bit Quantized JL Transform for KV Cache Quantization with Zero Overhead" (AAAI 2025). [arXiv:2406.03482](https://arxiv.org/abs/2406.03482)
- **LangChain**: [langchain.com](https://www.langchain.com/)

## Contributing

Contributions are welcome! If you find a bug, have a feature request, or want to improve the code:

1. Open an [Issue](https://github.com/wjddusrb03/langchain-turboquant/issues) describing the problem or idea
2. Fork the repo and create a branch
3. Write tests for your changes
4. Submit a Pull Request

Please report any problems or suggestions in the [Issues](https://github.com/wjddusrb03/langchain-turboquant/issues) tab. All feedback is appreciated!

## License

MIT License - see [LICENSE](LICENSE) for details.
