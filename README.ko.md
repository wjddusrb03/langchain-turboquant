# langchain-turboquant

**TurboQuant의 최초 LangChain 통합 라이브러리** - Google Research의 학습 불필요 벡터 압축 알고리즘 (ICLR 2026).

기존 LangChain 벡터 스토어를 그대로 대체하면서 **메모리 사용량을 약 6배 절감**하고, 정확도 손실은 거의 없습니다. GPU가 필요하지 않습니다.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests: 296 passed](https://img.shields.io/badge/tests-296%20passed-brightgreen.svg)](#테스트)

> [English README](README.md)

---

## 왜 langchain-turboquant인가?

대규모 RAG 파이프라인은 수백만 개의 임베딩 벡터를 메모리에 저장합니다. 1536차원(OpenAI `text-embedding-3-small`) 기준으로 벡터 하나당 **6 KB**가 필요합니다. 벡터 100만 개 = 임베딩만으로 **6 GB**입니다.

**TurboQuant은 벡터를 약 1 KB로 압축**합니다 (3비트 양자화). 검색 정확도를 유지하면서 메모리를 6배 줄여줍니다. Product Quantization(PQ)이나 IVFPQ와 달리, TurboQuant은 **코드북 학습이 필요 없습니다** - 어떤 임베딩에든 바로 적용 가능합니다.

| 기능 | langchain-turboquant | FAISS (PQ) | Chroma |
|---|---|---|---|
| 압축 비율 | **약 6배** (3비트) | 약 4배 (8비트 PQ) | 1배 (압축 없음) |
| 학습 필요 여부 | **불필요** | 필요 (코드북) | 해당 없음 |
| LangChain 드롭인 | **가능** | 부분적 | 가능 |
| GPU 필요 여부 | **불필요** | 선택적 | 불필요 |
| 비대칭 검색 | **지원** | 지원 | 해당 없음 |

## 작동 원리

TurboQuant은 [Google Research (ICLR 2026)](https://arxiv.org/abs/2504.19874)의 2단계 압축 알고리즘을 구현합니다:

### 1단계: PolarQuant (MSE 최적 스칼라 양자화)

1. **랜덤 직교 회전**: 벡터에 랜덤 직교 행렬을 곱합니다. 이를 통해 각 좌표가 동일한 분포(초구면 주변 분포)를 따르게 됩니다.
2. **Lloyd-Max 양자화**: 회전된 각 좌표를 초구면 주변 PDF에 대해 미리 계산된 최적 코드북으로 독립적으로 양자화합니다.

코드북은 분포로부터 해석적으로 계산되므로 학습 데이터가 필요 없습니다.

### 2단계: QJL (양자화된 Johnson-Lindenstrauss 잔차 보정)

1. **양자화 잔차**(원본과 1단계 복원의 차이)를 계산합니다.
2. 잔차를 랜덤 가우시안 행렬로 투영합니다.
3. 투영의 **부호 비트**(차원당 1비트)만 저장합니다.

검색 시에는 **비대칭 추정기**가 압축된 데이터에서 직접 내적을 근사 계산합니다 - 쿼리는 전체 정밀도를 유지하고, 저장된 벡터는 압축 상태로 남아있습니다.

### 압축 수식

차원 `d`, `b`비트 양자화, QJL 차원 `m`일 때:

```
벡터당 압축 비트 = d * b + m * 1 + 32 + 32
                = d * (b + 1) + 64

벡터당 원본 비트 = d * 32

압축 비율        = 32d / (d * (b+1) + 64)
```

d=1536, b=3일 때: **비율 = 7.7배** (이론적) / **약 6배** (uint8 저장 기준 실제)

## 설치

```bash
pip install langchain-turboquant
```

또는 소스에서 설치:

```bash
git clone https://github.com/wjddusrb03/langchain-turboquant.git
cd langchain-turboquant
pip install -e ".[dev]"
```

### 의존성

- Python >= 3.9
- NumPy >= 1.21
- SciPy >= 1.7
- LangChain Core >= 0.3

## 빠른 시작

```python
from langchain_turboquant import TurboQuantVectorStore
from langchain_openai import OpenAIEmbeddings

# 압축 벡터 스토어 생성 (3비트 = 약 6배 압축)
store = TurboQuantVectorStore(embedding=OpenAIEmbeddings(), bits=3)

# 문서 추가 - 기존 LangChain 벡터 스토어와 동일한 방식
store.add_texts(
    ["TurboQuant은 벡터를 6배 압축합니다",
     "LangChain은 LLM 애플리케이션을 위한 프레임워크입니다",
     "RAG는 검색과 생성을 결합합니다"],
    metadatas=[{"topic": "압축"}, {"topic": "프레임워크"}, {"topic": "rag"}]
)

# 검색
results = store.similarity_search("압축은 어떻게 작동하나요?", k=2)
for doc in results:
    print(doc.page_content)

# 메모리 절감 확인
print(store.memory_stats())
# {'num_documents': 3, 'dimension': 1536, 'bits': 3,
#  'compression_ratio': '7.7x', 'memory_saved_pct': '87.0%'}
```

### LangChain Retriever로 사용

```python
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

retriever = store.as_retriever(search_kwargs={"k": 3})

# RAG 체인에서 사용
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | ChatOpenAI()
)
```

### API 키 없이 데모 실행

포함된 데모를 가짜 임베딩으로 실행할 수 있습니다 (API 키 불필요):

```bash
python examples/rag_demo.py
```

## API 레퍼런스

### TurboQuantVectorStore

```python
TurboQuantVectorStore(
    embedding: Embeddings,  # LangChain 호환 임베딩 모델
    bits: int = 3,          # 좌표당 양자화 비트 (1-4, 권장: 3)
    qjl_dim: int = None,    # QJL 차원 (기본값: 임베딩 차원과 동일)
    seed: int = 42,         # 재현성을 위한 랜덤 시드
)
```

**메서드:**

| 메서드 | 설명 |
|---|---|
| `add_texts(texts, metadatas, ids)` | 텍스트를 임베딩, 압축, 저장 |
| `similarity_search(query, k)` | 가장 유사한 상위 k개 문서 반환 |
| `similarity_search_with_score(query, k)` | 코사인 유사도 점수와 함께 상위 k개 반환 |
| `similarity_search_by_vector(vector, k)` | 사전 계산된 임베딩 벡터로 검색 |
| `from_texts(texts, embedding, ...)` | 스토어를 생성하고 데이터를 채우는 클래스 메서드 |
| `delete(ids)` | ID로 문서 삭제 |
| `get_by_ids(ids)` | ID로 문서 조회 |
| `as_retriever(**kwargs)` | LangChain Retriever로 변환 |
| `save(path)` | 스토어를 디스크에 저장 |
| `load(path, embedding)` | 디스크에서 스토어 로드 |
| `memory_stats()` | 압축 통계 조회 |

### TurboQuantizer (저수준 API)

```python
from langchain_turboquant import TurboQuantizer

quantizer = TurboQuantizer(dim=1536, bits=3)

# 벡터 압축
compressed = quantizer.quantize(vectors)  # (n, 1536) -> CompressedVectors

# 비대칭 검색 (쿼리는 전체 정밀도, 데이터베이스는 압축 상태)
scores = quantizer.cosine_scores(query_vector, compressed)

# 복원 (평가용)
reconstructed = quantizer.dequantize(compressed)
```

## 설정별 압축 비율

| 차원 | 비트 | 이론적 비율 | 메모리 절감 |
|---|---|---|---|
| 384 | 3 | 5.8배 | 82.8% |
| 768 | 3 | 6.8배 | 85.3% |
| 1536 | 3 | 7.3배 | 86.3% |
| 3072 | 3 | 7.7배 | 87.0% |
| 1536 | 2 | 9.5배 | 89.5% |
| 1536 | 4 | 6.1배 | 83.6% |

차원이 높을수록 압축 효과가 더 큽니다 (norm/gamma에 대한 고정 64비트 오버헤드가 무시할 수 있게 됩니다).

## 테스트

프로젝트에는 다음을 검증하는 **296개의 포괄적 테스트**가 포함되어 있습니다:

- **수학적 정확성** (83개): Lloyd-Max 코드북 속성, 회전 행렬 직교성, MSE 경계, PDF 적분, 중심점 조건
- **엣지 케이스** (35개): NaN/Inf 벡터, 빈 배열, 유니코드 텍스트, dim=1/2/3, 영벡터, 대규모 배치
- **검색 리콜** (44개): 다양한 k/n/dim/bits에서의 top-k 리콜, 클러스터 구별, 비대칭 추정기 통계, 피어슨 상관계수
- **영속성** (29개): 저장/로드 왕복, 직렬화 형식, 추가/삭제 사이클 후 상태 일관성
- **엄밀한 검증** (68개): 압축 비율, 성능 벤치마크, 점수 정렬, 복원 품질
- **핵심 기능** (37개): VectorStore CRUD, 양자화기 연산, LangChain 통합

```bash
# 전체 테스트 실행
pytest tests/ -v

# 특정 테스트 모음 실행
pytest tests/test_math_stress.py -v     # 수학적 속성
pytest tests/test_recall_extensive.py -v # 검색 리콜
pytest tests/test_edge_cases.py -v       # 엣지 케이스
```

## 프로젝트 구조

```
langchain-turboquant/
├── src/langchain_turboquant/
│   ├── __init__.py          # 패키지 내보내기
│   ├── lloyd_max.py         # Lloyd-Max 최적 코드북 계산
│   ├── quantizer.py         # TurboQuantizer (PolarQuant + QJL)
│   └── vectorstore.py       # LangChain VectorStore 통합
├── tests/
│   ├── test_quantizer.py    # 핵심 양자화기 테스트
│   ├── test_vectorstore.py  # VectorStore API 테스트
│   ├── test_rigorous.py     # 엄밀한 검증
│   ├── test_math_stress.py  # 수학적 속성
│   ├── test_edge_cases.py   # 엣지 케이스
│   ├── test_recall_extensive.py  # 검색 리콜
│   └── test_persistence.py  # 영속성 테스트
├── examples/
│   └── rag_demo.py          # API 키 없이 실행 가능한 RAG 데모
├── pyproject.toml
├── LICENSE
└── README.md
```

## 참고 논문

- **TurboQuant**: Zandieh et al., "TurboQuant: Redefining Efficiency of KV Cache Compression for Large Language Models" (ICLR 2026). [arXiv:2504.19874](https://arxiv.org/abs/2504.19874)
- **PolarQuant**: Zandieh et al., "PolarQuant: Achieving High-Fidelity Vector Quantization via Polar Coordinates" (AISTATS 2026). [arXiv:2502.02617](https://arxiv.org/abs/2502.02617)
- **QJL**: Zandieh et al., "QJL: 1-Bit Quantized JL Transform for KV Cache Quantization with Zero Overhead" (AAAI 2025). [arXiv:2406.03482](https://arxiv.org/abs/2406.03482)
- **LangChain**: [langchain.com](https://www.langchain.com/)

## 기여하기

기여를 환영합니다! 버그를 발견하거나, 기능 제안이 있거나, 코드를 개선하고 싶다면:

1. [Issue](https://github.com/wjddusrb03/langchain-turboquant/issues)를 열어 문제나 아이디어를 설명해주세요
2. 레포를 포크하고 브랜치를 만드세요
3. 변경 사항에 대한 테스트를 작성하세요
4. Pull Request를 제출하세요

문제가 있거나 제안 사항이 있으면 [Issues](https://github.com/wjddusrb03/langchain-turboquant/issues) 탭에 알려주세요. 모든 피드백을 소중히 여깁니다!

## 라이선스

MIT 라이선스 - 자세한 내용은 [LICENSE](LICENSE)를 참조하세요.
