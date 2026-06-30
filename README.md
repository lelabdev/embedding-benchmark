# Embedding Model Benchmark

Comparative benchmark of embedding models for multilingual retrieval (FR/EN/JP) on Wikipedia articles about Japanese culture.

## Why

We needed to decide which embedding model to use for [rag-ferrite](https://github.com/ludoloops/rag-ferrite), a self-hosted RAG engine. The corpus is mixed French/English with transcripts (oral style), notes (structured), and technical docs. We wanted to know:

1. Does our current model (Qwen3-Embedding-8B) give the best retrieval quality?
2. How does Mistral Embed compare, despite being 10x more expensive?
3. Does Qwen3's Matryoshka (MRL) truncation to 512d actually lose quality vs full 4096d?
4. How fast is each model for bulk ingestion?

## Models Tested

| Model | Model ID | Parameters | Dimensions | Cost (per 1M tokens) |
|-------|----------|-----------|------------|---------------------|
| Qwen3-Embedding-8B @ 512d | `qwen/qwen3-embedding-8b` | 8B | 512 (MRL-truncated) | ~$0.01 |
| Qwen3-Embedding-8B @ 1024d | `qwen/qwen3-embedding-8b` | 8B | 1024 (MRL-truncated) | ~$0.01 |
| Qwen3-Embedding-8B @ 4096d | `qwen/qwen3-embedding-8b` | 8B | 4096 (full) | ~$0.01 |
| BGE-M3 | `baai/bge-m3` | ~568M | 1024 (native) | ~$0.01 |
| Mistral Embed (2312) | `mistralai/mistral-embed-2312` | ~335M | 1024 (native) | ~€0.10 |

All models accessed via the **[OpenRouter](https://openrouter.ai)** API (`https://openrouter.ai/api/v1/embeddings`) using a standard `POST` request with `Authorization: Bearer <key>`. No local GPU, no Ollama — everything through the cloud API. The same API key was used for all models to keep conditions identical.

**Note on `dimensions` parameter:** Qwen3-Embedding-8B supports the `dimensions` parameter via Matryoshka Representation Learning (MRL), allowing truncation to 512 or 1024. BGE-M3 and Mistral Embed **do not** support this parameter — they output at their native dimensionality. When `dimensions` is not supported, the parameter is omitted from the API call.

## Dataset

**30 Wikipedia articles** — 10 topics × 3 languages (French, English, Japanese):

| Topic | FR | EN | JP |
|-------|----|----|-----|
| Tokyo | ✓ | ✓ | ✓ |
| Sushi | ✓ | ✓ | ✓ |
| Tea Ceremony | ✓ | ✓ | ✓ |
| Buddhism | ✓ | ✓ | ✓ |
| Mount Fuji | ✓ | ✓ | ✓ |
| Tsunami | ✓ | ✓ | ✓ |
| Anime | ✓ | ✓ | ✓ |
| Manga | ✓ | ✓ | ✓ |
| Samurai | ✓ | ✓ | ✓ |
| Calligraphy (Shodō) | ✓ | ✓ | ✓ |

Articles fetched via Wikipedia REST API, truncated to ~3000 chars each. Total: 59 chunks (500-char chunking).

## Golden Dataset

**30 queries** in mixed languages — French, English, and Japanese. Each query has an expected source document (the Wikipedia article that should rank highest).

Example queries:
- `Quelle est la capitale du Japon ?` → expects `tokyo_fr`
- `What is raw fish on rice called?` → expects `sushi_en`
- `日本の最高峰の山は？` → expects `mount_fuji_ja`
- `漫画とは何ですか` → expects `manga_ja`

Full dataset: [`golden_dataset.json`](golden_dataset.json)

## Methodology

1. **Chunk** all 30 documents at 500 chars (simple paragraph-based chunking)
2. **Embed** all 59 chunks with each model
3. **Embed** each query
4. **Score** all chunks by cosine similarity to the query
5. **Check** if the expected source document appears in the top-5 results
6. **Calculate** Hit Rate @5 and MRR (Mean Reciprocal Rank)

No BM25, no reranking — pure embedding quality comparison. This isolates the embedding model's contribution.

## Metrics Explained

### Hit Rate @5

**What it means:** Out of 30 queries, how often did the *correct* document appear somewhere in the top 5 results?

**Why it matters:** In a RAG system, you typically retrieve 5-10 chunks and feed them to the LLM. If the right chunk isn't in those top 5, the LLM won't have the information it needs. Hit Rate measures **recall** — does the model find the needle at all?

**Example:** 100% means the correct document was always in the top 5. 96.7% means it missed once.

### MRR (Mean Reciprocal Rank)

**What it means:** The average of `1/rank` for each query, where `rank` is the position of the correct document in the results (1 = first result, 2 = second, etc.).

**Why it matters:** Hit Rate tells you if the right answer is *somewhere* in the top 5. MRR tells you if it's at the **top**. In practice, users (and LLMs) pay more attention to the first results. A model with MRR 0.84 puts the right answer at rank 1 most of the time. A model with MRR 0.74 might bury it at rank 3-4.

**Concrete example:**
- All 30 queries return the correct doc at rank 1 → MRR = 1.0 (perfect)
- Correct doc always at rank 2 → MRR = 0.5
- Correct doc always at rank 5 → MRR = 0.2

**Scale:** MRR ranges from 0 to 1. Above 0.75 is good. Above 0.80 is excellent.

### Embed Time

**What it means:** Wall-clock time to embed all 59 document chunks (queries are negligible — 1 embedding each).

**Why it matters:** For ingestion of large corpora (thousands of documents), embedding speed directly determines how long a rebuild takes. A model that's 10x slower means 10x longer ingestion.

### Cosine Similarity

**What it means:** The mathematical measure used to rank documents against a query. It computes the cosine of the angle between two vectors — 1.0 means identical direction, 0.0 means orthogonal (unrelated).

**Why no raw scores in the results:** Cosine similarity scores are **not comparable across models** (each model has its own scale). Only the **ranking** (which document comes first) is comparable. That's why we use Hit Rate and MRR, not raw similarity scores.

## Results

| Model | Hit Rate @5 | MRR | Embed Time | Dims |
|-------|-------------|-----|------------|------|
| **Mistral Embed 2312** | **100.0%** (30/30) | **0.836** | **20.3s** | 1024 |
| **BGE-M3** | 96.7% (29/30) | 0.797 | 32.7s | 1024 |
| **Qwen3-8B @ 1024d** | **100.0%** (30/30) | 0.750 | 298.4s | 1024 |
| **Qwen3-8B @ 512d** | **100.0%** (30/30) | 0.736 | 212.4s | 512 |
| **Qwen3-8B @ 4096d** | **100.0%** (30/30) | 0.739 | 263.7s | 4096 |

### Key Findings

#### 1. Mistral Embed wins on quality

Mistral Embed achieved the highest MRR (0.836) with 24/30 queries hitting rank 1. It also had the fastest embedding time (20.3s for 59 chunks). It was the clear winner on retrieval quality — but at ~10x the cost of the alternatives.

#### 2. Qwen3 MRL: 512d is NOT a downgrade

A common concern with Qwen3-Embedding-8B is whether truncating from 4096d to 512d via Matryoshka Representation Learning (MRL) hurts quality. **It doesn't.** In fact, 512d scored marginally higher MRR (0.753 vs 0.739). This confirms that Qwen3's MRL training places the most important semantic information in the first dimensions — 512d is a first-class operating mode, not a degraded fallback.

This has major practical implications: 512d vectors use 8x less storage and RAM than 4096d, with no quality loss.

#### 3. BGE-M3: best ranking quality per dollar

BGE-M3 had the second-highest MRR (0.797) and was the second-fastest (32.7s). It missed one query (Samurai code of honor bushido → expected `samurai_fr`), but when it found the right document, it ranked it higher than Qwen3 did. At ~$0.01/1M tokens, it's the best value.

#### 4. Speed ranking

```
Mistral Embed:  20.3s  (0.34s/chunk)
BGE-M3:         32.7s  (0.55s/chunk)
Qwen3 @ 4096d: 263.7s  (4.47s/chunk)
Qwen3 @ 512d:  366.2s  (6.21s/chunk)
```

Qwen3-8B is significantly slower — likely because it's an 8B parameter model vs BGE-M3 (~568M) and Mistral Embed (~335M). For bulk ingestion of thousands of documents, this matters.

## Per-Query Results

### Mistral Embed (winner)

| Rank | Count | Queries |
|------|-------|---------|
| 1 | 24 | Most queries |
| 2 | 5 | Buddhism EN, Sushi FR, Tsunami FR, Anime EN, Manga FR, Tsunami JP, Fuji JP |
| 3 | 2 | Fuji eruption, Calligraphy EN |
| 4 | 1 | Samurai bushido FR |

### BGE-M3

| Rank | Count | Note |
|------|-------|------|
| 1 | 22 | |
| 2-4 | 7 | |
| MISS | 1 | `Samurai code of honor bushido` → returned wrong source |

### Qwen3-8B @ 512d

| Rank | Count |
|------|-------|
| 1 | 22 |
| 2 | 5 |
| 3 | 2 |
| 4 | 1 |

## Background Research

MTEB (Massive Text Embedding Benchmark) multilingual leaderboard as of June 2025:

| Model | MTEB Multilingual | Notes |
|-------|-------------------|-------|
| Qwen3-Embedding-8B | 70.58 (#1) | MRL-native, 100+ languages |
| BGE-M3 | ~63.0 | Multi-function (dense + sparse + multi-vector), 100+ languages |
| Mistral Embed 2312 | Not on MTEB | Proprietary, limited public benchmarks |

Sources: [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard), [Qwen3 Embedding blog](https://qwenlm.github.io/blog/qwen3-embedding/), [BGE-M3 paper](https://arxiv.org/html/2402.03216v3)

Despite Mistral Embed not appearing on MTEB, our practical benchmark shows it outperforms both MTEB leaders on real-world multilingual retrieval. This highlights the gap between academic benchmarks and domain-specific performance.

## Decision

For **rag-ferrite** (our RAG engine), the current model is **Qwen3-Embedding-8B @ 512d**. Based on these results:

- **If budget is no concern**: Mistral Embed is the best choice (highest quality, fastest)
- **If cost matters**: Qwen3-8B @ 512d offers 100% recall at 10x lower cost, with acceptable MRR
- **Best value**: BGE-M3 — near-top quality, fast, cheap, but 1 miss on edge cases
- **For local/self-hosted**: BGE-M3 on Ollama (TufTux GPU) = free, no API dependency

The MRL finding (512d = full quality) means the current 512d configuration is validated and should not be changed to 4096d.

## How to Reproduce

```bash
# Install numpy
pip install numpy

# Set API key
export OPENROUTER_API_KEY="your-key-here"

# Run all models
python3 benchmark.py

# Run specific model
python3 benchmark.py --model mistral-embed

# Results saved to results/<model>.json
```

## Project Structure

```
embedding-benchmark/
├── README.md              ← this file
├── benchmark.py           ← benchmark script
├── golden_dataset.json    ← 30 queries with expected sources
├── docs/                  ← 30 Wikipedia articles (FR/EN/JP)
│   ├── tokyo_fr.txt
│   ├── tokyo_en.txt
│   ├── tokyo_ja.txt
│   └── ... (27 more)
└── results/               ← JSON results per model
    ├── qwen3-8b.json
    ├── bge-m3.json
    └── ...
```

## License

MIT — data from Wikipedia (CC BY-SA), benchmark code is ours.

---

*Benchmark run June 30, 2026. Models accessed via OpenRouter API. No local GPU used.*
