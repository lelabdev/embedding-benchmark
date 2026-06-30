#!/usr/bin/env python3
"""
Embedding Model Benchmark
Compare retrieval quality across embedding models.

Usage:
  python3 benchmark.py                    # Run all models
  python3 benchmark.py --model qwen3      # Run specific model
"""

import json, os, sys, time, argparse, urllib.request
import numpy as np
from pathlib import Path

# ─── Config ──────────────────────────────────────────────

DOCS_DIR = Path(__file__).parent / "docs"
QUERIES_FILE = Path(__file__).parent / "golden_dataset.json"
RESULTS_DIR = Path(__file__).parent / "results"

MODELS = {
    "qwen3-8b": {
        "provider": "openrouter",
        "model": "qwen/qwen3-embedding-8b",
        "dimensions": 4096,
        "url": "https://openrouter.ai/api/v1/embeddings",
    },
    "qwen3-8b-1024": {
        "provider": "openrouter",
        "model": "qwen/qwen3-embedding-8b",
        "dimensions": 1024,
        "url": "https://openrouter.ai/api/v1/embeddings",
    },
    "qwen3-8b-512": {
        "provider": "openrouter",
        "model": "qwen/qwen3-embedding-8b",
        "dimensions": 512,
        "url": "https://openrouter.ai/api/v1/embeddings",
    },
    "bge-m3": {
        "provider": "openrouter",
        "model": "baai/bge-m3",
        "dimensions": 1024,
        "url": "https://openrouter.ai/api/v1/embeddings",
    },
    "mistral-embed": {
        "provider": "openrouter",
        "model": "mistralai/mistral-embed-2312",
        "dimensions": None,  # doesn't support dimensions param
        "url": "https://openrouter.ai/api/v1/embeddings",
    },
    "nomic-embed": {
        "provider": "openrouter",
        "model": "nomic-ai/nomic-embed-text-v1.5",
        "dimensions": None,  # doesn't support dimensions param
        "url": "https://openrouter.ai/api/v1/embeddings",
    },
}

CHUNK_SIZE = 500  # chars per chunk
TOP_K = 5         # recall cutoff

# ─── Helpers ─────────────────────────────────────────────

def get_api_key():
    key = os.environ.get("OPENROUTER_API_KEY")
    if key:
        return key
    # Try pass
    import subprocess
    result = subprocess.run(["pass", "openrouter/api-key"], capture_output=True, text=True)
    if result.returncode == 0:
        return result.stdout.strip()
    raise RuntimeError("No OPENROUTER_API_KEY found. Set env var or install in pass.")

def chunk_text(text, size=CHUNK_SIZE):
    """Simple char-based chunking."""
    chunks = []
    # Skip header lines
    lines = text.split("\n")
    body = "\n".join(l for l in lines if not l.startswith("#") and not l.startswith("Source:"))
    body = body.strip()
    
    paragraphs = body.split("\n\n")
    current = ""
    for para in paragraphs:
        if len(current) + len(para) < size:
            current += "\n\n" + para if current else para
        else:
            if current:
                chunks.append(current.strip())
            current = para
    if current:
        chunks.append(current.strip())
    return [c for c in chunks if len(c) > 20]  # filter tiny chunks

def get_embedding(text, model_config, api_key):
    """Get embedding from OpenRouter API."""
    payload = {
        "model": model_config["model"],
        "input": text,
    }
    if model_config.get("dimensions"):
        payload["dimensions"] = model_config["dimensions"]
    data = json.dumps(payload).encode()
    
    req = urllib.request.Request(
        model_config["url"],
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    
    resp = urllib.request.urlopen(req, timeout=30)
    result = json.loads(resp.read())
    return result["data"][0]["embedding"]

def cosine_similarity(a, b):
    """Cosine similarity between two vectors."""
    a = np.array(a)
    b = np.array(b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return np.dot(a, b) / (norm_a * norm_b)

# ─── Benchmark ───────────────────────────────────────────

def load_docs():
    """Load and chunk all documents."""
    docs = []  # [{"id": "tokyo_fr_0", "source": "tokyo_fr", "text": "..."}]
    for f in sorted(DOCS_DIR.glob("*.txt")):
        source_id = f.stem  # e.g. "tokyo_fr"
        text = f.read_text()
        chunks = chunk_text(text)
        for i, chunk in enumerate(chunks):
            docs.append({
                "id": f"{source_id}_{i}",
                "source": source_id,
                "text": chunk,
            })
    return docs

def run_benchmark(model_name, model_config, api_key, docs, queries):
    """Run benchmark for one model."""
    print(f"\n{'='*60}")
    print(f"Model: {model_name} ({model_config['model']})")
    print(f"Dims: {model_config.get('dimensions', '?')}")
    print(f"{'='*60}")
    
    # 1. Embed all documents
    print(f"\nEmbedding {len(docs)} chunks...")
    doc_embeddings = []
    total_chars = 0
    embed_start = time.time()
    
    for i, doc in enumerate(docs):
        try:
            emb = get_embedding(doc["text"], model_config, api_key)
            doc_embeddings.append(emb)
            total_chars += len(doc["text"])
            if (i + 1) % 5 == 0:
                print(f"  {i+1}/{len(docs)} chunks embedded")
            time.sleep(0.1)  # gentle rate limit
        except Exception as e:
            print(f"  FAIL chunk {doc['id']}: {e}")
            doc_embeddings.append(None)
    
    embed_time = time.time() - embed_start
    valid = [e for e in doc_embeddings if e is not None]
    print(f"  Done: {len(valid)}/{len(docs)} chunks in {embed_time:.1f}s")
    
    # 2. Run queries
    print(f"\nRunning {len(queries)} queries...")
    hits = 0
    mrr_sum = 0
    per_query = []
    
    for q in queries:
        try:
            q_emb = get_embedding(q["question"], model_config, api_key)
            time.sleep(0.1)
        except Exception as e:
            print(f"  FAIL query: {q['question'][:40]}... — {e}")
            continue
        
        # Score all docs
        scores = []
        for i, doc_emb in enumerate(doc_embeddings):
            if doc_emb is not None:
                sim = cosine_similarity(q_emb, doc_emb)
                scores.append((sim, docs[i]))
        scores.sort(key=lambda x: x[0], reverse=True)
        
        # Check if expected source is in top-K
        expected = q["relevant_source_ids_keyword"]
        top_k = scores[:TOP_K]
        
        rank = None
        for i, (sim, doc) in enumerate(top_k):
            if doc["source"] == expected:
                rank = i + 1
                break
        
        hit = rank is not None
        rr = 1.0 / rank if rank else 0.0
        
        if hit:
            hits += 1
        mrr_sum += rr
        
        per_query.append({
            "question": q["question"],
            "expected_source": expected,
            "rank": rank,
            "hit": hit,
            "top1_source": top_k[0][1]["source"] if top_k else None,
            "top1_sim": round(top_k[0][0], 4) if top_k else None,
        })
        
        status = f"✓ rank {rank}" if hit else "✗ MISS"
        print(f"  {status} | {q['question'][:50]}")
    
    # 3. Results
    total_queries = len(per_query)
    hit_rate = hits / total_queries if total_queries else 0
    mrr = mrr_sum / total_queries if total_queries else 0
    
    results = {
        "model": model_name,
        "model_id": model_config["model"],
        "dimensions": model_config.get("dimensions"),
        "total_chunks": len(docs),
        "valid_embeddings": len(valid),
        "total_queries": total_queries,
        "hit_rate": round(hit_rate, 4),
        "mrr": round(mrr, 4),
        "embed_time_s": round(embed_time, 1),
        "total_chars_embedded": total_chars,
        "per_query": per_query,
    }
    
    print(f"\n  ── Results ──")
    print(f"  Hit Rate @{TOP_K}: {hit_rate:.1%} ({hits}/{total_queries})")
    print(f"  MRR: {mrr:.4f}")
    print(f"  Embed time: {embed_time:.1f}s")
    
    return results

# ─── Main ────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Embedding Model Benchmark")
    parser.add_argument("--model", choices=list(MODELS.keys()), help="Run specific model only")
    args = parser.parse_args()
    
    api_key = get_api_key()
    docs = load_docs()
    queries = json.loads(QUERIES_FILE.read_text())
    
    print(f"Loaded {len(docs)} chunks from {len(set(d['source'] for d in docs))} documents")
    print(f"Loaded {len(queries)} queries")
    
    models_to_run = [args.model] if args.model else list(MODELS.keys())
    all_results = {}
    
    for model_name in models_to_run:
        config = MODELS[model_name]
        results = run_benchmark(model_name, config, api_key, docs, queries)
        all_results[model_name] = results
        
        # Save individual results
        outfile = RESULTS_DIR / f"{model_name}.json"
        outfile.parent.mkdir(exist_ok=True)
        outfile.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    
    # Summary comparison
    if len(all_results) > 1:
        print(f"\n{'='*60}")
        print("SUMMARY COMPARISON")
        print(f"{'='*60}")
        print(f"{'Model':<20} {'Hit Rate':>10} {'MRR':>8} {'Time':>8}")
        print("-" * 48)
        for name, r in sorted(all_results.items(), key=lambda x: x[1]["hit_rate"], reverse=True):
            print(f"{name:<20} {r['hit_rate']:>9.1%} {r['mrr']:>8.4f} {r['embed_time_s']:>7.1f}s")
        
        # Save summary
        summary = {name: {k: v for k, v in r.items() if k != "per_query"} for name, r in all_results.items()}
        (RESULTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
        print(f"\nResults saved to {RESULTS_DIR}/")

if __name__ == "__main__":
    main()
