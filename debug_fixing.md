# Debug & Fixing Guide — RAG System
**Project:** RAG System for AI/ML Wikipedia Knowledge Base  

This document covers the most common errors you will encounter while building this RAG system and exactly how to fix them.

---

## 1. Environment & Installation Issues

### 1.1 `torch` / `sentence-transformers` version mismatch
**Symptom:**
```
ImportError: cannot import name 'cached_download' from 'huggingface_hub'
```
**Fix:**
```bash
pip install --upgrade huggingface_hub sentence-transformers transformers
```

---

### 1.2 ChromaDB migration error on existing DB
**Symptom:**
```
ValueError: You are using a deprecated version of Chroma. Please upgrade.
```
**Fix:**
```bash
# Delete old DB and re-index
rm -rf ./vectorstore/chroma_db
python src/embed.py
```

---

### 1.3 `tiktoken` fails on non-standard text
**Symptom:**
```
UnicodeDecodeError: 'utf-8' codec can't decode byte...
```
**Fix in `ingest.py`:**
```python
# When reading/writing files, always use encoding + error handling
with open(path, "w", encoding="utf-8", errors="replace") as f:
    json.dump(data, f, ensure_ascii=False)
```

---

### 1.4 `wikipedia-api` returns empty content
**Symptom:** `page.text` is an empty string.
**Fix:**
```python
import wikipediaapi

wiki = wikipediaapi.Wikipedia(
    language='en',
    user_agent='rag-project/1.0 (your@email.com)'  # required, or you get 403
)

page = wiki.page("Transformer (machine learning model)")
if not page.exists():
    print("Page not found, trying alternate title...")
```
Also check: some titles have disambiguation pages. Use the exact Wikipedia article title.

---

## 2. Embedding & Vector Store Issues

### 2.1 ChromaDB collection already exists error
**Symptom:**
```
chromadb.errors.UniqueConstraintError: Collection already exists
```
**Fix:**
```python
# Use get_or_create instead of create_collection
collection = client.get_or_create_collection(
    name="rag_wikipedia",
    metadata={"hnsw:space": "cosine"}
)
```

---

### 2.2 Slow embedding on CPU (large corpus)
**Symptom:** Embedding 1000+ chunks takes > 30 minutes.
**Fix:**
```python
# Use batch encoding + show_progress_bar
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(
    chunks,
    batch_size=64,       # increase if you have RAM
    show_progress_bar=True,
    convert_to_numpy=True
)
```

---

### 2.3 Dimension mismatch when switching models
**Symptom:**
```
chromadb.errors.InvalidDimensionException: Embedding dimension 768 does not match 384
```
**Fix:** You changed the embedding model but the old ChromaDB index uses a different dimension.
```bash
# Always delete the DB before re-indexing with a new model
rm -rf ./vectorstore/chroma_db
python src/embed.py
```

---

### 2.4 ChromaDB returns wrong results (cosine vs L2)
**Symptom:** Retrieved chunks seem semantically irrelevant.
**Fix:**
```python
# Explicitly set cosine distance space when creating collection
collection = client.get_or_create_collection(
    name="rag_wikipedia",
    metadata={"hnsw:space": "cosine"}  # <-- critical
)
```

---

## 3. Retrieval Issues

### 3.1 Re-ranker returns empty results
**Symptom:** Cross-encoder scores are all 0.0 or returns `[]`.
**Fix:**
```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# Scores expect (query, passage) pairs — check format
pairs = [(query, chunk["text"]) for chunk in candidates]
scores = reranker.predict(pairs)

# Sort by score descending
ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
top3 = [doc for _, doc in ranked[:3]]
```

---

### 3.2 Retrieved chunks have no metadata
**Symptom:** `chunk["metadata"]` is `{}` or raises `KeyError`.
**Fix in `embed.py`:**
```python
# Ensure metadata is passed during upsert
collection.upsert(
    ids=[chunk["chunk_id"] for chunk in batch],
    embeddings=embeddings.tolist(),
    documents=[chunk["text"] for chunk in batch],
    metadatas=[{                              # <-- don't skip this
        "source": chunk["source"],
        "url": chunk["url"],
        "section": chunk["section"]
    } for chunk in batch]
)
```

---

### 3.3 Query returns chunks from wrong topic
**Symptom:** Searching "BERT architecture" returns chunks about "BERT (name)" (the puppet).
**Fix:** Add metadata filtering:
```python
results = collection.query(
    query_embeddings=[query_embedding],
    n_results=20,
    where={"source": {"$in": ALLOWED_SOURCES}}  # filter to AI/ML articles only
)
```

---

## 4. Generation / LLM Issues

### 4.1 Groq API rate limit exceeded
**Symptom:**
```
groq.RateLimitError: Rate limit reached: 30 req/min
```
**Fix:**
```python
import time
from groq import Groq, RateLimitError

def call_groq_with_retry(client, messages, max_retries=3):
    for attempt in range(max_retries):
        try:
            return client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=messages,
                max_tokens=512
            )
        except RateLimitError:
            wait = 2 ** attempt  # exponential backoff: 1s, 2s, 4s
            print(f"Rate limited. Waiting {wait}s...")
            time.sleep(wait)
    raise Exception("Max retries exceeded")
```

---

### 4.2 LLM hallucinating despite context
**Symptom:** Answer mentions facts not present in retrieved chunks.
**Fix — Strengthen the prompt:**
```python
SYSTEM_PROMPT = """You are a grounded AI assistant. 
RULES:
1. Answer ONLY using information from the provided context.
2. If the context does not contain the answer, say exactly: "The provided context does not contain enough information to answer this question."
3. Do NOT use any prior knowledge or make assumptions.
4. Always end with: Source: [article title]"""
```
**Fix — Add NLI faithfulness check post-generation:**
```python
from transformers import pipeline

nli = pipeline("text-classification", model="cross-encoder/nli-deberta-v3-small")

def check_faithfulness(answer: str, context: str) -> float:
    result = nli(f"{context} [SEP] {answer}")
    # Returns "ENTAILMENT", "NEUTRAL", "CONTRADICTION"
    label_map = {"ENTAILMENT": 1.0, "NEUTRAL": 0.5, "CONTRADICTION": 0.0}
    return label_map.get(result[0]["label"], 0.5)
```

---

### 4.3 Groq returns empty `choices`
**Symptom:**
```python
IndexError: list index out of range on data.choices[0]
```
**Fix:**
```python
response = client.chat.completions.create(...)
if not response.choices:
    return "Error: LLM returned no response."
answer = response.choices[0].message.content.strip()
```

---

### 4.4 Context window exceeded
**Symptom:**
```
groq.BadRequestError: max_tokens + prompt tokens > context window
```
**Fix:** Truncate context before sending:
```python
import tiktoken

enc = tiktoken.get_encoding("cl100k_base")

def truncate_context(chunks: list[str], max_tokens: int = 3000) -> str:
    context = ""
    for chunk in chunks:
        tokens = enc.encode(chunk)
        if len(enc.encode(context)) + len(tokens) > max_tokens:
            break
        context += chunk + "\n\n"
    return context.strip()
```

---

## 5. Fine-Tuning Issues

### 5.1 `MultipleNegativesRankingLoss` crashes on small dataset
**Symptom:**
```
ValueError: in_batch_negatives requires batch_size >= 2
```
**Fix:**
```python
train_dataloader = DataLoader(
    train_examples,
    shuffle=True,
    batch_size=16   # must be ≥ 2, recommended ≥ 8
)
```

---

### 5.2 Fine-tuned model is worse than baseline
**Symptom:** MRR@10 drops after fine-tuning.
**Possible causes & fixes:**

| Cause | Fix |
|---|---|
| Too few training examples (<200) | Generate more QA pairs |
| Hard negatives are too easy | Use BM25 top-5 results as negatives (harder) |
| Learning rate too high | Reduce from 2e-5 to 5e-6 |
| Too many epochs | Reduce from 3 to 1–2, watch val loss |

```python
# Correct training config
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=2,
    warmup_steps=100,
    evaluation_steps=50,
    evaluator=evaluator,
    output_path="embeddings/finetuned/",
    optimizer_params={"lr": 2e-5},
    save_best_model=True   # saves best checkpoint, not last
)
```

---

### 5.3 QA pair generation with Groq is too slow
**Symptom:** Generating 500 QA pairs takes hours.
**Fix:** Use async batch generation:
```python
import asyncio
from groq import AsyncGroq

async def generate_qa_batch(chunks: list[str], client: AsyncGroq) -> list[dict]:
    tasks = [generate_qa(chunk, client) for chunk in chunks]
    return await asyncio.gather(*tasks)
```

---

## 6. RAGAS Evaluation Issues

### 6.1 RAGAS requires OpenAI key by default
**Symptom:**
```
openai.AuthenticationError: No API key provided
```
**Fix:** Configure RAGAS to use a different LLM judge:
```python
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from langchain_groq import ChatGroq

groq_llm = ChatGroq(model="llama-3.1-8b-instant", api_key=GROQ_API_KEY)
ragas_llm = LangchainLLMWrapper(groq_llm)

# Pass custom LLM to RAGAS
result = evaluate(
    dataset=eval_dataset,
    metrics=[faithfulness, answer_relevance],
    llm=ragas_llm
)
```

---

### 6.2 RAGAS `faithfulness` always returns 1.0
**Symptom:** Every answer scores perfect faithfulness — even clearly hallucinated ones.
**Fix:** This often happens when the LLM judge is also the generator (circular). Use a different judge model:
```python
from langchain_community.llms import Ollama

judge_llm = Ollama(model="mistral")  # different from generator
ragas_llm = LangchainLLMWrapper(judge_llm)
```

---

## 7. General Debugging Tips

### 7.1 Enable verbose logging everywhere
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### 7.2 Log every pipeline step to file
```python
import json
from datetime import datetime

def log_run(query, retrieved, answer):
    with open("logs/pipeline_log.jsonl", "a") as f:
        f.write(json.dumps({
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "retrieved_chunks": retrieved,
            "answer": answer
        }) + "\n")
```

### 7.3 Quick sanity check script
```python
# Run this after every major change
python -c "
from src.pipeline import RAGPipeline
rag = RAGPipeline()
result = rag.query('What is a transformer model?')
print('ANSWER:', result['answer'][:200])
print('SOURCES:', [r['source'] for r in result['sources']])
assert len(result['answer']) > 10, 'Answer is too short!'
print('Sanity check passed.')
"
```

---

*Debug guide version: 1.0 | Last updated: March 14, 2026*
