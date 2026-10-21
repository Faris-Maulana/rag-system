# Original Design Plan

> This is the plan written before implementation, kept for context on the reasoning.
> Where it differs from the code (e.g. it proposes RAGAS, nDCG and LLM fine-tuning),
> the [README](../README.md) describes what was actually built and measured.

---

## 1. Project Overview

We build a production-grade **Retrieval-Augmented Generation (RAG)** system on top of the **Wikipedia AI/ML knowledge base**. The system allows a user to ask domain-specific questions and receive accurate, grounded answers backed by retrieved evidence.

---

## 2. Knowledge Base Selection

**Choice: Wikipedia — Artificial Intelligence & Machine Learning Domain**

**Why Wikipedia?**
- Publicly available, no licensing issues
- Structured, high-quality text with clear section headings
- Broad coverage of AI/ML concepts (transformers, RAG, embeddings, LLMs, etc.) — directly relevant to domain-specific evaluation
- Easily fetched via the `wikipedia-api` Python library
- Represents a realistic corpus with natural language diversity, good for testing bias and retrieval quality

**Scope:** ~500 Wikipedia articles on topics: `Machine Learning`, `Deep Learning`, `Transformer (ML)`, `Large language model`, `Retrieval-augmented generation`, `BERT`, `GPT`, `Embeddings`, `Vector database`, `Fine-tuning`, and adjacent concepts.

---

## 3. System Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────┐
│         Query Processor             │  ← Query expansion / HyDE
└─────────────────┬───────────────────┘
                  │
                  ▼
┌─────────────────────────────────────┐
│       Embedding Model               │  ← fine-tuned sentence-transformers
│   (all-MiniLM-L6-v2 → fine-tuned)  │
└─────────────────┬───────────────────┘
                  │  dense vector
                  ▼
┌─────────────────────────────────────┐
│         ChromaDB Vector Store       │  ← top-k retrieval
└─────────────────┬───────────────────┘
                  │  retrieved chunks
                  ▼
┌─────────────────────────────────────┐
│        Re-ranker (cross-encoder)    │  ← ms-marco-MiniLM-L-6-v2
└─────────────────┬───────────────────┘
                  │  top-3 ranked chunks
                  ▼
┌─────────────────────────────────────┐
│         Prompt Constructor          │  ← context + query assembly
└─────────────────┬───────────────────┘
                  │
                  ▼
┌─────────────────────────────────────┐
│    LLM: Groq API (llama-3.1-8b)    │  ← free tier, fast inference
└─────────────────┬───────────────────┘
                  │
                  ▼
┌─────────────────────────────────────┐
│   Answer + Source Citations         │
└─────────────────────────────────────┘
```

---

## 4. Technology Stack (100% Free)

| Component | Tool | Reason |
|---|---|---|
| **LLM** | Groq API (`llama-3.1-8b-instant`) | Free tier, 30 req/min, very fast |
| **Fallback LLM** | Ollama (`mistral:7b`) | Fully local, no API needed |
| **Embeddings** | `sentence-transformers/all-MiniLM-L6-v2` | Fast, lightweight, good baseline |
| **Fine-tuned Embeddings** | `sentence-transformers` + `MultipleNegativesRankingLoss` | Domain-adapted |
| **Re-ranker** | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Free, local |
| **Vector Store** | ChromaDB (local persistent) | Free, no infra needed |
| **Framework** | LangChain | Modular, well-documented |
| **Evaluation** | RAGAS | Purpose-built RAG evaluation |
| **Knowledge Base Fetch** | `wikipedia-api` | Free, no auth |
| **Data Processing** | `pandas`, `tiktoken` | Standard |

---

## 5. Chunking Strategy

**Strategy: Recursive Semantic Chunking with Overlap**

```python
chunk_size    = 512 tokens   # fits within most embedding model context windows
chunk_overlap = 64 tokens    # prevents loss of context at boundaries
splitter      = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", ". ", " "],
    chunk_size=512,
    chunk_overlap=64
)
```

**Why?**
- Wikipedia articles use headings and paragraphs — recursive splitting respects natural boundaries
- 512 tokens is the sweet spot: not too large (reduces precision) nor too small (loses context)
- 64-token overlap ensures sentences split across chunk boundaries are still retrievable

**Metadata per chunk:**
```json
{
  "source": "Transformer (machine learning model)",
  "url": "https://en.wikipedia.org/wiki/...",
  "section": "Attention mechanism",
  "chunk_id": "transformer_001",
  "token_count": 487
}
```

---

## 6. Embedding Model

**Baseline:** `sentence-transformers/all-MiniLM-L6-v2`  
**Fine-tuned:** Same architecture, retrained with domain-specific triplets (see Section 4 of write-up)

Embeddings are stored in ChromaDB with L2 distance (inner product after normalization).

---

## 7. Retrieval Pipeline

1. **Query Embedding** — embed user query with the same fine-tuned model
2. **Top-K Dense Retrieval** — retrieve top-20 candidates from ChromaDB
3. **Cross-Encoder Re-ranking** — rerank top-20, keep top-3
4. **Context Assembly** — concatenate top-3 chunks with metadata headers

**Optional Enhancement — HyDE (Hypothetical Document Embedding):**
Generate a hypothetical answer with the LLM first, embed that, then retrieve. This bridges the query-document semantic gap.

---

## 8. Prompt Construction

```python
SYSTEM_PROMPT = """You are a precise AI assistant that answers questions strictly based 
on the provided context. If the answer is not in the context, say 
"I don't have enough information to answer that." Do NOT hallucinate.
Always cite the source article you used."""

USER_PROMPT = """
Context:
---
{context}
---

Question: {question}

Answer (with source citation):"""
```

**Why this prompt?**
- Forces grounding ("strictly based on context")
- Explicit hallucination prevention clause
- Source citation baked in — enables faithfulness evaluation

---

## 9. Fine-Tuning Strategy

### 9a. Fine-Tuning the Retriever (Embedding Model)

**Training objective:** `MultipleNegativesRankingLoss` (contrastive learning)

**Data generation from Wikipedia corpus:**
- Positive pairs: (section heading, section body) — natural anchor-positive
- Hard negatives: (question, wrong-topic chunk) mined via BM25 near-misses

**Steps:**
1. Auto-generate questions from chunks using the LLM (Groq)
2. Form triplets: `(query, positive_chunk, hard_negative_chunk)`
3. Fine-tune with `sentence-transformers` Trainer for 3 epochs
4. Evaluate with MRR@10 and nDCG@10 on a held-out set

### 9b. Fine-Tuning the Generator (LLM)

Since we use Groq (API-only), we instead apply:
- **Prompt tuning** — craft few-shot examples of correct QA pairs
- **Output constraints** — JSON-structured responses for consistency

If running Ollama locally, we can do LoRA fine-tuning with `llama-factory` on QA pairs derived from the corpus.

---

## 10. Bias Detection & Mitigation

**Sources of Bias:**
| Source | Type | Detection | Mitigation |
|---|---|---|---|
| Corpus | Recency / coverage bias (Wikipedia skews to popular topics) | Measure topic distribution via LDA | Balance corpus sampling |
| Embeddings | Semantic bias (MiniLM trained on web text) | Cosine similarity audit across topic clusters | Domain fine-tuning |
| Generator | Verbosity / positional bias | Compare answer length across query types | Temperature control + calibrated prompts |

**Detection tool:** RAGAS `answer_relevance` + custom token-level analysis  
**Mitigation:** Corpus rebalancing, debiased prompt templates, retrieval diversity (MMR)

---

## 11. Evaluation Framework

| Metric | Tool | Target |
|---|---|---|
| Retrieval Recall@3 | Custom + RAGAS | > 0.75 |
| MRR@10 | Custom | > 0.70 |
| Answer Faithfulness | RAGAS | > 0.80 |
| Answer Relevance | RAGAS | > 0.75 |
| Hallucination Rate | NLI model (deberta) | < 10% |
| Context Precision | RAGAS | > 0.70 |

---

## 12. Project Structure

```
rag-system/
├── data/
│   ├── raw/                  # Raw Wikipedia articles (.json)
│   └── processed/            # Chunked documents (.jsonl)
├── embeddings/
│   ├── baseline/             # Off-the-shelf model cache
│   └── finetuned/            # Fine-tuned checkpoint
├── vectorstore/
│   └── chroma_db/            # Persistent ChromaDB storage
├── src/
│   ├── ingest.py             # Wikipedia fetch + chunking
│   ├── embed.py              # Embedding + indexing
│   ├── retrieve.py           # Retriever + re-ranker
│   ├── generate.py           # LLM prompt + generation
│   ├── pipeline.py           # End-to-end RAG pipeline
│   ├── finetune_embeddings.py# Embedding fine-tuning
│   ├── bias_audit.py         # Bias detection
│   └── evaluate.py           # RAGAS evaluation runner
├── notebooks/
│   ├── 01_exploration.ipynb
│   ├── 02_finetuning.ipynb
│   └── 03_evaluation.ipynb
├── tests/
│   └── test_pipeline.py
├── app.py                    # Gradio/Streamlit demo UI
├── requirements.txt
├── .env.example
└── README.md
```

---

## 13. Trade-offs & Design Decisions

| Decision | Alternative | Why We Chose This |
|---|---|---|
| ChromaDB over FAISS | FAISS is faster at scale | ChromaDB has metadata filtering + persistence out of the box |
| Groq over OpenAI | OpenAI costs money | Groq free tier is generous (30k tokens/min) |
| Cross-encoder re-ranker | Pure dense retrieval | Re-ranking significantly improves precision at cost of latency (~100ms) |
| Wikipedia over arXiv | arXiv more technical | Wikipedia easier to parse, better structured, broader |
| RAGAS for eval | Manual eval | RAGAS automates faithfulness + relevance with LLM-as-judge |

---

*Document version: 1.0 | Last updated: March 14, 2026*
