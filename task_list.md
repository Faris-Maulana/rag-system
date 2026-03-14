# Task List — RAG System Build
**Project:** RAG System for AI/ML Wikipedia Knowledge Base  
**Deadline:** Sunday night (before Monday 1 PM WIB presentation)

---

## Legend
- `[ ]` Not started
- `[~]` In progress
- `[x]` Done
- `[!]` Blocked / Needs review

---

## Phase 0 — Environment Setup

- [ ] **T-00** Create project directory structure
  ```bash
  mkdir -p rag-system/{data/{raw,processed},embeddings/{baseline,finetuned},vectorstore/chroma_db,src,notebooks,tests}
  cd rag-system && git init
  ```

- [ ] **T-01** Create and activate virtual environment
  ```bash
  python -m venv .venv && source .venv/bin/activate
  ```

- [ ] **T-02** Install all dependencies
  ```bash
  pip install langchain langchain-community chromadb sentence-transformers \
    groq wikipedia-api tiktoken ragas pandas tqdm gradio \
    transformers datasets torch accelerate
  ```

- [ ] **T-03** Create `.env` file and `.env.example`
  ```
  GROQ_API_KEY=your_groq_api_key_here
  CHROMA_PERSIST_DIR=./vectorstore/chroma_db
  ```

- [ ] **T-04** Register Groq free account → get API key at console.groq.com

- [ ] **T-05** Create `requirements.txt`
  ```bash
  pip freeze > requirements.txt
  ```

---

## Phase 1 — Data Ingestion

- [ ] **T-10** Write `src/ingest.py` — Wikipedia article fetcher
  - Fetch 50+ articles on AI/ML topics via `wikipedia-api`
  - Save raw JSON: `{title, url, content, sections}`
  - Topics list: ML, DL, Transformers, LLMs, RAG, BERT, GPT, etc.

- [ ] **T-11** Write chunking logic inside `ingest.py`
  - Use `RecursiveCharacterTextSplitter(chunk_size=512, overlap=64)`
  - Add metadata: source, url, section, chunk_id, token_count
  - Save chunked docs to `data/processed/chunks.jsonl`

- [ ] **T-12** Run ingestion pipeline
  ```bash
  python src/ingest.py
  ```

- [ ] **T-13** Validate ingestion output
  - Assert > 1000 chunks generated
  - Check no empty chunks
  - Print stats: total chunks, avg token count, source distribution

---

## Phase 2 — Embedding & Indexing

- [ ] **T-20** Write `src/embed.py` — embedding + ChromaDB indexing
  - Load sentence-transformers model (`all-MiniLM-L6-v2`)
  - Batch embed all chunks (batch_size=64)
  - Upsert into ChromaDB with metadata

- [ ] **T-21** Run embedding pipeline
  ```bash
  python src/embed.py
  ```

- [ ] **T-22** Validate ChromaDB index
  - Run a test query: "What is a transformer model?"
  - Assert top-5 results are semantically relevant
  - Check metadata is stored correctly

---

## Phase 3 — Retrieval Pipeline

- [ ] **T-30** Write `src/retrieve.py`
  - ChromaDB dense retrieval (top-20)
  - Cross-encoder re-ranker (`ms-marco-MiniLM-L-6-v2`) → top-3
  - Return ranked chunks + metadata

- [ ] **T-31** (Optional) Implement HyDE retrieval
  - Generate hypothetical answer with LLM
  - Embed hypothetical answer → retrieve

- [ ] **T-32** Test retriever with 10 sample queries
  - Manually verify retrieved chunks are relevant
  - Log precision@3 informally

---

## Phase 4 — Generation Pipeline

- [ ] **T-40** Write `src/generate.py`
  - Initialize Groq client (`llama-3.1-8b-instant`)
  - Define system + user prompt templates
  - Add fallback to Ollama if Groq fails

- [ ] **T-41** Write `src/pipeline.py` — end-to-end RAG
  - `query → retrieve → rerank → prompt → generate → answer`
  - Return: `{answer, sources, context_chunks}`

- [ ] **T-42** Run 20 test queries end-to-end
  - Log all inputs/outputs to `logs/test_run.jsonl`
  - Verify answers are grounded in retrieved context

---

## Phase 5 — Fine-Tuning Embeddings

- [ ] **T-50** Write `src/finetune_embeddings.py`
  - Step 1: Auto-generate 500 QA pairs from chunks using Groq
  - Step 2: Mine hard negatives with BM25 (`rank_bm25`)
  - Step 3: Format as `InputExample(texts=[query, positive, negative])`
  - Step 4: Train with `MultipleNegativesRankingLoss` for 3 epochs
  - Save fine-tuned model to `embeddings/finetuned/`

- [ ] **T-51** Run fine-tuning
  ```bash
  python src/finetune_embeddings.py
  ```
  *(Expected: 30–60 minutes on CPU, 10 mins on GPU)*

- [ ] **T-52** Re-index ChromaDB with fine-tuned embeddings
  - Update `embed.py` to load from `embeddings/finetuned/`
  - Re-run embedding pipeline

- [ ] **T-53** Compare baseline vs fine-tuned retrieval
  - Run same 20 queries on both
  - Compute MRR@10 before/after
  - Document improvement in write-up

---

## Phase 6 — Bias Audit

- [ ] **T-60** Write `src/bias_audit.py`
  - Analyze topic distribution with `collections.Counter`
  - Check answer length distribution by query type
  - Check cosine similarity spread across topic clusters

- [ ] **T-61** Run bias audit
  ```bash
  python src/bias_audit.py
  ```

- [ ] **T-62** Apply MMR (Maximal Marginal Relevance) in retrieval
  - Replace simple top-k with MMR to improve diversity
  - `chroma.max_marginal_relevance_search(query, k=3, fetch_k=20)`

- [ ] **T-63** Document bias findings and mitigations in write-up

---

## Phase 7 — Evaluation

- [ ] **T-70** Prepare evaluation dataset
  - Generate 50 QA pairs: `{question, ground_truth_answer, context}`
  - Mix easy, medium, and domain-specific queries

- [ ] **T-71** Write `src/evaluate.py`
  - Run RAGAS: `faithfulness`, `answer_relevance`, `context_precision`, `context_recall`
  - Custom: Recall@3, MRR@10
  - Custom: Hallucination check via `deberta-v3` NLI model

- [ ] **T-72** Run full evaluation
  ```bash
  python src/evaluate.py
  ```

- [ ] **T-73** Generate evaluation report
  - Save results to `results/evaluation_report.json`
  - Create summary table for write-up

---

## Phase 8 — Demo UI

- [ ] **T-80** Write `app.py` — Gradio UI
  - Text input for user query
  - Display answer + retrieved sources
  - Show RAGAS scores per query (optional)

- [ ] **T-81** Run and test the demo
  ```bash
  python app.py
  ```

---

## Phase 9 — Write-Up & Documentation

- [ ] **T-90** Write `README.md`
  - Setup instructions
  - How to run each component
  - Architecture diagram (ASCII)

- [ ] **T-91** Write technical write-up document
  - Decisions and trade-offs for each of the 5 challenge sections
  - Include evaluation results table
  - Include before/after fine-tuning comparison

- [ ] **T-92** Final code review
  - Add docstrings to all functions
  - Remove debug prints
  - Make sure `.env` keys are not committed

---

## Priority Order (If Time Is Short)

| Priority | Tasks | Why |
|---|---|---|
| 🔴 MUST | T-00 to T-42 | Core working pipeline — non-negotiable |
| 🟡 SHOULD | T-50 to T-53 | Fine-tuning shows depth — high interview impact |
| 🟡 SHOULD | T-70 to T-73 | Evaluation proves it works — high impact |
| 🟢 NICE | T-60 to T-63 | Bias section shows maturity |
| 🟢 NICE | T-80 to T-81 | UI makes demo better but not critical |

---

## Time Estimate

| Phase | Estimated Time |
|---|---|
| Environment Setup | 30 min |
| Data Ingestion | 1 hour |
| Embedding & Indexing | 1 hour |
| Retrieval Pipeline | 1.5 hours |
| Generation Pipeline | 1.5 hours |
| Embedding Fine-tuning | 2–3 hours |
| Bias Audit | 1 hour |
| Evaluation | 2 hours |
| Demo UI | 1 hour |
| Write-up | 2 hours |
| **Total** | **~15 hours** |

---

*Task list version: 1.0 | Last updated: March 14, 2026*
