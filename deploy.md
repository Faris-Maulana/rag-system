# Deployment Guide — RAG System
**Project:** RAG System for AI/ML Wikipedia Knowledge Base  

This guide covers how to run the system locally for the demo, and optionally how to deploy it publicly for free using Hugging Face Spaces.

---

## 1. Local Deployment (Recommended for Demo)

### Prerequisites
- Python 3.10+
- Git
- 4GB+ RAM (8GB recommended for fine-tuning)
- Groq API key (free at [console.groq.com](https://console.groq.com))

---

### Step 1 — Clone & Setup
```bash
git clone <your-repo-url>
cd rag-system

# Create virtual environment
python -m venv .venv
source .venv/bin/activate        # Linux/Mac
# OR
.\.venv\Scripts\activate          # Windows

# Install dependencies
pip install -r requirements.txt
```

---

### Step 2 — Configure Environment
```bash
# Copy the example env file
cp .env.example .env

# Edit .env and add your Groq API key
nano .env
```

`.env` contents:
```bash
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
CHROMA_PERSIST_DIR=./vectorstore/chroma_db
EMBEDDING_MODEL=./embeddings/finetuned   # or all-MiniLM-L6-v2 for baseline
LOG_LEVEL=INFO
```

---

### Step 3 — Build the Knowledge Base (One-time)
```bash
# Step 3a: Fetch Wikipedia articles and chunk them
python src/ingest.py
# Expected output: data/processed/chunks.jsonl (~1000+ chunks)

# Step 3b: Embed and index into ChromaDB
python src/embed.py
# Expected output: vectorstore/chroma_db/ populated
# Expected time: ~5-10 minutes on CPU
```

---

### Step 4 — (Optional) Fine-Tune Embedding Model
```bash
# This generates QA pairs, trains embedding model, and re-indexes
python src/finetune_embeddings.py

# After fine-tuning, re-index with new embeddings
python src/embed.py --model embeddings/finetuned
```
> ⚠️ Skip this step for the demo if short on time. Use baseline embeddings.

---

### Step 5 — Run the Demo App
```bash
python app.py
```

Expected output:
```
Running on local URL:  http://127.0.0.1:7860
Running on public URL: https://xxxxxxxx.gradio.live
```

Open `http://localhost:7860` in your browser.

---

### Step 6 — Run Evaluation
```bash
python src/evaluate.py
# Results saved to: results/evaluation_report.json
```

---

## 2. Full Pipeline Command Reference

```bash
# Run ingestion only
python src/ingest.py

# Run embedding + indexing
python src/embed.py

# Test retrieval (interactive)
python src/retrieve.py --query "What is attention mechanism?"

# Test full RAG pipeline (single query)
python src/pipeline.py --query "How does BERT differ from GPT?"

# Run bias audit
python src/bias_audit.py

# Run full RAGAS evaluation
python src/evaluate.py

# Launch Gradio UI
python app.py
```

---

## 3. Project File Verification Checklist

Before the demo, verify these files exist:
```bash
# Check all key files are present
ls -la data/processed/chunks.jsonl           # chunked docs
ls -la vectorstore/chroma_db/                # vector index
ls -la embeddings/finetuned/                 # fine-tuned model (optional)
ls -la results/evaluation_report.json        # eval results
ls -la logs/pipeline_log.jsonl               # run logs
```

---

## 4. Public Deployment — Hugging Face Spaces (Free)

If you want a public shareable URL for the demo:

### Step 1 — Create a Hugging Face account
Go to [huggingface.co](https://huggingface.co) → Sign up (free)

### Step 2 — Create a new Space
- Go to `huggingface.co/new-space`
- SDK: **Gradio**
- Visibility: **Public**
- Space name: `rag-wikipedia-demo`

### Step 3 — Add your Groq API key as a Secret
- Go to Space Settings → Variables and Secrets
- Add: `GROQ_API_KEY` = your key

### Step 4 — Push code to the Space
```bash
# In your project directory
git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/rag-wikipedia-demo
git add .
git commit -m "Initial RAG system deployment"
git push hf main
```

### Step 5 — Pre-build the vector index
Since HF Spaces doesn't persist files between restarts, include a setup script:

```python
# In app.py — add at the top
import os

if not os.path.exists("./vectorstore/chroma_db"):
    print("Building knowledge base on first run...")
    os.system("python src/ingest.py && python src/embed.py")
```

> ⚠️ First cold start on HF Spaces will take 5–10 minutes to build the index.

---

## 5. Docker Deployment (Optional, Local)

For a more reproducible local demo:

### `Dockerfile`
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Build knowledge base at container build time
RUN python src/ingest.py && python src/embed.py

EXPOSE 7860

CMD ["python", "app.py", "--server_name", "0.0.0.0", "--server_port", "7860"]
```

### Build & Run
```bash
docker build -t rag-system .
docker run -p 7860:7860 -e GROQ_API_KEY=your_key rag-system
```

Access at: `http://localhost:7860`

---

## 6. Demo Preparation for Monday's Presentation

### Before the call:
- [ ] System is running locally at `localhost:7860`
- [ ] 5 prepared demo queries ready (see below)
- [ ] Evaluation report printed/visible
- [ ] Code editor open with key files: `pipeline.py`, `finetune_embeddings.py`, `evaluate.py`

### Suggested Demo Queries:
```
1. "What is the attention mechanism in transformers?"
   → Tests: core domain retrieval

2. "How does BERT differ from GPT in training objective?"
   → Tests: comparative reasoning

3. "What is retrieval-augmented generation?"
   → Tests: meta-domain query (RAG about RAG)

4. "Who invented the transformer architecture?"
   → Tests: factual grounding, hallucination resistance

5. "What are the limitations of large language models?"
   → Tests: nuanced multi-chunk retrieval
```

### Talking Points for Each Section:
| Section | What to Show |
|---|---|
| **Build the Pipeline** | Walk through `pipeline.py` → show live query → show retrieved chunks + answer |
| **Fine-Tune RAG** | Show MRR@10 before vs after fine-tuning in `results/` |
| **Reduce Bias** | Show `bias_audit.py` output → explain MMR fix |
| **Fine-Tune Embeddings** | Show `finetune_embeddings.py` training loop → explain triplet loss |
| **Evaluate Correctness** | Show RAGAS scores in `evaluation_report.json` |

---

## 7. Troubleshooting at Demo Time

| Problem | Quick Fix |
|---|---|
| Groq API down | `export USE_OLLAMA=true` → switch to local Ollama |
| ChromaDB empty | `python src/embed.py` (pre-run this the night before) |
| App won't start | Check `.env` has correct `GROQ_API_KEY` |
| Slow response (>10s) | Reduce `n_results` from 20 to 5 in `retrieve.py` |
| Port 7860 in use | `python app.py --server_port 7861` |

---

## 8. Repository Structure for Submission

```
rag-system/
├── src/                    ← all Python modules
├── data/processed/         ← chunked Wikipedia data
├── vectorstore/            ← ChromaDB index (gitignored for size)
├── embeddings/finetuned/   ← fine-tuned model (gitignored for size)
├── results/                ← evaluation results JSON
├── notebooks/              ← exploratory notebooks
├── app.py                  ← Gradio demo app
├── requirements.txt
├── .env.example            ← template (never commit .env)
├── Dockerfile              ← optional
└── README.md               ← setup + usage instructions
```

**`.gitignore` entries:**
```
.env
vectorstore/
embeddings/finetuned/
__pycache__/
*.pyc
.venv/
```

---

*Deployment guide version: 1.0 | Last updated: March 14, 2026*
