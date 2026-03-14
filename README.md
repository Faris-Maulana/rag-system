# RAG System for Wikipedia Knowledge Base

A production-grade Retrieval-Augmented Generation (RAG) system built over 50+ Wikipedia AI/ML articles. Developed using Python, LangChain, ChromaDB, Sentence-Transformers, and the Groq API (Llama-3.1-8b).

## Setup & Execution

### 1. Prerequisites
- Python 3.10+
- Groq API Key (Free tier at console.groq.com)

### 2. Installation
```bash
# Clone the repository
git clone <repository_url>
cd rag-system

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .\.venv\Scripts\activate  # Windows

# Install Dependencies
pip install -r requirements.txt
```

### 3. Environment Configuration
Create a `.env` file in the root directory and add your keys:
```env
GROQ_API_KEY=your_actual_key_here
CHROMA_PERSIST_DIR=./vectorstore/chroma_db
```

### 4. Running the Pipeline Step-by-Step

**Step 1: Data Ingestion**
Fetches Wikipedia articles and chunks them using RecursiveCharacterTextSplitter.
```bash
python src/ingest.py
```

**Step 2: Embedding & Indexing**
Generates embeddings using `all-MiniLM-L6-v2` and stores them in ChromaDB.
```bash
python src/embed.py
```

**Step 3: Test Retrieval**
Tests the Dense Retriever + Cross-Encoder Reranker pipeline.
```bash
python src/retrieve.py --query "What is a transformer model?"
```

**Step 4: Full Pipeline Test**
Tests the end-to-end RAG system from Query -> Retrieve -> Rerank -> Generate.
```bash
python src/pipeline.py --query "How does BERT differ from GPT?"
```

### 5. Advanced Features

**Bias Audit (MMR Retrieval)**
Evaluates chunk lengths and topic distributions, and demonstrates MMR diversity retrieval to mitigate coverage bias.
```bash
python src/bias_audit.py
```

**Fine-Tuning Embeddings**
Generates synthetic QA pairs and fine-tunes the embedding model using contrastive triplet loss to improve domain-specific retrieval metrics (MRR).
```bash
python src/finetune_embeddings.py --samples 100 --epochs 2
```

**RAGAS Evaluation**
Evaluates the RAG pipeline generating comprehensive reports for Faithfulness, Context Precision, and Answer Relevancy.
```bash
python src/evaluate.py --num_samples 20
```

### 6. Interactive Demo UI
Launch the interactive Gradio web interface to test queries live.
```bash
python app.py
```

## System Architecture

```text
User Query --> Embedding Model --> ChromaDB (Dense Retrieval Top-20)
                                        |
       Cross-Encoder Reranker (Top-3) <-+
               |
        Prompt Assembly
               |
    Groq LLM (llama-3.1-8b)
               |
    Answer + Grounded Sources
```
