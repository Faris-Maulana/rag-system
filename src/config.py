"""Runtime settings, read once from the environment (and `.env` if present)."""
import os

from dotenv import load_dotenv

load_dotenv()

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./vectorstore/chroma_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "rag_wikipedia")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
CHUNKS_PATH = os.getenv("CHUNKS_PATH", "data/processed/chunks.jsonl")
LOG_FILE = os.getenv("PIPELINE_LOG_FILE", "logs/pipeline_log.jsonl")
