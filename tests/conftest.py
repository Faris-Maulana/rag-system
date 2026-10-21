import hashlib
import uuid

import chromadb
import numpy as np
import pytest

DIM = 64


def _token_vec(token):
    seed = int(hashlib.md5(token.encode()).hexdigest()[:8], 16)
    return np.random.default_rng(seed).standard_normal(DIM)


class FakeEmbedder:
    """Deterministic bag-of-words embedder: texts sharing words get similar vectors."""

    def encode(self, texts, convert_to_numpy=True, **_):
        out = []
        for text in texts:
            v = sum((_token_vec(t) for t in text.lower().split()), np.zeros(DIM))
            out.append(v / (np.linalg.norm(v) or 1.0))
        return np.array(out)


class FakeReranker:
    """Scores a (query, passage) pair by word overlap."""

    def predict(self, pairs):
        return [len(set(q.lower().split()) & set(p.lower().split())) for q, p in pairs]


@pytest.fixture
def chroma_client():
    return chromadb.EphemeralClient()


@pytest.fixture
def collection_name():
    return f"test_{uuid.uuid4().hex[:8]}"


def index_docs(client, name, docs, embedder=None):
    """docs: list of (id, text, source)."""
    embedder = embedder or FakeEmbedder()
    col = client.get_or_create_collection(name=name, metadata={"hnsw:space": "cosine"})
    col.upsert(
        ids=[d[0] for d in docs],
        documents=[d[1] for d in docs],
        embeddings=embedder.encode([d[1] for d in docs]).tolist(),
        metadatas=[{"source": d[2], "url": f"https://example.org/{d[2]}", "section": "Summary"} for d in docs],
    )
    return col
