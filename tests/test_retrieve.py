import numpy as np

from conftest import FakeEmbedder, FakeReranker, index_docs
from retrieve import Retriever, mmr_select

DOCS = [
    ("a1", "BERT is a bidirectional encoder language model from Google", "BERT"),
    ("a2", "BERT is a bidirectional encoder language model from Google", "BERT (copy)"),
    ("b1", "GPT is an autoregressive decoder language model", "GPT"),
    ("c1", "Random forests average many decision trees", "Random forest"),
    ("d1", "Gradient descent minimises a loss function", "Gradient descent"),
]


def make_retriever(client, name):
    index_docs(client, name, DOCS)
    return Retriever(collection_name=name, embedding_model=FakeEmbedder(), reranker=FakeReranker(), client=client)


def test_mmr_lambda_one_is_plain_similarity():
    q = np.array([1.0, 0.0])
    docs = np.array([[0.9, 0.1], [1.0, 0.0], [0.0, 1.0]])
    assert mmr_select(q, docs, k=3, lambda_mult=1.0) == [1, 0, 2]


def test_mmr_prefers_novel_documents():
    q = np.array([1.0, 0.0])
    near_duplicates = np.array([[1.0, 0.0], [0.99, 0.01], [0.7, 0.7]])
    # Pure similarity would take the near-duplicate second; MMR takes the diverse one.
    assert mmr_select(q, near_duplicates, k=2, lambda_mult=0.3) == [0, 2]


def test_mmr_handles_empty_and_small_pools():
    assert mmr_select([1.0, 0.0], [], k=3) == []
    assert mmr_select([1.0, 0.0], [[1.0, 0.0]], k=3) == [0]


def test_retrieve_reranks_and_drops_duplicate_passages(chroma_client, collection_name):
    retriever = make_retriever(chroma_client, collection_name)
    results = retriever.retrieve("bidirectional encoder BERT", top_k_dense=5, top_k_rerank=3)

    texts = [r["text"] for r in results]
    assert len(texts) == len(set(texts)), "identical passages must not occupy two slots"
    assert results[0]["source"].startswith("BERT")
    assert [r["score"] for r in results] == sorted((r["score"] for r in results), reverse=True)
    assert all(0.0 <= r["dense_distance"] <= 2.0 for r in results)


def test_retrieve_on_empty_collection_returns_nothing(chroma_client, collection_name):
    retriever = Retriever(collection_name=collection_name, embedding_model=FakeEmbedder(),
                          reranker=FakeReranker(), client=chroma_client)
    assert retriever.retrieve("anything") == []


def test_retrieve_mmr_returns_k_unique_passages(chroma_client, collection_name):
    retriever = make_retriever(chroma_client, collection_name)
    results = retriever.retrieve_mmr("language model", k=3, fetch_k=5)
    assert len(results) == 3
    assert len({r["text"] for r in results}) == 3
