import pytest

from evaluate import check_context_precision, parse_score


@pytest.mark.parametrize("reply,expected", [
    ("0.85", 0.85), ("Score: 1.0", 1.0), ("1", 1.0), (".7", 0.7), ("0", 0.0),
    ("I'd say 8", 1.0), ("no idea", 0.0), ("", 0.0),
])
def test_parse_score(reply, expected):
    assert parse_score(reply) == pytest.approx(expected)


def test_context_precision_is_keyword_recall():
    chunks = [{"text": "Backpropagation computes the gradient using the chain rule."}]
    gt = "Backpropagation computes gradient via chain rule weights"
    # keywords > 4 chars: backpropagation, computes, gradient, chain, weights -> 4 of 5 present
    assert check_context_precision(gt, chunks) == pytest.approx(4 / 5)


def test_context_precision_empty_ground_truth():
    assert check_context_precision("a an the", [{"text": "x"}]) == 0.0


def test_evaluate_retrieval_compares_dense_and_reranked(chroma_client, collection_name):
    from conftest import FakeEmbedder, FakeReranker, index_docs
    from evaluate import evaluate_retrieval
    from retrieve import Retriever

    index_docs(chroma_client, collection_name, [
        ("1", "backpropagation computes the gradient with the chain rule", "Backpropagation"),
        ("2", "a perceptron is a linear classifier", "Perceptron"),
    ])
    retriever = Retriever(collection_name=collection_name, embedding_model=FakeEmbedder(),
                          reranker=FakeReranker(), client=chroma_client)
    rows = evaluate_retrieval(retriever, [{"question": "backpropagation gradient",
                                           "ground_truth": "backpropagation computes gradient"}], k=1)
    assert rows[0]["reranked_sources"] == ["Backpropagation"]
    assert rows[0]["reranked_context_precision"] == 1.0
