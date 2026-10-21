from finetune_embeddings import mine_hard_negatives, split_qa


def qa(n):
    return [{"question": f"q{i}", "positive_context": f"context number {i}"} for i in range(n)]


def test_split_qa_is_deterministic_and_disjoint():
    train, test = split_qa(qa(10), test_fraction=0.2)
    train2, test2 = split_qa(qa(10), test_fraction=0.2)
    assert (train, test) == (train2, test2)
    assert len(test) == 2 and len(train) == 8
    assert not {q["question"] for q in train} & {q["question"] for q in test}


def test_hard_negative_excludes_positive():
    corpus = [{"text": "transformers use self attention over token sequences " * 3},
              {"text": "attention heads in transformers compute weighted sums " * 3}]
    data = [{"question": "how do transformers use attention", "positive_context": corpus[0]["text"]}]
    (example,) = mine_hard_negatives(data, corpus)
    query, positive, negative = example.texts
    assert negative != positive
    assert negative == corpus[1]["text"]


def test_hard_negative_falls_back_to_random_chunk():
    # Every candidate is either the positive or too short, so the random fallback path runs.
    corpus = [{"text": "short"}, {"text": "tiny"}]
    data = [{"question": "short", "positive_context": "short"}]
    (example,) = mine_hard_negatives(data, corpus)
    assert example.texts[2] in {"short", "tiny"}
