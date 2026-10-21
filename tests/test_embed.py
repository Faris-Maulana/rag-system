import pytest

import embed
from conftest import index_docs


def test_get_collection_refuses_to_mix_embedding_models(chroma_client, monkeypatch, collection_name):
    monkeypatch.setattr(embed, "COLLECTION_NAME", collection_name)
    col = embed.get_collection(chroma_client, "model-a")
    index_docs(chroma_client, collection_name, [("x", "some text", "S")])
    assert col.count() == 1

    with pytest.raises(SystemExit, match="--reset"):
        embed.get_collection(chroma_client, "model-b")

    rebuilt = embed.get_collection(chroma_client, "model-b", reset=True)
    assert rebuilt.count() == 0
    assert rebuilt.metadata["embedding_model"] == "model-b"


def test_get_collection_same_model_is_idempotent(chroma_client, monkeypatch, collection_name):
    monkeypatch.setattr(embed, "COLLECTION_NAME", collection_name)
    embed.get_collection(chroma_client, "model-a")
    index_docs(chroma_client, collection_name, [("x", "some text", "S")])
    assert embed.get_collection(chroma_client, "model-a").count() == 1
