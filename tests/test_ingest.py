import tiktoken

from ingest import build_splitter, chunk_article, safe_filename


def test_safe_filename_strips_path_separators():
    assert safe_filename("Transformer (deep learning)") == "Transformer_(deep_learning)"
    assert "/" not in safe_filename("TCP/IP: a story")
    assert ":" not in safe_filename("TCP/IP: a story")


def test_chunk_article_ids_are_stable_and_unique():
    enc = tiktoken.get_encoding("cl100k_base")
    splitter = build_splitter(enc)
    long_text = "\n\n".join(f"Paragraph {i}. " + "attention is all you need " * 60 for i in range(6))
    sections = [{"title": "Summary", "text": "Short summary."}, {"title": "History", "text": long_text},
                {"title": "Empty", "text": "   "}]

    chunks = chunk_article("Transformer (deep learning)", "https://x", sections, splitter, enc)
    again = chunk_article("Transformer (deep learning)", "https://x", sections, splitter, enc)

    ids = [c["chunk_id"] for c in chunks]
    assert len(ids) == len(set(ids))
    assert ids == [c["chunk_id"] for c in again]
    assert ids[0] == "Transformer_(deep_learning)_0001"
    assert all(c["token_count"] <= 512 for c in chunks)
    assert {c["section"] for c in chunks} == {"Summary", "History"}
