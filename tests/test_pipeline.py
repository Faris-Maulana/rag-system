import json

from pipeline import RAGPipeline


class StubRetriever:
    def __init__(self, chunks):
        self.chunks = chunks

    def retrieve(self, question):
        return self.chunks


class StubGenerator:
    def __init__(self):
        self.seen = None

    def generate(self, question, contexts):
        self.seen = contexts
        return "answer"


def chunk(text, source, section="Summary"):
    return {"text": text, "source": source, "url": f"https://en.wikipedia.org/wiki/{source}", "section": section}


def test_query_dedupes_sources_and_logs(tmp_path):
    log = tmp_path / "logs" / "run.jsonl"
    gen = StubGenerator()
    chunks = [chunk("a", "BERT"), chunk("b", "BERT", "History"), chunk("c", "GPT")]
    rag = RAGPipeline(retriever=StubRetriever(chunks), generator=gen, log_file=str(log))

    result = rag.query("How does BERT differ from GPT?")

    assert result["answer"] == "answer"
    assert [s["source"] for s in result["sources"]] == ["BERT", "GPT"]
    assert gen.seen == ["a", "b", "c"]
    entry = json.loads(log.read_text().strip())
    assert entry["query"] == "How does BERT differ from GPT?"


def test_query_without_context_refuses_and_skips_llm():
    gen = StubGenerator()
    rag = RAGPipeline(retriever=StubRetriever([]), generator=gen, log_file=None)
    result = rag.query("Unknown?")
    assert result["answer"] == "I don't have enough information to answer that."
    assert gen.seen is None
