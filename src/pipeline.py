import argparse
import json
import os
from datetime import datetime

from config import GROQ_MODEL, LOG_FILE
from generate import Generator
from retrieve import Retriever


class RAGPipeline:
    def __init__(self, retriever=None, generator=None, log_file=LOG_FILE):
        print("Initializing RAG Pipeline...")
        self.retriever = retriever or Retriever()
        self.generator = generator or Generator()

        self.log_file = log_file
        if self.log_file:
            os.makedirs(os.path.dirname(self.log_file) or ".", exist_ok=True)

    def query(self, question: str) -> dict:
        """Run the end-to-end RAG pipeline from query to generation."""
        # Step 1: Retrieval
        print(f"\n[Retrieving context for query]: {question}")
        retrieved_candidates = self.retriever.retrieve(question)

        if not retrieved_candidates:
            return {
                "answer": "I don't have enough information to answer that.",
                "sources": [],
                "context_chunks": []
            }

        # Format contexts (extract raw strings)
        context_chunks = [chunk["text"] for chunk in retrieved_candidates]

        # Deduplicate sources for output
        sources = []
        seen_urls = set()
        for chunk in retrieved_candidates:
            if chunk["url"] not in seen_urls:
                sources.append({
                    "source": chunk["source"],
                    "url": chunk["url"],
                    "section": chunk["section"]
                })
                seen_urls.add(chunk["url"])

        # Step 2: Generation
        print(f"[Generating answer with {GROQ_MODEL}]")
        answer = self.generator.generate(question, context_chunks)

        # Format response
        result = {
            "answer": answer,
            "sources": sources,
            "context_chunks": retrieved_candidates
        }

        # Log to file
        self._log_run(question, retrieved_candidates, answer)

        return result

    def _log_run(self, query, retrieved, answer):
        """Log queries and results for debugging and analysis."""
        if not self.log_file:
            return
        with open(self.log_file, "a", encoding="utf-8", errors="replace") as f:
            f.write(json.dumps({
                "timestamp": datetime.now().isoformat(),
                "query": query,
                "retrieved_chunks": retrieved,
                "answer": answer
            }, ensure_ascii=False) + "\n")

def main():
    parser = argparse.ArgumentParser(description="End-to-End RAG Pipeline Demo")
    parser.add_argument("--query", type=str, default="What is a transformer model?", help="Test query")
    args = parser.parse_args()

    rag = RAGPipeline()
    result = rag.query(args.query)

    print("\n" + "="*50)
    print("FINAL ANSWER:")
    print("="*50)
    print(result["answer"])

    print("\n" + "="*50)
    print("SOURCES USED:")
    print("="*50)
    for source in result["sources"]:
        print(f"- {source['source']} ({source['section']}): {source['url']}")

if __name__ == '__main__':
    main()
