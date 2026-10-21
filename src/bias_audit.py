import json
import os
from collections import Counter

from config import CHROMA_PERSIST_DIR, CHUNKS_PATH


def load_chunks(filepath=CHUNKS_PATH):
    chunks = []
    if not os.path.exists(filepath):
        print(f"Error: {filepath} not found.")
        return chunks

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))
    return chunks

def analyze_topic_distribution(chunks):
    print("\n--- Topic Distribution (Coverage Bias) ---")
    sources = [c["source"] for c in chunks]
    counts = Counter(sources)

    top_5 = counts.most_common(5)
    bottom_5 = counts.most_common()[-5:]

    print("Top 5 Represented Topics (Potential Over-representation):")
    for source, count in top_5:
        print(f"  - {source}: {count} chunks")

    print("\nBottom 5 Represented Topics (Potential Under-representation):")
    for source, count in bottom_5:
        print(f"  - {source}: {count} chunks")

    print(f"\nTotal unique topics identified: {len(counts)}")
    return counts

def analyze_chunk_lengths(chunks):
    print("\n--- Chunk Length Analysis (Positional Bias) ---")
    lengths = [c["token_count"] for c in chunks]

    avg_len = sum(lengths) / len(lengths)
    max_len = max(lengths)
    min_len = min(lengths)

    print(f"Average token count per chunk: {avg_len:.1f}")
    print(f"Maximum token count: {max_len}")
    print(f"Minimum token count: {min_len}")

    # Check for unusually short chunks which might lack context
    short_chunks = sum(1 for n in lengths if n < 50)
    print(f"Chunks with < 50 tokens (Potential context loss): {short_chunks} out of {len(chunks)} ({(short_chunks/len(chunks))*100:.1f}%)")
    return {"avg": avg_len, "max": max_len, "min": min_len, "short": short_chunks}

def test_mmr_retrieval(query="What are the ethical implications of AI?", retriever=None):
    """
    Maximal Marginal Relevance (MMR) optimizes for both similarity to the query
    AND diversity among retrieved documents. This helps mitigate coverage bias
    where all top results might come from the exact same Wikipedia article.
    """
    print("\n--- MMR Diversity vs standard Top-K Retrieval ---")
    print(f"Query: '{query}'")

    if retriever is None:
        if not os.path.exists(CHROMA_PERSIST_DIR):
            print("ChromaDB not found. Cannot run retrieval bias test.")
            return
        from retrieve import Retriever
        retriever = Retriever()

    # lambda_mult=1.0 is plain similarity ranking (may suffer from homogenization bias)
    standard_sources = [c["source"] for c in retriever.retrieve_mmr(query, k=5, fetch_k=20, lambda_mult=1.0)]
    print("\nStandard Similarity Retrieval Sources:")
    for i, src in enumerate(standard_sources):
        print(f"  {i+1}. {src}")

    # fetch_k = 20 (candidate pool), k = 5 (final selection); 0=max diversity, 1=min diversity
    mmr_sources = [c["source"] for c in retriever.retrieve_mmr(query, k=5, fetch_k=20, lambda_mult=0.5)]
    print("\nMMR Retrieval Sources (lambda_mult=0.5):")
    for i, src in enumerate(mmr_sources):
        print(f"  {i+1}. {src}")

    std_unique = len(set(standard_sources))
    mmr_unique = len(set(mmr_sources))
    print(f"\nUnique sources retrieved - Standard: {std_unique}, MMR: {mmr_unique}")
    if mmr_unique > std_unique:
        print("[SUCCESS] MMR successfully increased topic diversity in retrieval.")
    else:
        print("[INFO] MMR did not increase source diversity for this query.")
    return std_unique, mmr_unique

def generate_report():
    print("=======================================")
    print("         BIAS AUDIT REPORT")
    print("=======================================")
    chunks = load_chunks()
    if chunks:
        analyze_topic_distribution(chunks)
        analyze_chunk_lengths(chunks)
    test_mmr_retrieval()
    print("=======================================")

if __name__ == "__main__":
    generate_report()
