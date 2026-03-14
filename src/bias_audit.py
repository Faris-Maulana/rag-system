import os
import sys
import json
from collections import Counter
import chromadb
from sentence_transformers import SentenceTransformer

# Add src to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

def load_chunks(filepath="data/processed/chunks.jsonl"):
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
    
    print("Top 5 Reprepsented Topics (Potential Over-representation):")
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
    short_chunks = sum(1 for l in lengths if l < 50)
    print(f"Chunks with < 50 tokens (Potential context loss): {short_chunks} out of {len(chunks)} ({(short_chunks/len(chunks))*100:.1f}%)")

def test_mmr_retrieval(query="What are the ethical implications of AI?"):
    """
    Maximal Marginal Relevance (MMR) optimizes for both similarity to the query
    AND diversity among retrieved documents. This helps mitigate coverage bias
    where all top results might come from the exact same Wikipedia article.
    """
    print("\n--- MMR Diversity vs standard Top-K Retrieval ---")
    print(f"Query: '{query}'")
    
    # Initialize ChromaDB
    chroma_path = "./vectorstore/chroma_db"
    if not os.path.exists(chroma_path):
        print("ChromaDB not found. Cannot run retrieval bias test.")
        return
        
    client = chromadb.PersistentClient(path=chroma_path)
    # Using Langchain's Chroma wrapper to leverage its built-in MMR capabilities 
    # instead of writing a custom numpy MMR implementation.
    from langchain_community.vectorstores import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings
    
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma(client=client, collection_name="rag_wikipedia", embedding_function=embeddings)
    
    # Standard Similarity Retrieval (may suffer from homogenization bias)
    standard_docs = db.similarity_search(query, k=5)
    print("\nStandard Similarity Retrieval Sources:")
    standard_sources = []
    for i, doc in enumerate(standard_docs):
        src = doc.metadata.get('source', 'Unknown')
        standard_sources.append(src)
        print(f"  {i+1}. {src}")
        
    # MMR Retrieval (forces diversity penalty)
    # fetch_k = 20 (candidate pool), k = 5 (final selection) 
    # lambda_mult controls diversity. 0=max diversity, 1=min diversity (standard similarity)
    mmr_docs = db.max_marginal_relevance_search(query, k=5, fetch_k=20, lambda_mult=0.5)
    print("\nMMR Retrieval Sources (lambda_mult=0.5):")
    mmr_sources = []
    for i, doc in enumerate(mmr_docs):
        src = doc.metadata.get('source', 'Unknown')
        mmr_sources.append(src)
        print(f"  {i+1}. {src}")
        
    std_unique = len(set(standard_sources))
    mmr_unique = len(set(mmr_sources))
    print(f"\nUnique sources retrieved - Standard: {std_unique}, MMR: {mmr_unique}")
    if mmr_unique > std_unique:
        print("[SUCCESS] MMR successfully increased topic diversity in retrieval.")
    else:
        print("[INFO] Standard retrieval already produced diverse results for this query.")

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
