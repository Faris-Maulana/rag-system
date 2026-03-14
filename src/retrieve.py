import argparse
import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder

class Retriever:
    def __init__(self, 
                 embedding_model_name="all-MiniLM-L6-v2", 
                 reranker_model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
                 chroma_path="./vectorstore/chroma_db",
                 collection_name="rag_wikipedia"):
        
        print(f"Loading embedding model: {embedding_model_name}")
        self.embedding_model = SentenceTransformer(embedding_model_name)
        
        print(f"Loading re-ranker model: {reranker_model_name}")
        self.reranker = CrossEncoder(reranker_model_name)
        
        print(f"Connecting to ChromaDB at: {chroma_path}")
        self.client = chromadb.PersistentClient(path=chroma_path)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def retrieve(self, query: str, top_k_dense=20, top_k_rerank=3):
        """
        Retrieves top relevant chunks for a given query.
        1. Dense retrieval using ChromaDB (top 20)
        2. Re-ranking using Cross-Encoder (top 3)
        """
        # 1. Embed query
        query_emb = self.embedding_model.encode([query], convert_to_numpy=True).tolist()
        
        # 2. Dense retrieval from ChromaDB
        results = self.collection.query(
            query_embeddings=query_emb,
            n_results=top_k_dense
        )
        
        if not results['documents'] or not results['documents'][0]:
            return []
            
        # Reformat ChromaDB results into a candidate list
        candidates = []
        for i in range(len(results['documents'][0])):
            candidates.append({
                "text": results['documents'][0][i],
                "source": results['metadatas'][0][i]["source"],
                "url": results['metadatas'][0][i]["url"],
                "section": results['metadatas'][0][i].get("section", ""),
                "dense_score": results['distances'][0][i] if 'distances' in results and results['distances'] else 0.0
            })
            
        # 3. Cross-encoder re-ranking
        # Pair format expected by predict: [(query, doc), ...]
        pairs = [(query, chunk["text"]) for chunk in candidates]
        scores = self.reranker.predict(pairs)
        
        # Combine scores and candidates
        for i, score in enumerate(scores):
            candidates[i]["score"] = float(score)
            
        # Sort by re-ranker score in descending order
        ranked_candidates = sorted(candidates, key=lambda x: x["score"], reverse=True)
        
        # 4. Return top-k re-ranked candidates
        return ranked_candidates[:top_k_rerank]

def main():
    parser = argparse.ArgumentParser(description="Test the retriever pipeline")
    parser.add_argument("--query", type=str, default="What is a transformer model?", help="Test query")
    args = parser.parse_args()
    
    retriever = Retriever()
    
    print(f"\nRetrieving for query: '{args.query}'\n")
    results = retriever.retrieve(args.query)
    
    if not results:
        print("No results found.")
        return
        
    for i, res in enumerate(results):
        print(f"--- Rank {i+1} (Re-ranker Score: {res['score']:.4f}) ---")
        print(f"Source: {res['source']} - {res['section']}")
        print(f"URL: {res['url']}")
        print(f"Text Snippet: {res['text'][:200]}...\n")

if __name__ == '__main__':
    main()
