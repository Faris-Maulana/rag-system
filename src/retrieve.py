import argparse

import chromadb
import numpy as np

from config import CHROMA_PERSIST_DIR, COLLECTION_NAME, EMBEDDING_MODEL, RERANKER_MODEL


def mmr_select(query_emb, doc_embs, k, lambda_mult=0.5):
    """Maximal Marginal Relevance: greedily pick k docs balancing query similarity and novelty.

    lambda_mult=1 reduces to plain similarity ranking, 0 maximises diversity.
    Returns indices into doc_embs in selection order.
    """
    doc_embs = np.asarray(doc_embs, dtype=float)
    if len(doc_embs) == 0:
        return []
    q = np.asarray(query_emb, dtype=float)
    q = q / (np.linalg.norm(q) or 1.0)
    d = doc_embs / np.clip(np.linalg.norm(doc_embs, axis=1, keepdims=True), 1e-12, None)
    query_sim = d @ q
    doc_sim = d @ d.T

    selected = [int(np.argmax(query_sim))]
    candidates = set(range(len(d))) - set(selected)
    while candidates and len(selected) < k:
        scores = {
            i: lambda_mult * query_sim[i] - (1 - lambda_mult) * doc_sim[i, selected].max()
            for i in candidates
        }
        best = max(scores, key=scores.get)
        selected.append(best)
        candidates.remove(best)
    return selected

class Retriever:
    def __init__(self,
                 embedding_model_name=EMBEDDING_MODEL,
                 reranker_model_name=RERANKER_MODEL,
                 chroma_path=CHROMA_PERSIST_DIR,
                 collection_name=COLLECTION_NAME,
                 embedding_model=None,
                 reranker=None,
                 client=None):
        # Models and client are injectable so the ranking logic can be tested without downloads.
        if embedding_model is None or reranker is None:
            from sentence_transformers import CrossEncoder, SentenceTransformer
        if embedding_model is None:
            print(f"Loading embedding model: {embedding_model_name}")
            embedding_model = SentenceTransformer(embedding_model_name)
        if reranker is None:
            print(f"Loading re-ranker model: {reranker_model_name}")
            reranker = CrossEncoder(reranker_model_name)
        if client is None:
            print(f"Connecting to ChromaDB at: {chroma_path}")
            client = chromadb.PersistentClient(path=chroma_path)

        self.embedding_model = embedding_model
        self.reranker = reranker
        self.collection = client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        indexed_with = (self.collection.metadata or {}).get("embedding_model")
        if indexed_with and isinstance(embedding_model_name, str) and indexed_with != embedding_model_name:
            print(f"[Warning] Index was built with '{indexed_with}' but queries use "
                  f"'{embedding_model_name}'. Set EMBEDDING_MODEL to match or re-index.")

    def _dense_candidates(self, query_emb, n_results, include_embeddings=False):
        include = ["documents", "metadatas", "distances"]
        if include_embeddings:
            include.append("embeddings")
        results = self.collection.query(
            query_embeddings=query_emb,
            n_results=n_results,
            include=include
        )
        if not results['documents'] or not results['documents'][0]:
            return [], []

        candidates, embeddings, seen_texts = [], [], set()
        for i, text in enumerate(results['documents'][0]):
            # Identical passages (e.g. the same article indexed under two titles) waste top-k slots.
            if text in seen_texts:
                continue
            seen_texts.add(text)
            meta = results['metadatas'][0][i]
            candidates.append({
                "text": text,
                "source": meta["source"],
                "url": meta["url"],
                "section": meta.get("section", ""),
                "dense_distance": float(results['distances'][0][i]),
            })
            if include_embeddings:
                embeddings.append(results['embeddings'][0][i])
        return candidates, embeddings

    def retrieve(self, query: str, top_k_dense=20, top_k_rerank=3):
        """
        Retrieves top relevant chunks for a given query.
        1. Dense retrieval using ChromaDB (top 20, cosine distance)
        2. Re-ranking using Cross-Encoder (top 3)
        """
        query_emb = self.embedding_model.encode([query], convert_to_numpy=True).tolist()
        candidates, _ = self._dense_candidates(query_emb, top_k_dense)
        if not candidates:
            return []

        pairs = [(query, chunk["text"]) for chunk in candidates]
        scores = self.reranker.predict(pairs)
        for i, score in enumerate(scores):
            candidates[i]["score"] = float(score)

        ranked_candidates = sorted(candidates, key=lambda x: x["score"], reverse=True)
        return ranked_candidates[:top_k_rerank]

    def retrieve_mmr(self, query: str, k=5, fetch_k=20, lambda_mult=0.5):
        """Dense retrieval re-ordered by MMR to spread results across more source articles."""
        query_emb = self.embedding_model.encode([query], convert_to_numpy=True).tolist()
        candidates, embeddings = self._dense_candidates(query_emb, fetch_k, include_embeddings=True)
        order = mmr_select(query_emb[0], embeddings, k, lambda_mult)
        return [candidates[i] for i in order]

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
