import os
import json
import argparse
from tqdm import tqdm
import chromadb
from sentence_transformers import SentenceTransformer

def load_chunks(filepath):
    chunks = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))
    return chunks

def main():
    parser = argparse.ArgumentParser(description="Embed Wikipedia chunks and store in ChromaDB")
    parser.add_argument("--model", type=str, default="all-MiniLM-L6-v2", 
                        help="Path or name of the embedding model to use")
    parser.add_argument("--data", type=str, default="data/processed/chunks.jsonl",
                        help="Path to the chunked data file")
    args = parser.parse_args()

    print(f"Loading chunks from {args.data}...")
    chunks = load_chunks(args.data)
    print(f"Loaded {len(chunks)} chunks.")

    if not chunks:
        print("No chunks found. Exiting.")
        return

    print(f"Loading embedding model: {args.model}")
    model = SentenceTransformer(args.model)

    print("Initializing ChromaDB...")
    persist_directory = "./vectorstore/chroma_db"
    
    # Optional logic: If we're changing models (e.g., to finetuned), we might need to clear the old DB
    # or use a different collection name to avoid dimension mismatch errors. 
    # For now, we stick to the required task.
    
    client = chromadb.PersistentClient(path=persist_directory)

    # Use get_or_create to avoid errors on re-runs and explicitly set cosine distance space (from debug guide)
    collection = client.get_or_create_collection(
        name="rag_wikipedia",
        metadata={"hnsw:space": "cosine"}
    )

    batch_size = 64
    print(f"Embedding and upserting chunks in batches of {batch_size}...")

    # Process in batches
    for i in tqdm(range(0, len(chunks), batch_size)):
        batch = chunks[i:i + batch_size]
        
        # Extract fields
        ids = [c["chunk_id"] for c in batch]
        texts = [c["text"] for c in batch]
        metadatas = [{
            "source": c["source"],
            "url": c["url"],
            "section": c["section"],
            "token_count": c["token_count"]
        } for c in batch]

        # Generate embeddings
        embeddings = model.encode(texts, convert_to_numpy=True).tolist()

        # Upsert to ChromaDB
        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas
        )

    print(f"Stored {collection.count()} total chunks in ChromaDB.")
    print("Embedding complete.")

    # Validation step inline
    print("\n--- Running Quick Validation ---")
    query = "What is a transformer model?"
    print(f"Query: '{query}'")
    
    query_emb = model.encode([query]).tolist()
    results = collection.query(
        query_embeddings=query_emb,
        n_results=1
    )
    
    if results['documents'] and results['documents'][0]:
        print(f"Top result source: {results['metadatas'][0][0]['source']}")
        print(f"Top result text snippet: {results['documents'][0][0][:100]}...")
        print("Validation successful!")
    else:
        print("Validation failed: No results returned.")


if __name__ == '__main__':
    main()
