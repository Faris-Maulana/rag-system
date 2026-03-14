import os
import json
import wikipediaapi
import tiktoken
from tqdm import tqdm
from langchain_text_splitters import RecursiveCharacterTextSplitter

def get_wikipedia_pages():
    topics = [
        "Machine learning", "Deep learning", "Transformer (machine learning model)", 
        "Large language model", "Retrieval-augmented generation", "BERT (language model)", 
        "Generative pre-trained transformer", "Artificial intelligence", 
        "Natural language processing", "Computer vision", "Reinforcement learning", 
        "Supervised learning", "Unsupervised learning", "Word embedding", 
        "Convolutional neural network", "Recurrent neural network", 
        "Support vector machine", "Decision tree learning", "Random forest", 
        "Gradient boosting", "K-means clustering", "Principal component analysis", 
        "Autoencoder", "Generative adversarial network", "Transformer architecture", 
        "Attention (machine learning)", "Language model", "Foundation model", 
        "Prompt engineering", "Fine-tuning (machine learning)", "Transfer learning", 
        "Overfitting", "Backpropagation", "Gradient descent", "Stochastic gradient descent", 
        "Activation function", "Perceptron", "Multilayer perceptron", "Q-learning", 
        "Markov decision process", "OpenAI", "Geoffrey Hinton", "Yann LeCun", 
        "Yoshua Bengio", "Andrew Ng", "Turing test", "Artificial general intelligence", 
        "Explainable artificial intelligence", "Ethics of artificial intelligence", 
        "Vector database", "Semantic search", "SentenceTransformers", "ChromaDB",
        "Sparse dictionary learning", "Data mining"
    ]
    return topics

def get_sections(section, parent_title=""):
    sections = []
    title = f"{parent_title} > {section.title}" if parent_title else section.title
    if section.text.strip():
        sections.append({
            "title": title,
            "text": section.text
        })
    for s in section.sections:
        sections.extend(get_sections(s, title))
    return sections

def main():
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    
    # Initialize Wikipedia API with custom user agent (required by Wikipedia)
    wiki = wikipediaapi.Wikipedia(
        language='en',
        user_agent='rag-project/1.0 (interview.candidate@example.com)'
    )
    
    topics = get_wikipedia_pages()
    
    # Initialize tokenizer for length calculation
    enc = tiktoken.get_encoding("cl100k_base")
    
    # Configure exact chunker requested
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " "],
        chunk_size=512,
        chunk_overlap=64,
        length_function=lambda x: len(enc.encode(x))
    )
    
    raw_articles = []
    processed_chunks = []
    chunk_counter = 0
    
    print(f"Fetching {len(topics)} topics from Wikipedia...")
    for topic in tqdm(topics):
        page = wiki.page(topic)
        if not page.exists():
            print(f"Warning: Page not found for topic: {topic}")
            continue
            
        # Extract title and sections
        sections_data = [{"title": "Summary", "text": page.summary}]
        for s in page.sections:
            sections_data.extend(get_sections(s))
            
        raw_article = {
            "title": page.title,
            "url": page.fullurl,
            "content": page.text,
            "sections": sections_data
        }
        raw_articles.append(raw_article)
        
        # Save raw JSON with safe encoding
        safe_title = page.title.replace("/", "_").replace(" ", "_").replace(":", "_")
        raw_path = f"data/raw/{safe_title}.json"
        with open(raw_path, "w", encoding="utf-8", errors="replace") as f:
            json.dump(raw_article, f, ensure_ascii=False, indent=2)
            
        # Perform chunking with LangChain's RecursiveCharacterTextSplitter
        for sec in sections_data:
            text = sec["text"]
            if not text.strip():
                continue
                
            chunks = splitter.split_text(text)
            for c in chunks:
                chunk_counter += 1
                token_count = len(enc.encode(c))
                chunk_id = f"{safe_title}_{chunk_counter:04d}"
                
                chunk_meta = {
                    "source": page.title,
                    "url": page.fullurl,
                    "section": sec["title"],
                    "chunk_id": chunk_id,
                    "token_count": token_count,
                    "text": c
                }
                processed_chunks.append(chunk_meta)
                
    # Save processed chunks to JSONL
    print(f"Saving {len(processed_chunks)} chunks...")
    chunks_path = "data/processed/chunks.jsonl"
    with open(chunks_path, "w", encoding="utf-8", errors="replace") as f:
        for chunk in processed_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
            
    print("Ingestion complete.")
    print(f"Total topics downloaded: {len(raw_articles)}")
    print(f"Total chunks generated: {len(processed_chunks)}")
    if processed_chunks:
        print(f"Average tokens per chunk: {sum(c['token_count'] for c in processed_chunks) / len(processed_chunks):.1f}")

if __name__ == '__main__':
    main()
