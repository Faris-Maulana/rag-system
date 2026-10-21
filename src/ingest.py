import json
import os
import re

import tiktoken
import wikipediaapi
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm

from config import CHUNKS_PATH

USER_AGENT = "rag-system/1.0 (https://github.com/Faris-Maulana/rag-system)"

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
        "Autoencoder", "Generative adversarial network",
        "Attention (machine learning)", "Language model", "Foundation model",
        "Prompt engineering", "Fine-tuning (machine learning)", "Transfer learning",
        "Overfitting", "Backpropagation", "Gradient descent", "Stochastic gradient descent",
        "Activation function", "Perceptron", "Multilayer perceptron", "Q-learning",
        "Markov decision process", "OpenAI", "Geoffrey Hinton", "Yann LeCun",
        "Yoshua Bengio", "Andrew Ng", "Turing test", "Artificial general intelligence",
        "Explainable artificial intelligence", "Ethics of artificial intelligence",
        "Vector database", "Semantic search", "Sentence embedding", "ChromaDB",
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

def safe_filename(title):
    return re.sub(r"[^\w\-.()]+", "_", title)

def build_splitter(enc):
    return RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " "],
        chunk_size=512,
        chunk_overlap=64,
        length_function=lambda x: len(enc.encode(x))
    )

def chunk_article(title, url, sections_data, splitter, enc):
    """Split an article's sections into chunk records with per-article stable IDs."""
    safe_title = safe_filename(title)
    chunks = []
    for sec in sections_data:
        text = sec["text"]
        if not text.strip():
            continue
        for c in splitter.split_text(text):
            chunks.append({
                "source": title,
                "url": url,
                "section": sec["title"],
                "chunk_id": f"{safe_title}_{len(chunks) + 1:04d}",
                "token_count": len(enc.encode(c)),
                "text": c
            })
    return chunks

def main():
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs(os.path.dirname(CHUNKS_PATH), exist_ok=True)

    wiki = wikipediaapi.Wikipedia(language='en', user_agent=USER_AGENT)

    topics = get_wikipedia_pages()
    enc = tiktoken.get_encoding("cl100k_base")
    splitter = build_splitter(enc)

    raw_articles = []
    processed_chunks = []
    seen_titles = set()

    print(f"Fetching {len(topics)} topics from Wikipedia...")
    for topic in tqdm(topics):
        page = wiki.page(topic)
        if not page.exists():
            print(f"Warning: Page not found for topic: {topic}")
            continue
        # Several topics are redirects to the same article (e.g. "Transformer architecture"
        # -> "Transformer (deep learning)"); indexing it twice puts duplicate chunks in the top-k.
        if page.title in seen_titles:
            print(f"Skipping duplicate: '{topic}' resolves to '{page.title}'")
            continue
        seen_titles.add(page.title)

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

        raw_path = f"data/raw/{safe_filename(page.title)}.json"
        with open(raw_path, "w", encoding="utf-8", errors="replace") as f:
            json.dump(raw_article, f, ensure_ascii=False, indent=2)

        processed_chunks.extend(chunk_article(page.title, page.fullurl, sections_data, splitter, enc))

    print(f"Saving {len(processed_chunks)} chunks...")
    with open(CHUNKS_PATH, "w", encoding="utf-8", errors="replace") as f:
        for chunk in processed_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    print("Ingestion complete.")
    print(f"Total articles downloaded: {len(raw_articles)}")
    print(f"Total chunks generated: {len(processed_chunks)}")
    if processed_chunks:
        print(f"Average tokens per chunk: {sum(c['token_count'] for c in processed_chunks) / len(processed_chunks):.1f}")

if __name__ == '__main__':
    main()
