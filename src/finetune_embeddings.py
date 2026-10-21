import argparse
import json
import os
import random
import time

import numpy as np

# HuggingFace / SentenceTransformers
import torch
from dotenv import load_dotenv

# Groq for synthetic QA generation
from groq import Groq, RateLimitError

# BM25 for hard negative mining
from rank_bm25 import BM25Okapi
from sentence_transformers import InputExample, SentenceTransformer, losses
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import CHUNKS_PATH, GROQ_MODEL

load_dotenv()

def generate_qa_pairs(chunks, sample_size=100, groq_batch_sleep=2):
    """Uses Groq to reliably generate synthetic Question-Answer pairs from text chunks."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY required for synthetic QA generation.")

    client = Groq(api_key=api_key)

    print(f"Generating {sample_size} synthetic QA pairs using llama-3.1-8b...")

    # We sample random chunks to generate questions from
    random.seed(42)
    sampled_chunks = random.sample(chunks, min(sample_size, len(chunks)))

    qa_data = []

    for attempt, chunk in enumerate(tqdm(sampled_chunks)):
        text = chunk["text"]

        prompt = f"""
        Given the following text chunk, generate exactly ONE highly specific question
        that can be answered exclusively by this text.

        Return ONLY the question string, nothing else. No formatting, no extra words.

        Text Context:
        {text}
        """

        # Retry logic for rate limits
        max_retries = 3
        for r in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=GROQ_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=100,
                    temperature=0.3
                )
                question = response.choices[0].message.content.strip().strip('"')

                qa_data.append({
                    "question": question,
                    "positive_context": text,
                    "chunk_id": chunk.get("chunk_id", str(attempt))
                })
                break
            except RateLimitError:
                wait_time = (r + 1) * 3
                print(f"Rate limited. Waiting {wait_time}s...")
                time.sleep(wait_time)
            except Exception as e:
                print(f"Error generating QA: {e}")
                break

        # Modest sleep to prevent hammering Groq rate limits
        time.sleep(groq_batch_sleep)

    return qa_data


def split_qa(qa_data, test_fraction=0.2, seed=42):
    """Deterministic train/held-out split of the synthetic QA pairs."""
    shuffled = list(qa_data)
    random.Random(seed).shuffle(shuffled)
    n_test = int(round(len(shuffled) * test_fraction))
    return shuffled[n_test:], shuffled[:n_test]


def mine_hard_negatives(qa_data, all_chunks):
    """Uses BM25 to find documents that match the query keywords but aren't the answer."""
    print(f"Mining hard negatives using BM25 for {len(qa_data)} queries...")

    # Prepare corpus for BM25
    corpus_texts = [c["text"] for c in all_chunks]

    # Simple tokenization for BM25
    tokenized_corpus = [doc.lower().split(" ") for doc in corpus_texts]
    bm25 = BM25Okapi(tokenized_corpus)

    finetune_triplets = []

    for item in tqdm(qa_data):
        query = item["question"]
        positive = item["positive_context"]

        # Get top 5 BM25 matches for the query
        tokenized_query = query.lower().split(" ")
        doc_scores = bm25.get_scores(tokenized_query)

        # Sort indices by score descending
        top_indices = np.argsort(doc_scores)[::-1][:10]

        # Find a hard negative (high lexical overlap, but different text from positive)
        hard_negative = None
        for idx in top_indices:
            candidate_text = corpus_texts[idx]
            # Ensure it's not simply a substring or exact match of the positive text
            if candidate_text != positive and len(candidate_text) > 50:
                hard_negative = candidate_text
                break

        # Fallback if no hard negative found
        if not hard_negative:
            hard_negative = random.choice(corpus_texts)

        finetune_triplets.append(
            InputExample(texts=[query, positive, hard_negative])
        )

    return finetune_triplets


def fine_tune_model(train_examples, base_model_name="all-MiniLM-L6-v2", epochs=2, output_dir="embeddings/finetuned/"):
    """Trains the embedding model using MultipleNegativesRankingLoss."""
    print(f"\n--- Fine-tuning {base_model_name} with {len(train_examples)} triplets ---")

    # Load model
    model = SentenceTransformer(base_model_name)

    # Determine batch size. 16 is good for MiniLM, but requires at least #examples >= batch size
    # MultipleNegativesRankingLoss REQUIRES batch size >= 2 for in-batch negatives to work.
    batch_size = min(16, len(train_examples))
    if batch_size < 2:
        print("Error: Too few examples for MultipleNegativesRankingLoss. Need >= 2.")
        return None

    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)

    # Use MultipleNegativesRankingLoss
    # This loss inherently uses the positive from one triplet as a negative for all other
    # queries in the batch (in-batch negatives), combined with our hard negatives.
    train_loss = losses.MultipleNegativesRankingLoss(model=model)

    os.makedirs(output_dir, exist_ok=True)

    print(f"Beginning training for {epochs} epochs...")
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=epochs,
        warmup_steps=int(len(train_dataloader) * epochs * 0.1),
        output_path=output_dir,
        show_progress_bar=True
    )

    print(f"Fine-tuning complete. Model saved to: {output_dir}")
    return output_dir


def evaluate_mrr(model_name, qa_data, all_chunks_texts, k=10):
    """Calculates MRR@10 for a given model on our generated QA dataset."""
    print(f"Evaluating MRR@{k} for model: {model_name}")
    model = SentenceTransformer(model_name)

    # Encode complete corpus
    corpus_embeddings = model.encode(all_chunks_texts, convert_to_tensor=True, show_progress_bar=True)

    queries = [item["question"] for item in qa_data]
    query_embeddings = model.encode(queries, convert_to_tensor=True)

    from sentence_transformers import util

    # Cosine similarity
    cos_scores = util.cos_sim(query_embeddings, corpus_embeddings)

    mrr_sum = 0.0
    for i in range(len(qa_data)):
        target_text = qa_data[i]["positive_context"]
        scores = cos_scores[i]

        # Get top-k indices
        top_k_idx = torch.topk(scores, k=min(k, len(all_chunks_texts)))[1].cpu().numpy()

        # Find if our target chunk is in top-K
        rank = 0
        for rank_idx, corpus_idx in enumerate(top_k_idx):
            if all_chunks_texts[corpus_idx] == target_text:
                rank = rank_idx + 1
                break

        if rank > 0:
            mrr_sum += 1.0 / rank

    mrr = mrr_sum / len(qa_data)
    print(f"MRR@{k}: {mrr:.4f}")
    return mrr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=100, help="Number of QA pairs to generate")
    parser.add_argument("--epochs", type=int, default=2, help="Number of fine-tuning epochs")
    parser.add_argument("--data", type=str, default=CHUNKS_PATH, help="Path to chunks")
    parser.add_argument("--test_fraction", type=float, default=0.2, help="Share of QA pairs held out for MRR")
    args = parser.parse_args()

    # Load data
    with open(args.data, "r", encoding="utf-8") as f:
        all_chunks = [json.loads(line) for line in f if line.strip()]

    all_chunks_texts = [c["text"] for c in all_chunks]

    print("=======================================")
    print("      EMBEDDING MODEL FINE-TUNING")
    print("=======================================")

    # 1. Generate Synthetic QA Data
    qa_data_file = "data/processed/synthetic_qa.json"
    if os.path.exists(qa_data_file):
        print(f"Loading existing QA pairs from {qa_data_file}")
        with open(qa_data_file, "r") as f:
            qa_data = json.load(f)
    else:
        qa_data = generate_qa_pairs(all_chunks, sample_size=args.samples)
        os.makedirs(os.path.dirname(qa_data_file), exist_ok=True)
        with open(qa_data_file, "w") as f:
            json.dump(qa_data, f, indent=2)

    # 2. Hold out a test split: measuring MRR on the training questions would only show memorisation
    train_qa, test_qa = split_qa(qa_data, test_fraction=args.test_fraction)
    print(f"QA pairs: {len(train_qa)} train / {len(test_qa)} held-out")
    if not test_qa:
        raise SystemExit("Not enough QA pairs for a held-out split; increase --samples.")

    # 3. Mine hard negatives (BM25) for the training split only
    train_triplets = mine_hard_negatives(train_qa, all_chunks)

    # 4. Evaluate baseline MRR on the held-out split
    baseline_model = "all-MiniLM-L6-v2"
    baseline_mrr = evaluate_mrr(baseline_model, test_qa, all_chunks_texts)

    # 5. Fine-tune model
    finetuned_dir = "embeddings/finetuned/"
    fine_tune_model(train_triplets, base_model_name=baseline_model, epochs=args.epochs, output_dir=finetuned_dir)

    # 6. Evaluate fine-tuned MRR on the same held-out split
    finetuned_mrr = evaluate_mrr(finetuned_dir, test_qa, all_chunks_texts)

    print("\n=======================================")
    print("           FINE-TUNING SUMMARY")
    print("=======================================")
    print(f"Held-out queries: {len(test_qa)}")
    print(f"Baseline ({baseline_model}) MRR@10: {baseline_mrr:.4f}")
    print(f"Fine-Tuned ({finetuned_dir}) MRR@10: {finetuned_mrr:.4f}")

    improvement = ((finetuned_mrr - baseline_mrr) / baseline_mrr) * 100 if baseline_mrr > 0 else 0
    print(f"Improvement: {improvement:+.2f}%")
    print("=======================================")

if __name__ == "__main__":
    main()
