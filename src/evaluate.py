import os
import sys
import json
import argparse
import time
import re
from tqdm import tqdm
from groq import Groq, RateLimitError

# Add src to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from pipeline import RAGPipeline

EVALUATION_QA_PAIRS = [
    {
        "question": "What is the attention mechanism in transformers?",
        "ground_truth": "The attention mechanism allows a model to weigh the importance of different parts of the input data differently. In transformers, self-attention allows the model to relate different words in a sequence to each other to compute representations."
    },
    {
        "question": "How does BERT differ from GPT?",
        "ground_truth": "BERT uses a bidirectional encoder to learn context from both left and right sides of a token simultaneously, whereas GPT uses a unidirectional (left-to-right) decoder, autoregressively predicting the next token."
    },
    {
        "question": "What is retrieval-augmented generation?",
        "ground_truth": "Retrieval-augmented generation (RAG) is a technique that grounds large language models by retrieving relevant facts from an external knowledge base before generating an answer, reducing hallucinations."
    },
    {
        "question": "Who invented the transformer architecture?",
        "ground_truth": "The transformer architecture was introduced by researchers at Google Brain and Google Research in the 2017 paper 'Attention Is All You Need', authored by Ashish Vaswani and others."
    },
    {
        "question": "What is a major limitation of large language models?",
        "ground_truth": "Major limitations include a tendency to hallucinate incorrect facts, high computational cost for training and inference, and the potential to reproduce biases present in their training data."
    },
    {
        "question": "What is the difference between supervised and unsupervised learning?",
        "ground_truth": "Supervised learning algorithms are trained using labeled data where inputs are paired with correct outputs, while unsupervised learning deals with unlabeled data to find hidden structures or patterns."
    },
    {
        "question": "What is backpropagation?",
        "ground_truth": "Backpropagation is an algorithm used in training neural networks to calculate the gradient of the loss function with respect to the weights by applying the chain rule, enabling weights to be updated via stochastic gradient descent."
    },
    {
        "question": "What is the purpose of an activation function in a neural network?",
        "ground_truth": "Activation functions introduce non-linearities into a neural network, allowing it to learn and model complex, non-linear relationships in the data."
    },
    {
        "question": "What is overfitting in machine learning?",
        "ground_truth": "Overfitting occurs when a model learns the training data too well, capturing noise and details that do not generalize to new, unseen data, resulting in poor performance on test sets."
    },
    {
        "question": "What does a convolutional neural network (CNN) excel at?",
        "ground_truth": "CNNs excel at processing grid-like topology data, such as images, making them highly effective for computer vision tasks like image classification and object detection."
    },
    {
        "question": "What are word embeddings?",
        "ground_truth": "Word embeddings are dense vector representations of words in a continuous vector space where semantically similar words are mapped to nearby points."
    },
    {
        "question": "What is a vector database?",
        "ground_truth": "A vector database is a specialized database designed to store, manage, and query high-dimensional vector embeddings efficiently, often used for similarity search."
    },
    {
        "question": "What is the Turing test?",
        "ground_truth": "The Turing test is a test of a machine's ability to exhibit intelligent behavior equivalent to, or indistinguishable from, that of a human, proposed by Alan Turing in 1950."
    },
    {
        "question": "What is transfer learning?",
        "ground_truth": "Transfer learning is a machine learning technique where a model developed for one task is reused as the starting point for a model on a second, related task."
    },
    {
        "question": "What is reinforcement learning?",
        "ground_truth": "Reinforcement learning is an area of machine learning where an agent learns to make decisions by performing actions in an environment to maximize cumulative reward."
    },
    {
        "question": "What is fine-tuning in the context of LLMs?",
        "ground_truth": "Fine-tuning involves taking a pre-trained language model and training it further on a smaller, domain-specific dataset to adapt its knowledge to specific tasks or domains."
    },
    {
        "question": "What is a multilayer perceptron?",
        "ground_truth": "A multilayer perceptron (MLP) is a class of feedforward artificial neural network consisting of at least three layers of nodes: an input layer, a hidden layer, and an output layer."
    },
    {
        "question": "What is a foundation model?",
        "ground_truth": "A foundation model is a large-scale AI model trained on a vast quantity of varied data without labeled examples, capable of being adapted to a wide range of downstream tasks."
    },
    {
        "question": "What is the role of ChromaDB in RAG?",
        "ground_truth": "ChromaDB acts as the vector store in a RAG pipeline, providing fast similarity search and storage for embedded document chunks to be retrieved based on a user query."
    },
    {
        "question": "What is prompt engineering?",
        "ground_truth": "Prompt engineering is the process of structuring text input that can be interpreted and understood by a generative AI model to produce optimized, desired outputs."
    }
]

def call_llm_judge(client, prompt):
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=20
            )
            return response.choices[0].message.content.strip()
        except RateLimitError:
            wait_time = (attempt + 1) * 3
            time.sleep(wait_time)
        except Exception as e:
            print(f"Error calling LLM judge: {e}")
            break
    return ""

def check_faithfulness(client, question, context, answer):
    prompt = f"""Given the following question, context extracted from a knowledge base, and the generated answer, determine if the answer is FULLY supported by the provided context. If the answer contains any information not present in the context, reply NO.

Question: {question}
Context: {context}
Answer: {answer}

Is the answer fully supported by the context? Reply with ONLY 'YES' or 'NO'."""
    
    response = call_llm_judge(client, prompt).upper()
    return 1.0 if "YES" in response else 0.0

def check_answer_relevance(client, question, answer):
    prompt = f"""Given the question and the generated answer, score how relevant and helpful the answer is to the question on a scale of 0.0 to 1.0. 
0.0 means completely irrelevant, 1.0 means perfectly directly answers the question.

Question: {question}
Answer: {answer}

Reply with ONLY a single float number between 0.0 and 1.0."""
    
    response = call_llm_judge(client, prompt)
    try:
        # Extract the first float found in the string
        match = re.search(r'0\.\d+|1\.0|0|1', response)
        if match:
            return float(match.group())
        return 0.0
    except ValueError:
        return 0.0

def check_context_precision(ground_truth, context_chunks):
    # Check if retrieved sources contain keywords from ground truth answer
    combined_context = " ".join([chunk["text"] for chunk in context_chunks]).lower()
    
    # Extract naive keywords from ground truth (words > 4 chars to skip common stop words)
    gt_words = set([re.sub(r'[^\w\s]', '', w.lower()) for w in ground_truth.split() if len(w) > 4])
    
    if not gt_words:
        return 0.0
        
    overlap_count = sum(1 for word in gt_words if word in combined_context)
    return overlap_count / len(gt_words)

def main():
    parser = argparse.ArgumentParser(description="Evaluate RAG Pipeline Manually")
    parser.add_argument("--num_samples", type=int, default=20, help="Number of samples to evaluate (max 20)")
    args = parser.parse_args()

    os.makedirs("results", exist_ok=True)
    
    print("Initializing RAG Pipeline for Evaluation...")
    print("Note: If Groq rate limits apply, execution will pause automatically.")
    rag = RAGPipeline()
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    
    samples = EVALUATION_QA_PAIRS[:args.num_samples]
    results_detailed = []

    faithfulness_scores = []
    relevance_scores = []
    precision_scores = []

    print(f"\nProcessing {len(samples)} evaluation queries manually...")
    for item in tqdm(samples):
        query = item["question"]
        ground_truth = item["ground_truth"]
        
        # Suppress RAG printing during eval loop
        import io
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        
        try:
            result = rag.query(query)
            ans = result["answer"]
            ctx_chunks = result["context_chunks"]
            combined_context = "\n".join([c["text"] for c in ctx_chunks])
        finally:
            sys.stdout = old_stdout
            
        # 1. Faithfulness
        f_score = check_faithfulness(client, query, combined_context, ans)
        faithfulness_scores.append(f_score)
        
        # 2. Answer Relevancy
        r_score = check_answer_relevance(client, query, ans)
        relevance_scores.append(r_score)
        
        # 3. Context Precision
        p_score = check_context_precision(ground_truth, ctx_chunks)
        precision_scores.append(p_score)
        
        results_detailed.append({
            "question": query,
            "ground_truth": ground_truth,
            "generated_answer": ans,
            "faithfulness_score": f_score,
            "answer_relevance_score": r_score,
            "context_precision_score": p_score
        })
        
        # Respect rate limits for Groq
        time.sleep(1)

    # Average metrics
    avg_faithfulness = sum(faithfulness_scores) / len(faithfulness_scores) if faithfulness_scores else 0
    avg_relevance = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0
    avg_precision = sum(precision_scores) / len(precision_scores) if precision_scores else 0

    # Save to JSON
    report_path = "results/evaluation_report.json"
    with open(report_path, "w") as f:
        json.dump(results_detailed, f, indent=4)
        
    print("\n" + "="*60)
    print("=== EVALUATION SUMMARY ===")
    print("="*60)
    print(f"Total Samples Evaluated: {len(samples)}")
    print("-" * 60)
    print(f"Faithfulness              : {avg_faithfulness:.4f}")
    print(f"Answer Relevancy          : {avg_relevance:.4f}")
    print(f"Context Precision         : {avg_precision:.4f}")
    print("-" * 60)
    print(f"Full detailed results saved to: {report_path}")
    
    target_scores = {
        "Faithfulness": (avg_faithfulness, 0.80),
        "Answer Relevancy": (avg_relevance, 0.75),
        "Context Precision": (avg_precision, 0.70)
    }
    
    print("\nTarget Achievement:")
    for metric, (score, target) in target_scores.items():
        status = "[PASS]" if score >= target else "[FAIL]"
        print(f"{metric:<20}: Score={score:.2f} | Target={target:.2f} -> {status}")

if __name__ == "__main__":
    main()
