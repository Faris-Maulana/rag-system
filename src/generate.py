import os
import time
from dotenv import load_dotenv
from groq import Groq, RateLimitError

# Load environment variables
load_dotenv()

SYSTEM_PROMPT = """You are a precise AI assistant that answers questions strictly based 
on the provided context. If the answer is not in the context, say 
"I don't have enough information to answer that." Do NOT hallucinate.
Always cite the source article you used."""

USER_PROMPT = """
Context:
---
{context}
---

Question: {question}

Answer (with source citation):"""

class Generator:
    def __init__(self, model_name="llama-3.1-8b-instant"):
        self.api_key = os.getenv("GROQ_API_KEY")
        if not self.api_key or self.api_key == "your_groq_api_key_here":
            raise ValueError("Groq API key is missing or invalid. Please check .env file.")
            
        self.client = Groq(api_key=self.api_key)
        self.model_name = model_name

    def format_prompt(self, question: str, contexts: list[str]) -> str:
        """Assembles context chunks into prompt format"""
        # Truncate context optionally using tiktoken if exceeding window
        # But for top-3 dense chunks (~1500 tokens), we're well under 8K limit
        full_context = "\n\n".join(contexts)
        return USER_PROMPT.format(context=full_context, question=question)

    def generate(self, question: str, context_chunks: list[str], max_retries=3) -> str:
        """Call LLM with backoff logic (helpful for free-tier rate limits)"""
        formatted_user_prompt = self.format_prompt(question, context_chunks)
        
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": formatted_user_prompt}
        ]

        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=0.0, # zero temperature for extraction/RAG
                    max_tokens=1024
                )
                
                if not response.choices:
                    return "Error: LLM returned no response."
                
                return response.choices[0].message.content.strip()

            except RateLimitError:
                wait = 2 ** attempt  # 1s, 2s, 4s
                print(f"[Warning] Groq Rate limit reached. Retrying in {wait}s...")
                time.sleep(wait)
            except Exception as e:
                print(f"Generation error: {e}")
                
        return "Error: Max retries exceeded connecting to LLM."
