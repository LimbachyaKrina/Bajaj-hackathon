# responder.py
from openai import OpenAI
import os
from dotenv import load_dotenv
load_dotenv()

client = OpenAI(api_key=os.getenv("RAY_OPENAI_API_KEY"))

MODEL_NAME = "gpt-4o"


def generate_structured_answer(question, chunks):
    """
    Formats a prompt for LLM with top-k chunks, returns structured JSON with:
    - answer
    - reasoning
    - clause/chunk references
    """
    context_text = "\n---\n".join([f"[{c['chunk_id']}]\n{c['content']}" for c in chunks])

    prompt = f"""
You are a legal and policy document analyst.
Answer the question below using ONLY the provided chunks.

Question:
{question}

Relevant Clauses:
{context_text}

Instructions:
- Carefully analyze the text to find exact matching clauses.
- Instructions:
- Be warm, clear, and supportive in tone — just like a helpful teammate.
- Stay consistent with past conversation if it helps answer the query.
- Focus only on the exact task asked. Do not guess beyond the chunks.
- Provide step-by-step answers when possible.
- Then explain WHY the clause matches the question.
- Reference the chunk IDs used.

Respond with this format:
{{
  "question": "...",
  "answer": "...",
  "reasoning": "...",
  "clauses": ["chunk_2", "chunk_4"]
}}
"""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=512
    )

    try:
        import json
        text = response.choices[0].message.content.strip()
        return json.loads(text)
    except Exception:
        return {
            "question": question,
            "answer": "Could not parse response.",
            "reasoning": "LLM returned unstructured text.",
            "clauses": [c["chunk_id"] for c in chunks]
        }
