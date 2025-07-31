import os
import re
import json
from openai import OpenAI

api_key = os.getenv("RAY_OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("❌ RAY_OPENAI_API_KEY is not set!")

client = OpenAI(api_key=api_key)
MODEL_NAME = "gpt-4o"

def generate_structured_answer(question, chunks):
    print("="*80)
    print(f"🔍 DEBUGGING FOR QUESTION:\n{question}\n")
    print("📄 Top Retrieved Chunks:")
    for i, chunk in enumerate(chunks):
        print(f"\n🔹 Chunk {i+1}: {chunk['chunk_id']}")
        print("-" * 60)
        print(chunk["content"])
        print("-" * 60)

    context_text = "\n---\n".join([f"[{c['chunk_id']}]\n{c['content']}" for c in chunks])

    prompt = f"""
        You are an expert in analyzing legal, insurance, and compliance policy documents.
        Use ONLY the provided chunks to answer the question below.

        ---
        Question:
        {question}

        Relevant Clauses:
        {context_text}
        ---

        Instructions:
        - Only use the chunks.
        - No external info.
        - Use clause language/limits as-is.
        - State all conditions.
        - Mention clause IDs like chunk_3_pg_7.
        - Keep concise and factual.

        Format:
        {{
        "question": "...",
        "answer": "...",
        "reasoning": "...",
        "clauses": ["chunk_3_pg_7", "chunk_6_pg_8"]
        }}
    """

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=512
        )
        text = response.choices[0].message.content.strip()
        json_block = re.search(r"\{.*\}", text, re.DOTALL).group()
        return json.loads(json_block)
    except Exception as e:
        print("🚨 Error in generate_structured_answer:", e)
        return {
            "question": question,
            "answer": "Could not parse or generate response.",
            "reasoning": str(e),
            "clauses": [c["chunk_id"] for c in chunks]
        }
