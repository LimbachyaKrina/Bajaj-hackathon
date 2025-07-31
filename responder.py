# responder.py
from openai import OpenAI
import os
import re
from langchain.llms import OpenAI as LangOpenAI

api_key = os.getenv("RAY_OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("❌ RAY_OPENAI_API_KEY is not set in environment variables!")

client = OpenAI(api_key=api_key)

llm = LangOpenAI(openai_api_key=api_key)
# when i am using the llm model (omt)

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

    
    """
    Formats a prompt for LLM with top-k chunks, returns structured JSON with:
    - answer
    - reasoning
    - clause/chunk references
    """
    context_text = "\n---\n".join([f"[{c['chunk_id']}]\n{c['content']}" for c in chunks])

    prompt = f"""
            You are an expert in analyzing legal, insurance, and compliance policy documents.

            Use the chunks provided below to answer the user question as accurately as possible.
            Do NOT use any external knowledge — rely ONLY on the given text.

            ---
            Question:
            {question}

            Relevant Clauses:
            {context_text}
            ---

           Instructions:
            - Extract the answer based only on the chunks.
            - Avoid speculation or generic responses.
            - Use exact terms, clause language, or limits (like days, months, ₹ values) mentioned.
            - If the answer depends on conditions (e.g., eligibility period, age, continuous coverage), clearly mention **all** such conditions.
            - Mention the clause IDs (chunk_X_pg_Y) that support your answer.
            - Keep the answer concise, factual, and easy to verify.
            - Your goal is to maximize alignment with the expected ground truth.
            ---

            Respond in the following JSON format:
            {{
            "question": "...",
            "answer": "...",           // concise and directly answers the question
            "reasoning": "...",        // explain why the chunks support the answer
            "clauses": ["chunk_3_pg_7", "chunk_6_pg_8"]
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
        # text = response.choices[0].message.content.strip()
        # return json.loads(text)
        text = response.choices[0].message.content.strip()
        json_block = re.search(r"\{.*\}", text, re.DOTALL).group()
        return json.loads(json_block)
    except Exception:
        return {
            "question": question,
            "answer": "Could not parse response.",
            "reasoning": "LLM returned unstructured text.",
            "clauses": [c["chunk_id"] for c in chunks]
        }
