# benchmarking.py (with GPT evaluation)
import requests
import time
import csv
from difflib import SequenceMatcher
from openai import OpenAI
import os
import re
import json
from dotenv import load_dotenv
load_dotenv()

API_URL = "http://localhost:8000/api/v1/hackrx/run"
BLOB_URL = "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D"

client = OpenAI(api_key=os.getenv("RAY_OPENAI_API_KEY"))
# print("client : ",client)
TEST_SET = [
    {
        "question": "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?",
        "expected": "The Grace Period for payment of the premium shall be thirty days."
    },
    {
        "question": "What is the waiting period for pre-existing diseases to be covered?",
        "expected": "Covered after 36 months of continuous coverage"
    },
    {
        "question": "Does this policy cover maternity expenses and under what conditions?",
        "expected": "female Insured Person should have been continuously covered for at least 24 months"
    },
    {
        "question": "Are medical expenses for an organ donor covered under this policy?",
        "expected": "expenses incurred in respect of an organ donor’s Hospitalisation"
    },
    {
        "question": "What is the limit for cataract surgery under Plan A?",
        "expected": "Up to 15% of SI or INR 60,000 whichever is lower"
    }
]


def fuzzy_match(a, b):
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def gpt_judgment(question, expected, predicted):
    prompt = f"""
You are an expert evaluator. Given the original question, the expected answer, and the model's predicted answer,
rate how well the predicted answer aligns with the expected answer on a scale of 0.0 to 1.0.

Question: {question}
Expected Answer: {expected}
Predicted Answer: {predicted}

Justify your score and then provide the final numeric score.
Respond in this format:
{{
  "score": float,
  "explanation": "..."
}}
"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=256
    )


    print(response.choices[0].message.content.strip())
    text = response.choices[0].message.content.strip()
    json_block = re.search(r"\{.*\}", text, re.DOTALL).group()
    return json.loads(json_block)
    # try:
    # except:
    #     return {"score": 0.0, "explanation": "Invalid JSON returned."}


def run_benchmark():
    results = []
    questions = [q["question"] for q in TEST_SET]
    payload = {
        "documents": BLOB_URL,
        "questions": questions
    }

    start_time = time.time()
    response = requests.post(API_URL, json=payload)
    elapsed = time.time() - start_time

    response.raise_for_status()
    predictions = response.json().get("answers", [])

    for i, pred in enumerate(predictions):
        gt = TEST_SET[i]["expected"]
        predicted = pred["answer"]
        reasoning = pred.get("reasoning", "")
        clauses = ", ".join(pred.get("clauses", []))
        score = fuzzy_match(predicted, gt)

        gpt_eval = gpt_judgment(pred["question"], gt, predicted)

        results.append({
            "Question": pred["question"],
            "Predicted Answer": predicted,
            "Expected": gt,
            "Fuzzy Score": round(score, 3),
            "GPT Score": gpt_eval.get("score", 0.0),
            "GPT Reason": gpt_eval.get("explanation", "N/A"),
            "Reasoning": reasoning,
            "Clauses": clauses,
            "Latency (s)": round(elapsed, 2)
        })

    with open("rag_benchmark_results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print("✅ Benchmark complete. Results saved to rag_benchmark_results.csv")


if __name__ == "__main__":
    run_benchmark()
