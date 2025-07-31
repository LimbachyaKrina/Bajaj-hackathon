# benchmarking.py
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

API_URL = os.getenv("API_URL", "http://localhost:8000/hackrx/run")
API_KEY = os.getenv("HACKRX_API_KEY")
BLOB_URL = "https://hackrx.blob.core.windows.net/assets/hackrx_6/policies/CHOTGDP23004V012223.pdf?sv=2023-01-03&st=2025-07-30T06%3A46%3A49Z&se=2025-09-01T06%3A46%3A00Z&sr=c&sp=rl&sig=9szykRKdGYj0BVm1skP%2BX8N9%2FRENEn2k7MQPUp33jyQ%3D"

client = OpenAI(api_key=os.getenv("RAY_OPENAI_API_KEY"))

TEST_SET = [
    {
        "question": "What does OPD stand for in this document?",
        "expected": "Out-Patient Department (treatment without hospital admission)."
    },
    {
        "question": "What does PTD stand for in the context of personal accident cover?",
        "expected": "Permanent Total Disability."
    },
    {
        "question": "Which condition is not covered under Emergency Accidental Hospitalization?",
        "expected": "Cosmetic treatment or plastic surgery unless required for accident/burns."
    },
    {
        "question": "What is the meaning of \u201cAccident\u201d as per the document?",
        "expected": "A sudden, unforeseen, and involuntary event caused by external, visible, and violent means."
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

    text = response.choices[0].message.content.strip()
    json_block = re.search(r"\{.*\}", text, re.DOTALL).group()
    return json.loads(json_block)

def run_benchmark():
    results = []
    questions = [q["question"] for q in TEST_SET]
    payload = {
        "documents": BLOB_URL,
        "questions": questions
    }

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    start_time = time.time()
    response = requests.post(API_URL, json=payload, headers=headers)
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
