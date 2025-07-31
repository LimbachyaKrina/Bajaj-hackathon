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
        "question": "What is the grace period for premium payment?",
        "expected": "The Grace Period for payment of the premium shall be thirty days."
    },
    {
        "question": "After how long are pre-existing diseases covered?",
        "expected": "Covered after 36 months of continuous coverage."
    },
    {
        "question": "Does this policy cover maternity expenses, and what are the eligibility conditions?",
        "expected": "The female Insured Person should have been continuously covered for at least 24 months before availing this benefit."
    },
    {
        "question": "What is the cataract surgery limit under Plan A?",
        "expected": "Up to 15% of SI or INR 60,000 whichever is lower."
    },
    {
        "question": "Does the policy cover expenses for an organ donor?",
        "expected": "The policy covers organ donor’s hospitalization expenses, provided the organ is donated to an Insured Person and complies with the Transplantation of Human Organs Act, 1994."
    },
    {
        "question": "Is there coverage for AYUSH treatment under this policy?",
        "expected": "Yes, inpatient AYUSH treatment is covered up to the Sum Insured in AYUSH hospitals."
    },
    {
        "question": "What is the ambulance reimbursement limit under Plan A?",
        "expected": "Up to INR 2,500 per insured person in a policy year."
    },
    {
        "question": "What is the waiting period for cataract surgery?",
        "expected": "Covered after a waiting period of two years."
    },
    {
        "question": "How is No Claim Discount (NCD) calculated?",
        "expected": "A 5% discount on the base premium is provided on renewal if no claims were made in the expiring policy term."
    },
    {
        "question": "Are infertility treatments covered?",
        "expected": "Yes, infertility treatment is covered up to INR 50,000 after a waiting period of two years."
    },
    {
        "question": "What is the hospital cash benefit under Plan A?",
        "expected": "INR 500 per day for a maximum of 5 days after 3 days of hospitalization."
    },
    {
        "question": "Is coverage available for air ambulance services under Plan A?",
        "expected": "Not covered under Plan A."
    },
    {
        "question": "Does the policy cover HIV/AIDS related medical expenses?",
        "expected": "Yes, coverage is provided for treatment of HIV at different stages, including AIDS."
    },
    {
        "question": "Is anti-rabies vaccination covered?",
        "expected": "Yes, up to INR 5,000 per policy year without hospitalization."
    },
    {
        "question": "What is the waiting period for maternity claims?",
        "expected": "24 months of continuous coverage is required."
    },
    {
        "question": "Are dental treatments covered under this policy?",
        "expected": "Covered only if necessitated due to an injury."
    },
    {
        "question": "Does the policy cover correction of vision due to refractive error?",
        "expected": "Yes, covered for errors ≥ 7.5 dioptres after a 2-year waiting period."
    },
    {
        "question": "What is the hospitalization limit for Domiciliary treatment?",
        "expected": "Up to INR 1,00,000 under Plan A."
    },
    {
        "question": "Does the policy include coverage for robotic surgery?",
        "expected": "Yes, under modern treatment, up to 25% of the Sum Insured."
    },
    {
        "question": "Are vaccinations for children covered under the policy?",
        "expected": "Yes, vaccinations are covered up to INR 1,000 for male child up to 12 and female child up to 14 years under Plan A."
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
