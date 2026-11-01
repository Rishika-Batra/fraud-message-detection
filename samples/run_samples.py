"""
run_samples.py
Fraud Message Detection

Purpose:
Executes 5 benchmark test messages through the full analyze() pipeline:
  1. Clear Fraud: Fake KYC / account blocked SMS
  2. Clear Fraud: Electricity bill disconnection threat
  3. Borderline: Work-from-home / task-based investment scheme
  4. Normal Legitimate: Student campus meeting chat
  5. Genuine Bank OTP / Advisory: Legitimate bank notification to evaluate false-positive handling

Saves outputs to:
  - samples/sample_outputs.json (structured raw data)
  - samples/sample_outputs.md (formatted evaluation report)

Usage:
    python samples/run_samples.py
"""

import json
import os
import sys

# Ensure repository root is on sys.path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from app.pipeline import analyze

# 5 representative test cases
TEST_CASES = [
    {
        "id": "sample_01",
        "category": "Clear Fraud (KYC / Phishing)",
        "message": "Dear customer, your SBI bank account has been suspended due to pending KYC verification. Please click http://bit.ly/sbi-kyc-update to update your documents immediately."
    },
    {
        "id": "sample_02",
        "category": "Clear Fraud (Urgent Utility Disconnection Threat)",
        "message": "URGENT: Your electricity power supply will be disconnected tonight at 9:30 PM by electricity office because your previous month bill was not updated. Please contact power officer immediately at 9876543210."
    },
    {
        "id": "sample_03",
        "category": "Borderline / Task Investment Scam",
        "message": "Part-time job offer: Earn Rs 2500 - 5000 per day by simply reviewing hotels and liking YouTube videos from home. No experience needed. Contact HR manager Priya on WhatsApp now: +91-9871122334."
    },
    {
        "id": "sample_04",
        "category": "Normal Legitimate (Personal Message)",
        "message": "Hey, are we meeting at the campus library at 4 PM?"
    },
    {
        "id": "sample_05",
        "category": "Genuine Bank Alert / Potential Classifier False Positive",
        "message": "482910 is your secret OTP for transaction of INR 1,500.00 at Flipkart. Never share your OTP with anyone, including bank staff. If not done by you, call 1800112211 - SBI"
    }
]


def run_benchmark():
    print("=" * 70)
    print("Running Sample Benchmark Evaluation for Fraud Message Detection")
    print("=" * 70)

    results = []
    for case in TEST_CASES:
        print(f"\n[*] Evaluating [{case['id']}] {case['category']}...")
        print(f"    Message: \"{case['message']}\"")
        output = analyze(case["message"])

        result_item = {
            "id": case["id"],
            "category": case["category"],
            "message": case["message"],
            "model_label": output.get("label"),
            "confidence": output.get("confidence"),
            "retrieved_scores": output.get("retrieved_scores", []),
            "sources": output.get("sources", []),
            "explanation": output.get("explanation")
        }
        results.append(result_item)

        print(f"    -> Label: {result_item['model_label']} (Confidence: {result_item['confidence'] * 100:.1f}%)")
        print(f"    -> Retrieved Scores: {result_item['retrieved_scores']}")

    # Save to samples/sample_outputs.json
    samples_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(samples_dir, "sample_outputs.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[+] Saved JSON results to: {json_path}")

    # Save to samples/sample_outputs.md
    md_path = os.path.join(samples_dir, "sample_outputs.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Sample Benchmark Outputs: Fraud Message Detection\n\n")
        f.write("**System:** Grounded Explanation Pipeline\n\n")
        f.write("This document records the end-to-end pipeline execution across 5 benchmark messages, "
                "illustrating fraud detection, semantic advisory retrieval, and explanation grounding.\n\n")

        for item in results:
            f.write(f"## {item['id'].upper()}: {item['category']}\n\n")
            f.write(f"**Input Message:**\n> {item['message']}\n\n")
            f.write(f"- **Classifier Label:** `{item['model_label'].upper()}`\n")
            f.write(f"- **Model Confidence:** `{item['confidence'] * 100:.1f}%`\n")
            f.write(f"- **FAISS Similarity Scores:** `{item['retrieved_scores']}`\n\n")

            if item["sources"]:
                f.write("**Retrieved Official Sources:**\n")
                for s in item["sources"]:
                    f.write(f"- [{s['title']}]({s['url']})\n")
                f.write("\n")
            else:
                f.write("**Retrieved Official Sources:** None (Bypassed or below threshold)\n\n")

            f.write(f"**Pipeline Explanation:**\n\n{item['explanation']}\n\n")
            f.write("---\n\n")

    print(f"[+] Saved Markdown results to: {md_path}")
    print("\nBenchmark run completed successfully.")


if __name__ == "__main__":
    run_benchmark()

