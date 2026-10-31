"""
pipeline.py
Fraud Message Detection

Purpose:
Core analysis pipeline that integrates:
1. Fine-tuned DistilBERT classifier (model.predict.predict).
2. RAG retriever (rag.retriever.retrieve).
3. Grounded LLM explainer (rag.explain.explain).

Decision Logic:
- Call predict(message) first.
- If label == "legitimate":
    Returns immediately with confidence and:
    explanation = "Message looks legitimate; no explanation needed."
- If label == "fraud":
    Retrieves top 3 advisory notes from FAISS.
    Calls explain() to generate grounded rationale.
    Collects sources and similarity scores.
"""

import sys
import os
from typing import Dict, Any, List

# Ensure repository root is on sys.path so model and rag imports work reliably
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from model.predict import predict
from rag.retriever import retrieve
from rag.explain import explain


def analyze(message: str) -> Dict[str, Any]:
    """
    Analyzes an incoming SMS or text message through classification and RAG explanation.

    Args:
        message (str): The raw message text.

    Returns:
        dict: {
            "label": "fraud" | "legitimate",
            "confidence": float (0..1),
            "explanation": str,
            "sources": List[Dict[str, str]],  # [{"title": ..., "url": ...}]
            "retrieved_scores": List[float]   # Similarity scores of retrieved notes
        }
    """
    # Defensive check for empty or non-string inputs
    clean_message = str(message or "").strip()
    if not clean_message:
        return {
            "label": "legitimate",
            "confidence": 1.0,
            "explanation": "No message text provided.",
            "sources": [],
            "retrieved_scores": []
        }

    # Step 1: Run the fine-tuned DistilBERT classifier
    classification = predict(clean_message)
    label = classification.get("label", "legitimate")
    confidence = classification.get("confidence", 0.0)

    # Step 2: If classified as legitimate, bypass RAG retrieval
    if label == "legitimate":
        return {
            "label": "legitimate",
            "confidence": confidence,
            "explanation": "Message looks legitimate; no explanation needed.",
            "sources": [],
            "retrieved_scores": []
        }

    # Step 3: If flagged as fraud, retrieve top 3 semantically relevant advisories
    retrieved_notes = retrieve(clean_message, k=3)

    # Step 4: Generate grounded explanation with citations
    explanation_text = explain(clean_message, retrieved_notes)

    # Step 5: Format unique sources and similarity scores for the response
    sources: List[Dict[str, str]] = []
    seen_urls = set()
    for note in retrieved_notes:
        url = note.get("url", "")
        title = note.get("title", "")
        if url and url not in seen_urls:
            seen_urls.add(url)
            sources.append({"title": title, "url": url})

    retrieved_scores = [note.get("score", 0.0) for note in retrieved_notes]

    return {
        "label": "fraud",
        "confidence": confidence,
        "explanation": explanation_text,
        "sources": sources,
        "retrieved_scores": retrieved_scores
    }


if __name__ == "__main__":
    # Test script directly via command line
    test_msg = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "Dear customer, your bank account is suspended due to pending KYC. Click http://bit.ly/update to verify."
    )
    print(f"\nAnalyzing: \"{test_msg}\"")
    result = analyze(test_msg)
    print(f"Label:            {result['label']}")
    print(f"Confidence:       {result['confidence']}")
    print(f"Retrieved Scores: {result['retrieved_scores']}")
    print(f"Sources:          {result['sources']}")
    print(f"Explanation:\n{result['explanation']}\n")

