"""
explain.py
Fraud Message Detection

Purpose:
Generates a grounded, factual explanation of why a suspicious message looks fraudulent,
or honestly flags potential false positives if the message appears legitimate.

Core Requirements:
1. Uses Google Gemini API (google-genai SDK) with key from GEMINI_API_KEY (.env).
2. Uses ONE clear prompt template enforcing:
   - Grounding strictly on retrieved advisory notes (no hallucinations/outside knowledge).
   - Inline bracket citations [1], [2] and sources listed at the end.
   - Honest handling: If notes do not support fraud (e.g. genuine OTP/bank alert),
     state clearly that notes do not show a fraud pattern and classifier may be wrong.
   - Brevity: Under 120 words in simple English.
3. If no notes were retrieved, skips the LLM and returns the exact fallback message.
4. Catches API/network errors gracefully without crashing.
"""

import os
from typing import List, Dict, Any
from dotenv import load_dotenv

# Load environment variables from .env file (if present)
load_dotenv()

# Configurable Gemini model name (easy to swap)
GEMINI_MODEL_NAME = "gemini-2.5-flash"

# Single unified prompt template enforcing all grounding and viva constraints
EXPLANATION_PROMPT_TEMPLATE = """You are an expert fraud-analysis assistant for an SMS security pipeline.
A machine learning classifier flagged the message below as "fraud".
Evaluate the message using ONLY the provided advisory notes below.

Message to evaluate:
"{message}"

Retrieved Advisory Notes:
{notes_block}

Strict Instructions:
1. Explain why the message looks fraudulent using ONLY the facts in the provided notes above.
2. Cite the notes inline using brackets like [1], [2] to support your points, and list the cited sources at the end with their Title and URL.
3. HONESTY RULE: If the notes do not support the message being fraud (for example, if the message appears to be a legitimate bank transaction alert or genuine OTP), state clearly that the retrieved notes do not show a fraud pattern and that the classifier may be wrong (false positive). Do not invent reasons.
4. Do NOT use outside knowledge, unverified assumptions, or imaginary facts.
5. Keep your entire response under 120 words in simple, direct English.
"""


def _format_notes_block(notes: List[Dict[str, Any]]) -> str:
    """Formats the list of retrieved notes into a numbered reference block for the LLM."""
    formatted_items = []
    for idx, item in enumerate(notes, start=1):
        formatted_items.append(
            f"[{idx}] Title: {item.get('title', 'Advisory')}\n"
            f"    Source: {item.get('source_name', 'Official Advisory')} ({item.get('url', 'N/A')})\n"
            f"    Advisory: {item.get('note', '')}"
        )
    return "\n\n".join(formatted_items)


def explain(message: str, notes: List[Dict[str, Any]]) -> str:
    """
    Generates a grounded explanation for a flagged message using retrieved notes.

    Args:
        message (str): The suspicious SMS or text message.
        notes (List[Dict[str, Any]]): Retrieved advisory notes from the RAG retriever.

    Returns:
        str: Grounded explanation with citations, or an honest false-positive disclaimer,
             or a fallback if no notes were retrieved / API key is missing.
    """
    # Rule 1: If retriever returned nothing, skip LLM call completely
    if not notes:
        return "No relevant fraud-advisory note was found for this message, so I can't give a grounded explanation."

    # Read GEMINI_API_KEY from environment
    api_key = os.getenv("GEMINI_API_KEY", "").strip()

    # Friendly check if API key is not yet set or is still the placeholder
    if not api_key or api_key == "your_key_here":
        return (
            "Gemini API key is not configured. Please add your GEMINI_API_KEY to the .env file "
            "to generate grounded LLM explanations."
        )

    # Format the prompt
    notes_block = _format_notes_block(notes)
    prompt = EXPLANATION_PROMPT_TEMPLATE.format(
        message=message,
        notes_block=notes_block
    )

    # Call Gemini API via the modern google-genai SDK
    try:
        from google import genai

        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=GEMINI_MODEL_NAME,
            contents=prompt
        )

        if response and response.text:
            return response.text.strip()
        return "Explanation could not be generated: received empty response from LLM."

    except Exception as e:
        # Graceful error handling (does not crash the app or pipeline)
        return (
            f"Unable to generate explanation due to an API error: {str(e)}. "
            "Please check your GEMINI_API_KEY and network connection."
        )


if __name__ == "__main__":
    # Test script directly
    sample_msg = "Your account is blocked due to pending KYC. Click http://bit.ly/bank-kyc to update."
    dummy_notes = [
        {
            "id": "kb01",
            "title": "Fake KYC Verification and Account Blocked SMS Scam",
            "note": "Scammers send urgent text messages claiming that your bank account is blocked due to pending KYC verification. Genuine banks never suspend accounts without formal notice or mandate KYC updates via random SMS links.",
            "source_name": "RBI - BE(A)WARE Booklet on Modus Operandi of Financial Frauds",
            "url": "https://www.rbi.org.in/scripts/BS_PressReleaseDisplay.aspx?prid=53372",
            "score": 0.82
        }
    ]

    print("[*] Testing explain() with sample KYC scam note...")
    explanation = explain(sample_msg, dummy_notes)
    print("\n--- Explanation Output ---")
    print(explanation)

