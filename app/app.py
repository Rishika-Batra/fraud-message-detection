"""
app.py
Fraud Message Detection

Purpose:
Interactive Gradio web demo for Fraud Message Detection with Grounded Explanations.
Integrates:
- Fine-tuned DistilBERT classifier
- FAISS knowledge retrieval from trusted Indian cyber-advisories
- Grounded Gemini explanation with inline citations and false-positive honesty

Usage:
    python app/app.py
"""

import os
import sys
import gradio as gr

# Ensure root directory is in python search path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from app.pipeline import analyze


def process_message(message_text: str):
    """
    Handles user submission from the Gradio interface:
    1. Runs the message through the analysis pipeline.
    2. Formats label, percentage confidence, explanation, and clickable sources.
    """
    if not message_text or not message_text.strip():
        return (
            "N/A",
            "0.0%",
            "Please enter or paste an SMS message to analyze.",
            "No sources to display."
        )

    result = analyze(message_text.strip())

    label = result.get("label", "Unknown").upper()
    confidence_val = result.get("confidence", 0.0)
    confidence_str = f"{confidence_val * 100:.1f}%"
    explanation = result.get("explanation", "No explanation available.")

    # Format sources into clickable markdown links
    sources_list = result.get("sources", [])
    if not sources_list:
        sources_md = "*No external advisories cited for this message.*"
    else:
        sources_md = "### Verified Sources & Official Advisories:\n"
        for idx, src in enumerate(sources_list, start=1):
            title = src.get("title", "Official Advisory")
            url = src.get("url", "#")
            sources_md += f"- **[{idx}]** [{title}]({url})\n"

    # Prepend visual badge to label
    if label == "FRAUD":
        formatted_label = f"🚨 FRAUD (Suspicious)"
    else:
        formatted_label = f"✅ LEGITIMATE (Safe)"

    return formatted_label, confidence_str, explanation, sources_md


# Curated example messages for testing and viva demonstration
EXAMPLE_MESSAGES = [
    [
        "Your bank account has been suspended due to incomplete KYC. Click http://bit.ly/bank-kyc-update to verify your identity immediately."
    ],
    [
        "URGENT: Your electricity power supply will be disconnected tonight at 9:30 PM due to unpaid bill. Call our electricity officer at 9876543210 immediately."
    ],
    [
        "Congratulations! You have won a cash prize of Rs 50,000 in the Diwali lucky draw. Click http://claim-prize.cc to claim your money."
    ],
    [
        "Earn Rs 2000 to 5000 daily working from home just by rating 5-star Google map locations and YouTube videos. Contact HR on WhatsApp: 9811223344."
    ],
    [
        "Hey, are we meeting at the campus library at 4 PM?"
    ]
]

# Custom CSS for clean presentation during viva
CUSTOM_CSS = """
.container { max-width: 900px; margin: auto; }
.output-box { border-radius: 8px; }
"""

with gr.Blocks(title="Fraud Message Detection Demo") as demo:
    gr.Markdown(
        """
        # 🛡️ Fraud Message Detection with Grounded Explanations
        This pipeline detects fraudulent SMS messages using a fine-tuned **DistilBERT** classifier and generates **grounded, cited explanations** using **RAG (FAISS + MiniLM)** and **Google Gemini**.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            input_message = gr.Textbox(
                lines=5,
                placeholder="Paste SMS text message here (e.g., KYC alert, bank SMS, job offer)...",
                label="Incoming Message Text"
            )
            submit_btn = gr.Button("🔍 Analyze Message", variant="primary")
            clear_btn = gr.Button("Clear")

        with gr.Column(scale=1):
            out_label = gr.Textbox(label="Classification Label", interactive=False)
            out_confidence = gr.Textbox(label="Model Confidence", interactive=False)
            out_explanation = gr.Markdown(label="Grounded Explanation")
            out_sources = gr.Markdown(label="Official Advisory Sources")

    gr.Markdown("### Try Example Messages:")
    gr.Examples(
        examples=EXAMPLE_MESSAGES,
        inputs=[input_message],
        label="Click any example to load it into the analyzer:"
    )

    submit_btn.click(
        fn=process_message,
        inputs=[input_message],
        outputs=[out_label, out_confidence, out_explanation, out_sources]
    )

    clear_btn.click(
        fn=lambda: ("", "", "", "", ""),
        inputs=[],
        outputs=[input_message, out_label, out_confidence, out_explanation, out_sources]
    )


if __name__ == "__main__":
    print("[*] Starting Gradio web demo at http://127.0.0.1:7860 ...")
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)

