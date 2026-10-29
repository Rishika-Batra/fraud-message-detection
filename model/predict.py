"""
predict.py
College Project: Fraud / Spam SMS Detection

This module exposes a predict(message) function for inference:
1. Lazy-loads the fine-tuned DistilBERT model and tokenizer once on the first call.
2. Selects the fastest available hardware accelerator (CUDA -> MPS -> CPU).
3. Tokenizes and runs inference on the input SMS message.
4. Returns a dictionary: {"label": "fraud" or "legitimate", "confidence": <float 0-1>}.
5. Can be imported by other modules (e.g. from model.predict import predict)
   or executed directly from the command line.
"""

import os
import sys
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Module-level variables to cache the loaded model and tokenizer in memory (singleton pattern)
_model = None
_tokenizer = None
_device = None

def _get_device():
    """Pick the fastest available compute device: CUDA -> Apple MPS -> CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def _load_model():
    """
    Loads the saved model and tokenizer from model/saved_model/.
    Called only once on the first predict() invocation to avoid reloading overhead.
    """
    global _model, _tokenizer, _device

    if _model is not None and _tokenizer is not None:
        return  # Already loaded into memory

    # Locate the saved_model directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    saved_model_dir = os.path.join(script_dir, "saved_model")

    # Show a clear, friendly error if the directory or weights do not exist
    if not os.path.exists(saved_model_dir) or not os.listdir(saved_model_dir):
        raise FileNotFoundError(
            f"Error: Saved model directory not found at '{saved_model_dir}'. "
            "Please train the model first by running 'python model/train_distilbert.py'."
        )

    # Determine hardware device
    _device = _get_device()

    # Load tokenizer and model weights
    _tokenizer = AutoTokenizer.from_pretrained(saved_model_dir)
    _model = AutoModelForSequenceClassification.from_pretrained(saved_model_dir)
    _model.to(_device)
    _model.eval()  # Put into evaluation mode (disables dropout)

def predict(message: str) -> dict:
    """
    Predicts whether an SMS message is legitimate or fraudulent.

    Args:
        message (str): The raw SMS message text.

    Returns:
        dict: {"label": "fraud" | "legitimate", "confidence": float}
    """
    # Step 1: Ensure model and tokenizer are loaded into memory
    _load_model()

    # Step 2: Tokenize input text (truncating up to 128 tokens as done during training)
    inputs = _tokenizer(
        str(message),
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors="pt"
    )

    # Move tensors to the compute device
    inputs = {k: v.to(_device) for k, v in inputs.items()}

    # Step 3: Run forward inference without computing gradients
    with torch.no_grad():
        outputs = _model(**inputs)
        # Apply softmax across the logits to get class probabilities
        probs = torch.softmax(outputs.logits, dim=-1).squeeze(0)
        pred_idx = torch.argmax(probs).item()
        confidence = round(float(probs[pred_idx].item()), 4)

    # Label 0: Legitimate (Ham), Label 1: Fraud (Spam)
    label_str = "fraud" if pred_idx == 1 else "legitimate"

    return {
        "label": label_str,
        "confidence": confidence
    }

if __name__ == "__main__":
    # Allow running directly from terminal: python model/predict.py "some message"
    if len(sys.argv) < 2:
        print('Usage: python model/predict.py "your SMS message here"')
        sys.exit(1)

    input_message = " ".join(sys.argv[1:])
    result = predict(input_message)
    print(f"\nMessage: \"{input_message}\"")
    print(f"Result:  {result}")
