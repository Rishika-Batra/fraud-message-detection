"""
train_distilbert.py
College Project: Fraud / Spam SMS Detection

This script fine-tunes a pre-trained Transformer model (DistilBERT)
for binary text classification (0 = Legitimate / Ham, 1 = Fraud / Spam).

Pipeline:
1. Automatically selects the best available compute hardware (CUDA, Apple Silicon MPS, or CPU).
2. Loads tokenized text using the Hugging Face DistilBERT tokenizer (max length: 128).
3. Creates PyTorch Datasets and DataLoaders (batch size: 16).
4. Loads DistilBERT with a 2-class classification head (distilbert-base-uncased).
5. Fine-tunes the model for 3 epochs using the AdamW optimizer (learning rate: 2e-5).
6. Records and visualizes training & validation loss across epochs, saving model/results/loss_curve.png.
7. Evaluates performance on the unseen test set (data/test.csv):
   - Computes Accuracy, Precision, Recall, and F1 for the Fraud class (label 1).
   - Saves metrics to model/results/distilbert_metrics.json.
   - Saves a Confusion Matrix plot to model/results/confusion_matrix.png.
   - Saves the first 10 misclassified test examples to model/results/misclassified_examples.csv.
8. Saves the final trained model and tokenizer to model/saved_model/.
"""

import json
import os
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

# ==========================================
# 1. Custom PyTorch Dataset for SMS Messages
# ==========================================
class SMSDataset(Dataset):
    """
    Wraps SMS messages and numeric labels for PyTorch DataLoader.
    Tokenizes text on-the-fly to token IDs, attention masks, and tensors.
    """
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        # Tokenize the input text with truncation and padding up to max_length
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),         # Token IDs
            "attention_mask": encoding["attention_mask"].squeeze(0), # 1 for real tokens, 0 for padding
            "labels": torch.tensor(label, dtype=torch.long)          # Target class (0 or 1)
        }

def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)

    # ==========================================
    # 2. Hardware Selection (CUDA / MPS / CPU)
    # ==========================================
    # Automatically pick GPU if available (CUDA for NVIDIA, MPS for Apple Silicon Mac), else CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # ==========================================
    # 3. Setup File Paths & Output Directory
    # ==========================================
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(script_dir, ".."))
    data_dir = os.path.join(repo_root, "data")
    results_dir = os.path.join(script_dir, "results")
    saved_model_dir = os.path.join(script_dir, "saved_model")

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(saved_model_dir, exist_ok=True)

    train_path = os.path.join(data_dir, "train.csv")
    val_path = os.path.join(data_dir, "val.csv")
    test_path = os.path.join(data_dir, "test.csv")

    # ==========================================
    # 4. Load Data & Initialize Tokenizer
    # ==========================================
    print("Loading datasets...")
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    # DistilBERT is a lightweight, distilled version of BERT that retains ~97% of its performance
    model_name = "distilbert-base-uncased"
    print(f"Loading tokenizer: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Hyperparameters requested
    batch_size = 16
    max_length = 128
    epochs = 3
    learning_rate = 2e-5

    # Create PyTorch datasets
    train_dataset = SMSDataset(train_df["message"], train_df["label"], tokenizer, max_length)
    val_dataset = SMSDataset(val_df["message"], val_df["label"], tokenizer, max_length)
    test_dataset = SMSDataset(test_df["message"], test_df["label"], tokenizer, max_length)

    # Create DataLoaders for mini-batch processing
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # ==========================================
    # 5. Load Pre-trained DistilBERT Model
    # ==========================================
    print(f"Loading pre-trained model: {model_name} (num_labels=2)...")
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model.to(device)

    # Optimizer: AdamW with weight decay helps prevent overfitting during fine-tuning
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # ==========================================
    # 6. Training & Validation Loop
    # ==========================================
    train_losses = []
    val_losses = []

    print("\nStarting DistilBERT fine-tuning...")
    print(f"Total Epochs: {epochs} | Batch Size: {batch_size} | Learning Rate: {learning_rate}")
    print("-" * 55)

    for epoch in range(1, epochs + 1):
        # --- Training Phase ---
        model.train()
        total_train_loss = 0.0

        for batch in train_loader:
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Forward pass: model automatically calculates cross-entropy loss when 'labels' are passed
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            # Backward pass: compute gradients
            loss.backward()

            # Update weights
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # --- Validation Phase ---
        model.eval()
        total_val_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                total_val_loss += outputs.loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    print("-" * 55)
    print("Fine-tuning completed successfully!")

    # ==========================================
    # 7. Save Loss Curve
    # ==========================================
    loss_curve_path = os.path.join(results_dir, "loss_curve.png")
    plt.figure(figsize=(7, 4.5))
    epochs_range = range(1, epochs + 1)
    plt.plot(epochs_range, train_losses, marker="o", color="#1f77b4", label="Train Loss", linewidth=2)
    plt.plot(epochs_range, val_losses, marker="s", color="#ff7f0e", label="Validation Loss", linewidth=2)
    plt.title("DistilBERT Training and Validation Loss Curve", fontsize=13, fontweight="bold")
    plt.xlabel("Epoch", fontsize=11)
    plt.ylabel("Cross-Entropy Loss", fontsize=11)
    plt.xticks(epochs_range)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(loss_curve_path, dpi=300)
    plt.close()
    print(f"Loss curve saved to: {loss_curve_path}")

    # ==========================================
    # 8. Test Set Evaluation
    # ==========================================
    print("\nEvaluating on unseen test set...")
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(batch["labels"].tolist())

    # Calculate metrics for the positive/fraud class (label 1)
    test_acc = accuracy_score(all_labels, all_preds)
    test_prec = precision_score(all_labels, all_preds, pos_label=1)
    test_rec = recall_score(all_labels, all_preds, pos_label=1)
    test_f1 = f1_score(all_labels, all_preds, pos_label=1)

    metrics = {
        "accuracy": round(float(test_acc), 4),
        "precision": round(float(test_prec), 4),
        "recall": round(float(test_rec), 4),
        "f1": round(float(test_f1), 4)
    }

    # Save metrics to JSON
    metrics_path = os.path.join(results_dir, "distilbert_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # Print results to console
    print("=" * 45)
    print("DISTILBERT TEST EVALUATION RESULTS")
    print("=" * 45)
    print(f"Accuracy:                {metrics['accuracy']:.4f}")
    print(f"Fraud Precision (pos=1): {metrics['precision']:.4f}")
    print(f"Fraud Recall    (pos=1): {metrics['recall']:.4f}")
    print(f"Fraud F1-Score  (pos=1): {metrics['f1']:.4f}")
    print("=" * 45)
    print(f"Metrics saved to: {metrics_path}")

    # ==========================================
    # 9. Confusion Matrix Image
    # ==========================================
    cm_path = os.path.join(results_dir, "confusion_matrix.png")
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Legitimate (0)", "Fraud (1)"]
    )
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(cmap="Blues", ax=ax, values_format="d")
    plt.title("DistilBERT Test Confusion Matrix", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"Confusion matrix image saved to: {cm_path}")

    # ==========================================
    # 10. Save First 10 Misclassified Examples
    # ==========================================
    misclassified_path = os.path.join(results_dir, "misclassified_examples.csv")
    misclassified_indices = [
        i for i, (y_true, y_hat) in enumerate(zip(all_labels, all_preds)) if y_true != y_hat
    ]

    print(f"\nTotal misclassified test examples: {len(misclassified_indices)} out of {len(all_labels)}")
    
    # Take up to the first 10 misclassified examples
    first_10_indices = misclassified_indices[:10]
    misclassified_df = pd.DataFrame({
        "message": [test_df.iloc[i]["message"] for i in first_10_indices],
        "true_label": [all_labels[i] for i in first_10_indices],
        "predicted_label": [all_preds[i] for i in first_10_indices]
    })
    misclassified_df.to_csv(misclassified_path, index=False)
    print(f"Saved first {len(misclassified_df)} misclassified examples to: {misclassified_path}")

    # ==========================================
    # 11. Save Fine-Tuned Model & Tokenizer
    # ==========================================
    print(f"\nSaving model and tokenizer to: {saved_model_dir}...")
    model.save_pretrained(saved_model_dir)
    tokenizer.save_pretrained(saved_model_dir)
    print("Model and tokenizer saved successfully!")

if __name__ == "__main__":
    main()
