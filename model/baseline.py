"""
baseline.py
College Project: Fraud / Spam SMS Detection

This script trains a baseline Machine Learning model:
1. Loads training and testing data from CSV files.
2. Converts raw text messages into numerical features using TF-IDF (Term Frequency-Inverse Document Frequency).
3. Trains a Logistic Regression classifier with class_weight='balanced' to handle class imbalance.
4. Predicts labels on the unseen test set (data/test.csv).
5. Computes evaluation metrics (Accuracy, Precision, Recall, F1) for the fraud class (label 1).
6. Saves the metrics to model/results/baseline_metrics.json and prints them for viva presentation.
"""

import json
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def main():
    # Step 1: Define file paths relative to the project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(script_dir, ".."))
    train_path = os.path.join(repo_root, "data", "train.csv")
    test_path = os.path.join(repo_root, "data", "test.csv")
    results_dir = os.path.join(script_dir, "results")
    metrics_path = os.path.join(results_dir, "baseline_metrics.json")

    # Step 2: Load the train and test datasets
    # train.csv is used for fitting, test.csv for final unbiased evaluation
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X_train, y_train = train_df["message"], train_df["label"]
    X_test, y_test = test_df["message"], test_df["label"]

    # Step 3: Convert text to numerical features using TF-IDF Vectorizer
    # TF-IDF measures how important a word is to a message relative to all messages.
    # We fit only on training data to prevent data leakage from the test set.
    vectorizer = TfidfVectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Step 4: Train Logistic Regression with balanced class weights
    # class_weight='balanced' automatically adjusts weights inversely proportional
    # to class frequencies, giving higher importance to rare fraud (spam) cases.
    clf = LogisticRegression(class_weight="balanced", random_state=42)
    clf.fit(X_train_tfidf, y_train)

    # Step 5: Evaluate the trained classifier on the test dataset
    y_pred = clf.predict(X_test_tfidf)

    # Step 6: Calculate metrics focusing on the positive/fraud class (label 1)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, pos_label=1)
    rec = recall_score(y_test, y_pred, pos_label=1)
    f1 = f1_score(y_test, y_pred, pos_label=1)

    metrics = {
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1": round(float(f1), 4)
    }

    # Step 7: Save metrics to JSON file for reporting and tracking
    os.makedirs(results_dir, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # Step 8: Print the evaluation results clearly for viva presentation
    print("=" * 45)
    print("BASELINE EVALUATION RESULTS (TF-IDF + LogReg)")
    print("=" * 45)
    print(f"Accuracy:                {metrics['accuracy']:.4f}")
    print(f"Fraud Precision (pos=1): {metrics['precision']:.4f}")
    print(f"Fraud Recall    (pos=1): {metrics['recall']:.4f}")
    print(f"Fraud F1-Score  (pos=1): {metrics['f1']:.4f}")
    print("=" * 45)
    print(f"Metrics saved to: {metrics_path}")

if __name__ == "__main__":
    main()
