"""
prepare_data.py
College Project: Fraud / Spam SMS Detection

This script performs the complete data preparation pipeline:
1. Downloads the UCI SMS Spam Collection dataset directly from the official repository.
2. Extracts and parses the tab-separated SMS data into a pandas DataFrame.
3. Removes duplicate SMS messages to avoid data leakage and model bias.
4. Encodes labels numerically: 'ham' -> 0 (legitimate) and 'spam' -> 1 (fraud/spam).
5. Performs a stratified split (80% train, 10% validation, 10% test) with random seed 42,
   maintaining the exact class ratio across all splits.
6. Saves the splits as data/train.csv, data/val.csv, and data/test.csv.
7. Prints message counts and class distributions for viva presentation.
"""

import io
import os
import urllib.request
import zipfile
import pandas as pd
from sklearn.model_selection import train_test_split

# URL to the official UCI SMS Spam Collection dataset zip archive
DATASET_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"

def main():
    print("Step 1: Downloading UCI SMS Spam Collection dataset...")
    # Add a standard user-agent header to avoid potential bot blocks
    request = urllib.request.Request(
        DATASET_URL,
        headers={"User-Agent": "Mozilla/5.0"}
    )
    with urllib.request.urlopen(request) as response:
        zip_bytes = response.read()

    print("Step 2: Extracting dataset from zip...")
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
        # The file inside the archive is named 'SMSSpamCollection'
        with z.open("SMSSpamCollection") as f:
            raw_data = f.read().decode("utf-8", errors="replace")

    # The dataset is tab-separated with no header:
    # Column 0: label ('ham' or 'spam')
    # Column 1: SMS message text
    df = pd.read_csv(
        io.StringIO(raw_data),
        sep="\t",
        header=None,
        names=["raw_label", "message"],
        quoting=3  # csv.QUOTE_NONE prevents parsing errors from quotes inside SMS
    )
    print(f"Total raw messages downloaded: {len(df)}")

    # Step 3: Remove duplicate messages
    # In SMS datasets, identical spam messages often repeat.
    # Removing duplicates prevents identical messages from appearing across both train and test splits.
    df = df.drop_duplicates(subset=["message"]).reset_index(drop=True)
    print(f"Messages after removing duplicates: {len(df)}")

    # Step 4: Map labels to numeric values ('ham' -> 0, 'spam' -> 1)
    # Machine learning and NLP models require numerical targets.
    df["label"] = df["raw_label"].map({"ham": 0, "spam": 1})

    # Keep only the target label and text message columns
    df = df[["label", "message"]]

    # Step 5: Stratified Split (80% train, 10% validation, 10% test)
    # Why stratified? Spam datasets are highly imbalanced (~87% ham, ~13% spam).
    # Stratification ensures each split has the exact same class proportions.
    
    # First split: 80% train and 20% temporary set (val + test)
    train_df, temp_df = train_test_split(
        df,
        test_size=0.20,
        random_state=42,
        stratify=df["label"]
    )

    # Second split: Split the remaining 20% equally (50/50) into 10% val and 10% test
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=42,
        stratify=temp_df["label"]
    )

    # Step 6: Save splits to CSV in data/ directory
    # Find the repository root (parent directory of 'model/')
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(script_dir, ".."))
    data_dir = os.path.join(repo_root, "data")
    os.makedirs(data_dir, exist_ok=True)

    train_path = os.path.join(data_dir, "train.csv")
    val_path = os.path.join(data_dir, "val.csv")
    test_path = os.path.join(data_dir, "test.csv")

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    print("\nFiles successfully saved:")
    print(f"  - Train: {train_path}")
    print(f"  - Val:   {val_path}")
    print(f"  - Test:  {test_path}")

    # Step 7: Print message counts for each class in every split
    print("\n" + "="*50)
    print("DATASET SPLIT SUMMARY (Class Distribution)")
    print("="*50)

    def print_split_summary(split_name, split_data):
        ham_count = (split_data["label"] == 0).sum()
        spam_count = (split_data["label"] == 1).sum()
        total = len(split_data)
        ham_pct = (ham_count / total) * 100
        spam_pct = (spam_count / total) * 100
        print(f"\n{split_name} Set:")
        print(f"  Total messages: {total}")
        print(f"  Ham  (0): {ham_count:4d} ({ham_pct:.2f}%)")
        print(f"  Spam (1): {spam_count:4d} ({spam_pct:.2f}%)")

    print_split_summary("Train (80%)", train_df)
    print_split_summary("Validation (10%)", val_df)
    print_split_summary("Test (10%)", test_df)
    print("="*50)

if __name__ == "__main__":
    main()
