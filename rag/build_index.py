"""
build_index.py
Fraud Message Detection

Purpose:
Builds a local semantic search index from our fraud knowledge base (rag/knowledge_base.json).
1. Combines each note's title and text.
2. Generates dense embeddings using sentence-transformers ('all-MiniLM-L6-v2').
3. Normalizes embeddings to unit length so that inner product equals cosine similarity.
4. Stores the vectors in a FAISS IndexFlatIP (Inner Product) index.
5. Saves the FAISS index (kb.index) and note metadata (metadata.json) into rag/index/.

Usage:
    python rag/build_index.py
"""

import json
import os
import sys
import numpy as np

# RAG paths relative to repository root
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KB_FILE = os.path.join(BASE_DIR, "knowledge_base.json")
INDEX_DIR = os.path.join(BASE_DIR, "index")
INDEX_FILE = os.path.join(INDEX_DIR, "kb.index")
METADATA_FILE = os.path.join(INDEX_DIR, "metadata.json")

# Embedding model name (runs 100% locally on CPU, no API key needed)
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


def build_index():
    """Reads knowledge_base.json, embeds notes, and builds the FAISS index."""
    # Step 1: Verify knowledge base exists
    if not os.path.exists(KB_FILE):
        print(f"Error: Knowledge base file not found at '{KB_FILE}'")
        sys.exit(1)

    with open(KB_FILE, "r", encoding="utf-8") as f:
        kb_entries = json.load(f)

    if not kb_entries:
        print("Error: Knowledge base is empty.")
        sys.exit(1)

    print(f"[*] Loaded {len(kb_entries)} entries from {os.path.basename(KB_FILE)}")

    # Step 2: Prepare combined text (title + note) for semantic embedding
    corpus_texts = []
    metadata = []
    for entry in kb_entries:
        # Combining title and note provides richer semantic context for queries
        combined_text = f"{entry['title']}: {entry['note']}"
        corpus_texts.append(combined_text)
        metadata.append({
            "id": entry.get("id"),
            "title": entry.get("title"),
            "note": entry.get("note"),
            "source_name": entry.get("source_name"),
            "url": entry.get("url")
        })

    # Step 3: Lazy-load sentence-transformers and FAISS
    print(f"[*] Loading embedding model '{EMBEDDING_MODEL_NAME}'...")
    try:
        from sentence_transformers import SentenceTransformer
        import faiss
    except ImportError as e:
        print(f"Error: Missing required package ({e}). Please run: pip install -r requirements.txt")
        sys.exit(1)

    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    # Step 4: Compute normalized embeddings
    # Normalizing makes the inner-product (dot product) identical to cosine similarity
    print(f"[*] Embedding {len(corpus_texts)} knowledge-base notes...")
    embeddings = model.encode(
        corpus_texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    ).astype("float32")

    dimension = embeddings.shape[1]  # 384 dimensions for all-MiniLM-L6-v2
    print(f"[*] Embedding dimension: {dimension}")

    # Step 5: Create FAISS IndexFlatIP (Inner Product)
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    print(f"[*] FAISS index built with {index.ntotal} vectors.")

    # Step 6: Save the FAISS index and metadata to rag/index/
    os.makedirs(INDEX_DIR, exist_ok=True)

    faiss.write_index(index, INDEX_FILE)
    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"[+] Index successfully saved to: {INDEX_FILE}")
    print(f"[+] Metadata successfully saved to: {METADATA_FILE}")
    print("[+] Done! You can now run the retriever or demo app.\n")


if __name__ == "__main__":
    build_index()

