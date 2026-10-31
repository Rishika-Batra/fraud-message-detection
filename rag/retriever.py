"""
retriever.py
Fraud Message Detection

Purpose:
Retrieves the most semantically relevant fraud advisory notes from our FAISS index.
1. Encodes the input query using the local SentenceTransformer ('all-MiniLM-L6-v2').
2. Queries the pre-built FAISS index in rag/index/ for the top-k nearest neighbors.
3. Applies a similarity threshold (SIMILARITY_THRESHOLD = 0.35).
   - If the top match score is below this threshold, returns an empty list []
     (indicating the message has no matching fraud advisory in the knowledge base).
4. Caches the model and index in memory so subsequent calls are fast.
"""

import json
import os
import sys
from typing import List, Dict, Any

# Paths relative to this file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_DIR = os.path.join(BASE_DIR, "index")
INDEX_FILE = os.path.join(INDEX_DIR, "kb.index")
METADATA_FILE = os.path.join(INDEX_DIR, "metadata.json")

# Model name and tunable similarity threshold
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
SIMILARITY_THRESHOLD = 0.35  # Scores range from 0.0 to 1.0 (cosine similarity)

# Module-level singletons for lazy-loading
_embedder = None
_index = None
_metadata = None


def _load_retrieval_resources():
    """
    Loads the embedding model, FAISS index, and note metadata into memory once.
    Re-uses cached instances on subsequent calls to avoid disk/GPU reload overhead.
    """
    global _embedder, _index, _metadata

    if _embedder is not None and _index is not None and _metadata is not None:
        return

    # Check if index exists; if not, raise an informative error
    if not os.path.exists(INDEX_FILE) or not os.path.exists(METADATA_FILE):
        raise FileNotFoundError(
            f"FAISS index or metadata not found in '{INDEX_DIR}'. "
            "Please build the index first by running: python rag/build_index.py"
        )

    # Lazy-load sentence-transformers and FAISS
    from sentence_transformers import SentenceTransformer
    import faiss

    # Load FAISS index and metadata
    _index = faiss.read_index(INDEX_FILE)
    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        _metadata = json.load(f)

    # Load embedding model
    _embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)


def retrieve(query: str, k: int = 3, threshold: float = SIMILARITY_THRESHOLD) -> List[Dict[str, Any]]:
    """
    Finds top-k most relevant fraud advisory notes for a given input message.

    Args:
        query (str): The SMS or message text to search against the knowledge base.
        k (int): Number of top notes to retrieve (default: 3).
        threshold (float): Minimum cosine similarity score. If the best score
                           falls below this threshold, returns an empty list [].

    Returns:
        List[Dict[str, Any]]: List of retrieved notes sorted by similarity score,
                              or empty list [] if top score < threshold.
                              Each dict contains: id, title, note, source_name, url, score.
    """
    if not query or not str(query).strip():
        return []

    _load_retrieval_resources()

    # Encode query and normalize so dot-product equals cosine similarity
    query_vector = _embedder.encode(
        [str(query).strip()],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    ).astype("float32")

    # Search top k in FAISS
    # distances are inner products of normalized vectors -> cosine similarity
    distances, indices = _index.search(query_vector, k)

    scores = distances[0]
    top_indices = indices[0]

    # Check if the best score meets the minimum relevance threshold
    if len(scores) == 0 or scores[0] < threshold:
        return []

    results = []
    for score, idx in zip(scores, top_indices):
        if idx < 0 or idx >= len(_metadata):
            continue
        # Also ensure individual items added have a meaningful score
        if score < threshold:
            continue

        item = _metadata[idx].copy()
        item["score"] = round(float(score), 4)
        results.append(item)

    return results


if __name__ == "__main__":
    # Test directly from the command line: python rag/retriever.py "your message"
    test_query = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "Dear user, your electricity power will be disconnected tonight. Call officer at 9876543210 immediately."
    )

    print(f"\n[Retriever Test Query]: \"{test_query}\"")
    retrieved_notes = retrieve(test_query, k=3)

    if not retrieved_notes:
        print(f"No notes met the similarity threshold ({SIMILARITY_THRESHOLD}).")
    else:
        print(f"Retrieved {len(retrieved_notes)} relevant note(s):\n")
        for i, note in enumerate(retrieved_notes, 1):
            print(f"[{i}] {note['title']} (Score: {note['score']})")
            print(f"    Source: {note['source_name']} ({note['url']})")
            print(f"    Note:   {note['note']}\n")

