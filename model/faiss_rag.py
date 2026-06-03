"""
Low-confidence RAG-Lite path (proposal §4.3).

Loads the FAISS index built from WHO/UNICEF guidelines and performs
a k-NN semantic search to retrieve the most relevant factual excerpt
when the MLP classifier confidence is below the 0.75 threshold.
"""

import os
import re
import pickle
import streamlit as st
import faiss
from sentence_transformers import SentenceTransformer

_DATA_DIR    = os.path.join(os.path.dirname(__file__), "..", "data")
_INDEX_PATH  = os.path.join(_DATA_DIR, "faiss_who.index")
_CHUNKS_PATH = os.path.join(_DATA_DIR, "faiss_who_chunks.pkl")

# Cosine-similarity floor (IndexFlatIP on normalized vectors == cosine).
# Below this, the "best" match is just noise — return nothing so the
# empathetic conversational pipeline handles the message instead of
# dumping an irrelevant corpus fragment.
_MIN_RELEVANCE = 0.35


def _tidy_excerpt(text: str, max_sentences: int = 3) -> str:
    """Trim a raw chunk to whole sentences so we don't show mid-sentence fragments."""
    text = re.sub(r"\s+", " ", text).strip()
    # Drop a leading partial sentence (chunk often starts mid-word/clause).
    first_cap = re.search(r"[A-Z]", text)
    if first_cap and first_cap.start() < 40:
        text = text[first_cap.start():]
    sentences = re.split(r"(?<=[.!?])\s+", text)
    sentences = [s for s in sentences if len(s.split()) >= 4]
    excerpt = " ".join(sentences[:max_sentences]).strip()
    return excerpt or text


@st.cache_resource(show_spinner=False)
def _load_who_index():
    """Load FAISS index + chunk store. Returns (index, chunks) or (None, None)."""
    if not os.path.exists(_INDEX_PATH) or not os.path.exists(_CHUNKS_PATH):
        return None, None
    index  = faiss.read_index(_INDEX_PATH)
    chunks = pickle.load(open(_CHUNKS_PATH, "rb"))
    return index, chunks


@st.cache_resource(show_spinner=False)
def _load_who_encoder():
    return SentenceTransformer("all-MiniLM-L6-v2")


def retrieve_from_corpus(user_text: str, k: int = 1) -> dict | None:
    """
    Find the most relevant WHO/UNICEF guideline chunk for user_text.

    Returns a dict {"text": str, "source": str} or None if the index
    has not been built yet (run data/build_who_corpus.py first).
    """
    index, chunks = _load_who_index()
    if index is None:
        return None

    encoder   = _load_who_encoder()
    query_vec = encoder.encode(
        [user_text], normalize_embeddings=True
    )[0].astype("float32").reshape(1, -1)

    D, I  = index.search(query_vec, k)
    score = float(D[0][0])      # cosine similarity (vectors are normalized)
    best  = int(I[0][0])

    # Reject weak matches — better to say nothing than to surface noise.
    if score < _MIN_RELEVANCE:
        return None

    chunk = dict(chunks[best])
    chunk["text"]  = _tidy_excerpt(chunk["text"])
    chunk["score"] = score
    return chunk
