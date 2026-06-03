"""
Semantic retrieval using FAISS IndexFlatIP (inner-product = cosine on L2-normalised vectors).
Used for the high-confidence path: retrieve the most relevant counseling response
from the Amod dataset given a user query + predicted intent.
"""

import numpy as np
import faiss
import streamlit as st
from sentence_transformers import SentenceTransformer


@st.cache_resource(show_spinner=False)
def load_encoder():
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_data(show_spinner=False)
def build_index(_df):
    """
    Encode every user_input row and store in a FAISS IndexFlatIP.
    Vectors are L2-normalised so inner product == cosine similarity.
    Returns (faiss_index, original_row_positions, column_name).
    """
    encoder = load_encoder()
    col     = "user_input" if "user_input" in _df.columns else "bot_response"
    texts   = _df[col].fillna("").tolist()

    embeddings = encoder.encode(
        texts, batch_size=64, show_progress_bar=False, normalize_embeddings=True
    ).astype("float32")

    dim   = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)   # Inner-product index (cosine on unit vectors)
    index.add(embeddings)

    return index, list(range(len(texts))), col


def semantic_retrieve(user_text, df, intent):
    """
    Return the df row whose user_input is semantically closest to user_text,
    restricted to rows matching `intent`. Uses FAISS k-NN search (k=1).
    Returns None if the intent subset is empty.
    """
    subset = df[df["intent"] == intent]
    if subset.empty:
        return None

    encoder = load_encoder()
    full_index, _, col = build_index(df)

    # Encode query
    query_vec = encoder.encode(
        [user_text], normalize_embeddings=True
    )[0].astype("float32").reshape(1, -1)

    # Build a sub-index restricted to the intent subset (k-NN in subset only)
    idx      = subset.index.tolist()
    sub_embs = np.array([
        full_index.reconstruct(i) for i in idx
    ], dtype="float32")

    dim       = sub_embs.shape[1]
    sub_index = faiss.IndexFlatIP(dim)
    sub_index.add(sub_embs)

    _, I      = sub_index.search(query_vec, k=1)   # k-NN search
    best_local  = int(I[0][0])
    best_global = idx[best_local]

    return df.loc[best_global]
