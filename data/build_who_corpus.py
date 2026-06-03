"""
Build the knowledge base for the low-confidence RAG-Lite path.

Priority order:
  1. WHO PDFs manually placed in data/who_corpus/  (most authentic)
  2. Auto-download WHO PDFs from iris.who.int       (attempted, validated)
  3. HuggingFace: nbertagnolli/counsel-chat          (automatic fallback —
     real counseling Q&A data from an online therapy platform, not synthetic)

Steps:
  1. Collect text from any of the above sources
  2. EDA on the corpus
  3. Clean & chunk into ~256-word overlapping segments
  4. Embed with all-MiniLM-L6-v2
  5. Save FAISS index  → data/faiss_who.index
     Save chunk store  → data/faiss_who_chunks.pkl

Run once after prepare_data.py:
    python data/build_who_corpus.py
"""

import os
import re
import sys
import pickle
import requests
import pdfplumber
import numpy as np
import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer

# Windows consoles default to cp1252 and crash on the box-drawing characters
# used in the progress output. Force UTF-8 so the script runs anywhere.
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

DATA_DIR   = os.path.dirname(os.path.abspath(__file__))
CORPUS_DIR = os.path.join(DATA_DIR, "who_corpus")
os.makedirs(CORPUS_DIR, exist_ok=True)

# ── WHO documents on iris.who.int (public domain) ─────────────────────────────
# iris.who.int now runs DSpace 7 (an Angular SPA). The old
# /bitstream/handle/<id>/<file>.pdf links return the app shell, NOT the PDF.
# The real download lives behind the DSpace REST API, reached by resolving the
# persistent handle → item → ORIGINAL bundle → English bitstream → /content.
IRIS_API = "https://iris.who.int/server/api"

WHO_DOCUMENTS = [
    {
        "name"  : "WHO_Mental_Health_Action_Plan_2013-2030.pdf",
        "handle": "10665/345301",
        "source": "WHO Mental Health Action Plan 2013–2030",
    },
    {
        "name"  : "WHO_mhGAP_Intervention_Guide_v2.pdf",
        "handle": "10665/250239",
        "source": "WHO mhGAP Intervention Guide v2.0",
    },
    {
        "name"  : "WHO_Preventing_Suicide_Global_Imperative.pdf",
        "handle": "10665/131056",
        "source": "WHO — Preventing Suicide: A Global Imperative",
    },
    {
        "name"  : "WHO_Mental_Health_Atlas_2020.pdf",
        "handle": "10665/345946",
        "source": "WHO Mental Health Atlas 2020",
    },
]

_HTTP_HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}


def resolve_who_pdf_url(handle: str) -> str | None:
    """
    Resolve a WHO iris handle (e.g. '10665/250239') to the direct download
    URL of its English PDF via the DSpace 7 REST API.

    Returns the /content URL string, or None if it can't be resolved.
    """
    try:
        # 1. handle → item
        item = requests.get(
            f"{IRIS_API}/pid/find", params={"id": f"hdl:{handle}"},
            headers={**_HTTP_HEADERS, "Accept": "application/json"}, timeout=60,
        ).json()
        item_id = item["uuid"]

        # 2. item → bundles → ORIGINAL
        bundles = requests.get(
            f"{IRIS_API}/core/items/{item_id}/bundles",
            headers={**_HTTP_HEADERS, "Accept": "application/json"}, timeout=60,
        ).json()
        original = next(
            (b for b in bundles["_embedded"]["bundles"] if b["name"] == "ORIGINAL"),
            None,
        )
        if original is None:
            return None

        # 3. ORIGINAL → bitstreams
        bitstreams = requests.get(
            original["_links"]["bitstreams"]["href"],
            headers={**_HTTP_HEADERS, "Accept": "application/json"}, timeout=60,
        ).json()["_embedded"]["bitstreams"]

        pdfs = [b for b in bitstreams if b["name"].lower().endswith(".pdf")]
        if not pdfs:
            return None

        # 4. Prefer the English edition (-eng.pdf), else the largest PDF.
        eng = next((b for b in pdfs if "-eng" in b["name"].lower()), None)
        chosen = eng or max(pdfs, key=lambda b: b.get("sizeBytes", 0))
        return chosen["_links"]["content"]["href"]

    except Exception as e:
        print(f"             [resolve failed] {e}")
        return None

# ── helpers ───────────────────────────────────────────────────────────────────
def is_valid_pdf(path: str) -> bool:
    """Check magic bytes — real PDFs start with %PDF."""
    try:
        with open(path, "rb") as f:
            return f.read(4) == b"%PDF"
    except Exception:
        return False


def extract_pdf_text(pdf_path: str) -> list:
    """Return list of (page_num, text) from a PDF."""
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, 1):
            t = page.extract_text()
            pages.append((i, t or ""))
    return pages


def chunk_text(text: str, chunk_size: int = 256, overlap: int = 50) -> list:
    """Overlapping ~256-word chunks."""
    text  = re.sub(r"\s+", " ", text).strip()
    words = text.split()
    step  = max(1, chunk_size - overlap)
    return [
        " ".join(words[i : i + chunk_size])
        for i in range(0, max(1, len(words) - overlap), step)
        if len(" ".join(words[i : i + chunk_size]).strip()) > 60
    ]


def is_reference_chunk(text: str) -> bool:
    """
    Detect bibliography / reference-list chunks so they don't get indexed.
    WHO PDFs end with long citation lists ("Bohnert KM, Ilgen MA, ... Addiction.
    2014;...") which are useless as conversational knowledge and look broken
    if ever surfaced to a user.
    """
    # Journal-citation cues: "et al", DOIs, "doi:", volume;issue:page patterns,
    # and "Author AB, Author CD," style initials.
    cue_hits = len(re.findall(
        r"\bet al\b|\bdoi:|\bvol\.|;\s*\d+\s*[:(]|\b\d{4};\d+|\bpp?\.\s*\d+",
        text, flags=re.IGNORECASE,
    ))
    # Count "Surname AB," author-initial groups — dense in reference lists.
    author_inits = len(re.findall(r"[A-Z][a-z]+ [A-Z]{1,3}(?:,|\.)", text))
    # Lots of 4-digit years also signals a citation block.
    years = len(re.findall(r"\b(19|20)\d{2}\b", text))
    return cue_hits >= 2 or author_inits >= 6 or years >= 6


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1 — collect raw documents
# ═══════════════════════════════════════════════════════════════════════════════
raw_documents = []   # list of {"source": str, "text": str}

# ── 1a. Manual PDFs already in who_corpus/ ────────────────────────────────────
manual_pdfs = [
    f for f in os.listdir(CORPUS_DIR)
    if f.lower().endswith(".pdf") and is_valid_pdf(os.path.join(CORPUS_DIR, f))
]
if manual_pdfs:
    print(f"Found {len(manual_pdfs)} valid PDF(s) in {CORPUS_DIR}/")
    for fname in manual_pdfs:
        path = os.path.join(CORPUS_DIR, fname)
        print(f"  Parsing {fname} ...")
        try:
            pages = extract_pdf_text(path)
            full  = "\n".join(t for _, t in pages if t.strip())
            raw_documents.append({"source": fname.replace(".pdf", ""), "text": full})
        except Exception as e:
            print(f"    [WARNING] {e}")

# ── 1b. Auto-download WHO PDFs (attempt — validate bytes) ─────────────────────
if not raw_documents:
    print("── Attempting WHO PDF downloads from iris.who.int ───────────")
    for doc in WHO_DOCUMENTS:
        dest = os.path.join(CORPUS_DIR, doc["name"])
        # skip if already a valid cached PDF
        if os.path.exists(dest) and is_valid_pdf(dest):
            print(f"  [cached]   {doc['name']}")
            try:
                pages = extract_pdf_text(dest)
                full  = "\n".join(t for _, t in pages if t.strip())
                raw_documents.append({"source": doc["source"], "text": full})
            except Exception as e:
                print(f"    [WARNING] {e}")
            continue

        print(f"  [resolve]  {doc['name']}  (handle {doc['handle']}) ...")
        try:
            pdf_url = resolve_who_pdf_url(doc["handle"])
            if not pdf_url:
                raise ValueError("could not resolve PDF download URL via REST API")

            print(f"  [download] {pdf_url}")
            r = requests.get(
                pdf_url, timeout=120,
                headers=_HTTP_HEADERS, allow_redirects=True,
            )
            r.raise_for_status()

            # Validate — must be a real PDF
            if not r.content.startswith(b"%PDF"):
                raise ValueError(
                    f"Not a PDF — server returned {len(r.content)} bytes "
                    f"starting with: {r.content[:80]!r}"
                )

            with open(dest, "wb") as f:
                f.write(r.content)
            print(f"             saved ({len(r.content)//1024:,} KB)")

            pages = extract_pdf_text(dest)
            full  = "\n".join(t for _, t in pages if t.strip())
            raw_documents.append({"source": doc["source"], "text": full})

        except Exception as e:
            print(f"             [FAILED] {e}")
            # Remove the bad file so it doesn't get cached
            if os.path.exists(dest):
                os.remove(dest)

# ── 1c. HuggingFace fallback — nbertagnolli/counsel-chat ─────────────────────
if not raw_documents:
    print("\n── WHO PDFs unavailable. Loading fallback dataset from HuggingFace ──")
    print("   Dataset: nbertagnolli/counsel-chat")
    print("   Source : real counseling Q&A from an online therapy platform\n")

    from datasets import load_dataset
    ds = load_dataset("nbertagnolli/counsel-chat", split="train")
    df = ds.to_pandas()

    # ── EDA ──────────────────────────────────────────────────────────────────
    print("── EDA ──────────────────────────────────────────────────────")
    print(f"Shape      : {df.shape}")
    print(f"Columns    : {df.columns.tolist()}")
    print(f"\nNull values:\n{df.isnull().sum()}")
    print(f"\nDuplicate rows: {df.duplicated().sum()}")
    if "topic" in df.columns:
        print(f"\nTopic distribution:\n{df['topic'].value_counts().to_string()}")

    # ── Identify text columns ─────────────────────────────────────────────────
    # counsel-chat has: questionText, answerText, topic, upvotes, etc.
    answer_col   = next((c for c in df.columns if "answer" in c.lower()), None)
    question_col = next((c for c in df.columns if "question" in c.lower()), None)
    topic_col    = next((c for c in df.columns if "topic" in c.lower()), None)

    if answer_col is None:
        raise ValueError(f"Cannot find answer column in {df.columns.tolist()}")

    # ── Clean ─────────────────────────────────────────────────────────────────
    print(f"\n── Cleaning (using '{answer_col}' as knowledge text) ────────")
    before = len(df)
    df = df.dropna(subset=[answer_col])
    df = df.drop_duplicates(subset=[answer_col])
    df[answer_col] = df[answer_col].str.strip()
    df = df[df[answer_col].str.len() >= 100]
    print(f"After cleaning: {len(df):,} rows  (removed {before - len(df):,})")

    lengths = df[answer_col].str.len()
    print(f"Answer length — min:{lengths.min()}  max:{lengths.max()}  mean:{lengths.mean():.0f}")

    for _, row in df.iterrows():
        topic  = row[topic_col] if topic_col else "mental health"
        source = f"Counsel-Chat — {topic}"
        raw_documents.append({"source": source, "text": str(row[answer_col])})

    print(f"\nLoaded {len(raw_documents):,} counseling responses as knowledge base")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2 — chunk
# ═══════════════════════════════════════════════════════════════════════════════
if not raw_documents:
    print("\n[ERROR] No data sources available. Exiting.")
    exit(1)

all_chunks = []
dropped_refs = 0
print("\n── Chunking ─────────────────────────────────────────────────")
for doc in raw_documents:
    chunks = chunk_text(doc["text"])
    kept   = [c for c in chunks if not is_reference_chunk(c)]
    dropped_refs += len(chunks) - len(kept)
    src    = doc["source"][:55]
    print(f"  {src:<55} → {len(kept):>4} chunks  ({len(chunks)-len(kept)} refs dropped)")
    for c in kept:
        all_chunks.append({"source": doc["source"], "text": c})

print(f"\nTotal corpus: {len(all_chunks):,} chunks  ({dropped_refs} reference chunks filtered out)")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3 — embed + FAISS
# ═══════════════════════════════════════════════════════════════════════════════
print("\nEncoding with all-MiniLM-L6-v2 ...")
encoder    = SentenceTransformer("all-MiniLM-L6-v2")
texts      = [c["text"] for c in all_chunks]
embeddings = encoder.encode(
    texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True
).astype("float32")

dim   = embeddings.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(embeddings)
print(f"FAISS IndexFlatIP — {index.ntotal:,} vectors, dim={dim}")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4 — save
# ═══════════════════════════════════════════════════════════════════════════════
index_path  = os.path.join(DATA_DIR, "faiss_who.index")
chunks_path = os.path.join(DATA_DIR, "faiss_who_chunks.pkl")

faiss.write_index(index, index_path)
pickle.dump(all_chunks, open(chunks_path, "wb"))

print(f"\nSaved FAISS index  → {index_path}")
print(f"Saved chunk store  → {chunks_path}")
print("\nDone. Run: streamlit run app.py")
