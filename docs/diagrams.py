"""
Generate the explanatory diagrams used in the MindSpace DS project notes PDF.
Run via build_docs.py — produces PNGs in docs/_img/.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

IMG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_img")
os.makedirs(IMG_DIR, exist_ok=True)

# ── palette ───────────────────────────────────────────────────────────────────
PURPLE = "#6c5ce7"
PURPLE_L = "#ece9fb"
BLUE = "#2f80ed"
BLUE_L = "#e6f0fc"
GREEN = "#27ae60"
GREEN_L = "#e4f6ec"
ORANGE = "#e67e22"
ORANGE_L = "#fbeee0"
GREY = "#566573"
GREY_L = "#eceff1"
INK = "#2d3436"


def _box(ax, x, y, w, h, text, face, edge, fontsize=10, weight="normal", text_color=INK):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=2.2",
        linewidth=1.6, edgecolor=edge, facecolor=face, mutation_aspect=1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, color=text_color, weight=weight, wrap=True)


def _arrow(ax, x1, y1, x2, y2, color=GREY, label=None, ls="-"):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=16,
        linewidth=1.7, color=color, linestyle=ls,
        shrinkA=2, shrinkB=2, zorder=1))
    if label:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 1.6, label, ha="center",
                va="bottom", fontsize=8.5, color=color, weight="bold")


def _new(figsize=(10, 5.6)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    return fig, ax


def _save(fig, name):
    path = os.path.join(IMG_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


# ═══════════════════════════════════════════════════════════════════════════════
# 1. High-level overview
# ═══════════════════════════════════════════════════════════════════════════════
def overview():
    fig, ax = _new((10, 5.8))
    ax.text(50, 96, "MindSpace — How a message becomes a reply",
            ha="center", fontsize=13, weight="bold", color=INK)

    _box(ax, 4, 70, 20, 14, "User\n(types / speaks)", BLUE_L, BLUE, 10, "bold")
    _box(ax, 40, 70, 22, 14, "Streamlit\nWeb App", PURPLE_L, PURPLE, 10, "bold")
    _box(ax, 78, 70, 18, 14, "Reply\nshown", GREEN_L, GREEN, 10, "bold")

    _box(ax, 36, 46, 30, 14,
         "TF-IDF  +  MLP classifier\n-> intent + confidence", GREY_L, GREY, 9.5, "bold")

    _box(ax, 6, 18, 28, 16,
         "HIGH confidence (>= 0.75)\nEmpathetic response from\nreal counseling data\n(+ tone & repetition guards)",
         GREEN_L, GREEN, 9)
    _box(ax, 64, 18, 30, 16,
         "LOW confidence (< 0.75)\nFAISS search over\nauthentic WHO guidelines\n(relevance gate 0.35)",
         ORANGE_L, ORANGE, 9)

    _arrow(ax, 24, 77, 40, 77)
    _arrow(ax, 62, 77, 78, 77)
    _arrow(ax, 51, 70, 51, 60)
    _arrow(ax, 44, 46, 24, 34, GREEN, "confident")
    _arrow(ax, 58, 46, 76, 34, ORANGE, "unsure")
    _arrow(ax, 20, 34, 47, 70, GREEN, ls="--")
    _arrow(ax, 80, 34, 56, 70, ORANGE, ls="--")

    ax.text(50, 6, "Two response paths, chosen automatically by the model's confidence",
            ha="center", fontsize=9, style="italic", color=GREY)
    return _save(fig, "fig_overview.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 2. ML training pipeline
# ═══════════════════════════════════════════════════════════════════════════════
def ml_pipeline():
    fig, ax = _new((10, 4.6))
    ax.text(50, 95, "The Machine-Learning Pipeline (offline, run once)",
            ha="center", fontsize=13, weight="bold", color=INK)

    steps = [
        ("Real dataset\ndair-ai/emotion\n(~16k labelled)", BLUE_L, BLUE),
        ("EDA + Cleaning\nnulls, duplicates,\nshort rows", PURPLE_L, PURPLE),
        ("NLP preprocess\ntokenise, stopwords,\nlemmatise", PURPLE_L, PURPLE),
        ("TF-IDF\ntext -> numbers\n(1-2 grams)", GREY_L, GREY),
        ("MLP classifier\n80/20 split,\nclass weights", ORANGE_L, ORANGE),
        ("Evaluate + Save\n~93% test acc\nmodel.pkl", GREEN_L, GREEN),
    ]
    n = len(steps)
    w, h, gap = 13.5, 22, 2.5
    x0 = (100 - (n * w + (n - 1) * gap)) / 2
    y = 50
    centers = []
    for i, (txt, fc, ec) in enumerate(steps):
        x = x0 + i * (w + gap)
        _box(ax, x, y, w, h, txt, fc, ec, 8.3, "bold" if i in (4, 5) else "normal")
        centers.append((x, x + w))
        if i > 0:
            _arrow(ax, centers[i - 1][1], y + h / 2, x, y + h / 2)

    ax.text(50, 30,
            "Output artifacts:  model/intent_model.pkl   +   model/vectorizer.pkl",
            ha="center", fontsize=9.5, color=INK, weight="bold")
    ax.text(50, 22,
            "Metrics produced:  training & testing accuracy, per-class precision/recall/F1,\n"
            "confusion matrix, and a confidence histogram (with the 0.75 threshold line).",
            ha="center", fontsize=8.7, color=GREY)
    return _save(fig, "fig_ml_pipeline.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 3. RAG knowledge-base pipeline
# ═══════════════════════════════════════════════════════════════════════════════
def rag_pipeline():
    fig, ax = _new((10, 4.4))
    ax.text(50, 95, "The WHO Knowledge Base (RAG-Lite, offline build)",
            ha="center", fontsize=13, weight="bold", color=INK)

    steps = [
        ("4 WHO PDFs\nvia DSpace\nREST API", BLUE_L, BLUE),
        ("Extract text\n(pdfplumber)", PURPLE_L, PURPLE),
        ("Chunk\n~256-word\noverlaps", PURPLE_L, PURPLE),
        ("Drop reference\n/ citation\nchunks", GREY_L, GREY),
        ("Embed\nMiniLM-L6\n(384-dim)", ORANGE_L, ORANGE),
        ("FAISS index\n554 vectors\n(cosine)", GREEN_L, GREEN),
    ]
    n = len(steps)
    w, h, gap = 13.5, 22, 2.5
    x0 = (100 - (n * w + (n - 1) * gap)) / 2
    y = 48
    centers = []
    for i, (txt, fc, ec) in enumerate(steps):
        x = x0 + i * (w + gap)
        _box(ax, x, y, w, h, txt, fc, ec, 8.3, "bold" if i in (0, 5) else "normal")
        centers.append((x, x + w))
        if i > 0:
            _arrow(ax, centers[i - 1][1], y + h / 2, x, y + h / 2)

    ax.text(50, 28,
            "Output artifacts:  data/faiss_who.index   +   data/faiss_who_chunks.pkl",
            ha="center", fontsize=9.5, color=INK, weight="bold")
    ax.text(50, 20,
            "Sources: WHO Mental Health Action Plan, mhGAP Intervention Guide,\n"
            "Preventing Suicide, and Mental Health Atlas 2020 — all public-domain WHO.",
            ha="center", fontsize=8.7, color=GREY)
    return _save(fig, "fig_rag_pipeline.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Confidence routing decision flow
# ═══════════════════════════════════════════════════════════════════════════════
def routing():
    fig, ax = _new((10, 6.2))
    ax.text(50, 97, "Runtime Decision Flow (per user message)",
            ha="center", fontsize=13, weight="bold", color=INK)

    _box(ax, 36, 84, 28, 10, "User message", BLUE_L, BLUE, 10, "bold")
    _box(ax, 30, 64, 40, 11,
         "Safety & small-talk checks\n(crisis words, greeting, gibberish, very short)",
         GREY_L, GREY, 8.8)
    _box(ax, 33, 47, 34, 10, "TF-IDF + MLP\n-> intent, confidence", PURPLE_L, PURPLE, 9.5, "bold")

    # decision diamond (as a box)
    _box(ax, 36, 31, 28, 10, "confidence >= 0.75 ?", ORANGE_L, ORANGE, 9.5, "bold")

    _box(ax, 4, 10, 30, 14,
         "Empathetic path\nTone correction + retrieve from\ncounseling data + trim + guards",
         GREEN_L, GREEN, 8.7)
    _box(ax, 66, 10, 30, 14,
         "WHO retrieval path\nFAISS nearest chunk;\nif relevance >= 0.35 show excerpt,\nelse empathetic fallback",
         ORANGE_L, ORANGE, 8.4)

    _arrow(ax, 50, 84, 50, 75)
    _arrow(ax, 50, 64, 50, 57)
    _arrow(ax, 50, 47, 50, 41)
    _arrow(ax, 40, 31, 19, 24, GREEN, "YES")
    _arrow(ax, 60, 31, 81, 24, ORANGE, "NO")

    ax.text(50, 3,
            "Crisis messages are caught first and always get a supportive, escalation response.",
            ha="center", fontsize=8.7, style="italic", color=GREY)
    return _save(fig, "fig_routing.png")


def build_all():
    return {
        "overview": overview(),
        "ml": ml_pipeline(),
        "rag": rag_pipeline(),
        "routing": routing(),
    }


if __name__ == "__main__":
    for k, v in build_all().items():
        print(f"  {k:10s} -> {v}")
