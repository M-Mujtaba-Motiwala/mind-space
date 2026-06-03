"""
Build the MindSpace DS project-notes PDF.

Generates the explanatory diagrams (via diagrams.py) and assembles a
multi-page, plain-English PDF that explains the whole project end to end:
what it is, how it works, the ML + RAG pipelines, the file map, results,
a glossary, and a viva (oral-exam) question bank.

Run:  python docs/build_docs.py
Out:  docs/MindSpace_DS_Project_Notes.pdf
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import diagrams  # noqa: E402

from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.units import cm  # noqa: E402
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle  # noqa: E402
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY  # noqa: E402
from reportlab.platypus import (  # noqa: E402
    SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak,
    Table, TableStyle, ListFlowable, ListItem, HRFlowable,
)

# ── palette (matches diagrams.py) ──────────────────────────────────────────────
INK = colors.HexColor("#2d3436")
PURPLE = colors.HexColor("#6c5ce7")
BLUE = colors.HexColor("#2f80ed")
GREEN = colors.HexColor("#27ae60")
ORANGE = colors.HexColor("#e67e22")
GREY = colors.HexColor("#566573")
GREY_L = colors.HexColor("#eceff1")
PURPLE_L = colors.HexColor("#ece9fb")

OUT = os.path.join(HERE, "MindSpace_DS_Project_Notes.pdf")


# ── styles ─────────────────────────────────────────────────────────────────────
def _styles():
    ss = getSampleStyleSheet()
    styles = {}
    styles["title"] = ParagraphStyle(
        "title", parent=ss["Title"], fontName="Helvetica-Bold",
        fontSize=26, leading=30, textColor=PURPLE, spaceAfter=6)
    styles["subtitle"] = ParagraphStyle(
        "subtitle", parent=ss["Normal"], fontName="Helvetica",
        fontSize=13, leading=18, textColor=GREY, alignment=TA_CENTER, spaceAfter=4)
    styles["h1"] = ParagraphStyle(
        "h1", parent=ss["Heading1"], fontName="Helvetica-Bold",
        fontSize=17, leading=21, textColor=PURPLE, spaceBefore=14, spaceAfter=7)
    styles["h2"] = ParagraphStyle(
        "h2", parent=ss["Heading2"], fontName="Helvetica-Bold",
        fontSize=13, leading=17, textColor=BLUE, spaceBefore=10, spaceAfter=4)
    styles["body"] = ParagraphStyle(
        "body", parent=ss["Normal"], fontName="Helvetica",
        fontSize=10.5, leading=15.5, textColor=INK, alignment=TA_JUSTIFY,
        spaceAfter=6)
    styles["bullet"] = ParagraphStyle(
        "bullet", parent=ss["Normal"], fontName="Helvetica",
        fontSize=10.5, leading=15, textColor=INK, spaceAfter=2)
    styles["caption"] = ParagraphStyle(
        "caption", parent=ss["Normal"], fontName="Helvetica-Oblique",
        fontSize=9, leading=12, textColor=GREY, alignment=TA_CENTER, spaceAfter=10)
    styles["q"] = ParagraphStyle(
        "q", parent=ss["Normal"], fontName="Helvetica-Bold",
        fontSize=10.5, leading=14.5, textColor=INK, spaceBefore=7, spaceAfter=1)
    styles["a"] = ParagraphStyle(
        "a", parent=ss["Normal"], fontName="Helvetica",
        fontSize=10.5, leading=14.5, textColor=GREY, alignment=TA_JUSTIFY,
        spaceAfter=3)
    styles["code"] = ParagraphStyle(
        "code", parent=ss["Code"], fontName="Courier",
        fontSize=9, leading=12, textColor=INK, backColor=GREY_L,
        borderPadding=4, spaceAfter=6)
    return styles


def bullets(items, S, style="bullet"):
    return ListFlowable(
        [ListItem(Paragraph(t, S[style]), leftIndent=8, value="•") for t in items],
        bulletType="bullet", bulletColor=PURPLE, leftIndent=12, spaceAfter=6)


def rule():
    return HRFlowable(width="100%", thickness=0.8, color=GREY_L,
                      spaceBefore=4, spaceAfter=8)


def fig(path, S, caption, width=16.5 * cm):
    from reportlab.lib.utils import ImageReader
    iw, ih = ImageReader(path).getSize()
    h = width * ih / iw
    return [Image(path, width=width, height=h),
            Paragraph(caption, S["caption"])]


# ── page furniture ─────────────────────────────────────────────────────────────
def _footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(GREY)
    canvas.drawString(2 * cm, 1.1 * cm, "MindSpace DS — Project Notes")
    canvas.drawRightString(A4[0] - 2 * cm, 1.1 * cm, f"Page {doc.page}")
    canvas.setStrokeColor(GREY_L)
    canvas.line(2 * cm, 1.4 * cm, A4[0] - 2 * cm, 1.4 * cm)
    canvas.restoreState()


# ═══════════════════════════════════════════════════════════════════════════════
def build():
    imgs = diagrams.build_all()
    S = _styles()
    story = []

    # ── TITLE PAGE ──────────────────────────────────────────────────────────────
    story += [
        Spacer(1, 3.5 * cm),
        Paragraph("MindSpace", S["title"]),
        Paragraph("A Mental-Health Support Chatbot", S["subtitle"]),
        Paragraph("Data-Science Project Notes", S["subtitle"]),
        Spacer(1, 0.5 * cm),
        HRFlowable(width="55%", thickness=1.2, color=PURPLE,
                   spaceBefore=4, spaceAfter=14, hAlign="CENTER"),
        Paragraph(
            "An intent classifier (TF-IDF + neural network) that routes each "
            "message to either an empathetic reply learned from real counseling "
            "conversations, or an evidence-based answer retrieved from official "
            "WHO mental-health guidelines.", S["subtitle"]),
        Spacer(1, 2.5 * cm),
        Paragraph("Plain-English explanation of what the system does, how every "
                  "part works, and how to defend it in a viva.", S["caption"]),
        PageBreak(),
    ]

    # ── 1. WHAT IS MINDSPACE ────────────────────────────────────────────────────
    story += [
        Paragraph("1. What is MindSpace (in one minute)", S["h1"]),
        Paragraph(
            "MindSpace is a web chatbot that listens to how a person is feeling "
            "and replies supportively. You type (or speak) a message; the app "
            "figures out the <b>intent</b> behind it — for example anxiety, low "
            "mood, stress, or a general question — and how <b>confident</b> it is "
            "about that guess. Based on that confidence it picks one of two ways "
            "to answer.", S["body"]),
        Paragraph(
            "It is built as a <b>data-science project</b>, not a black box. Every "
            "stage uses real, public data and standard, explainable techniques: "
            "we clean a labelled emotion dataset, turn text into numbers with "
            "TF-IDF, train a small neural network to recognise intent, measure "
            "how good it is with proper train/test metrics, and connect it to a "
            "searchable library of official World Health Organization (WHO) "
            "guidance.", S["body"]),
        Paragraph("The two answer paths", S["h2"]),
        bullets([
            "<b>Confident</b> (the model is sure what the user means): reply with "
            "an <b>empathetic message</b> drawn from real counseling conversations, "
            "after tone-correction and anti-repetition checks.",
            "<b>Unsure</b> (the model is not sure): <b>search the WHO knowledge "
            "base</b> for the most relevant passage and show that — but only if it "
            "clears a relevance bar; otherwise fall back to gentle support.",
        ], S),
        Spacer(1, 0.2 * cm),
    ]
    story += fig(imgs["overview"], S,
                 "Figure 1 — From a typed message to a reply. The classifier's "
                 "confidence decides which of the two paths is used.")
    story.append(PageBreak())

    # ── 2. PROJECT PROPOSAL ALIGNMENT ───────────────────────────────────────────
    story += [
        Paragraph("2. How this matches the project proposal", S["h1"]),
        Paragraph(
            "The proposal asked for a complete data-science pipeline — real data, "
            "exploratory analysis and cleaning, NLP preprocessing, a trained model "
            "with a proper train/test evaluation, and visual results — applied to a "
            "useful problem. MindSpace delivers exactly that, and adds a retrieval "
            "layer so answers can be grounded in authoritative sources.", S["body"]),
    ]
    t = Table([
        [Paragraph("<b>Proposal requirement</b>", S["bullet"]),
         Paragraph("<b>How MindSpace meets it</b>", S["bullet"])],
        [Paragraph("Real dataset", S["bullet"]),
         Paragraph("dair-ai/emotion (~16k labelled messages) for training; "
                   "Amod mental-health counseling conversations for replies; "
                   "4 official WHO PDFs for the knowledge base.", S["bullet"])],
        [Paragraph("EDA &amp; cleaning", S["bullet"]),
         Paragraph("Null/duplicate removal, very-short-row filtering, class "
                   "balance inspection, text-length checks.", S["bullet"])],
        [Paragraph("NLP preprocessing", S["bullet"]),
         Paragraph("Tokenisation, stop-word removal, lemmatisation (NLTK).",
                   S["bullet"])],
        [Paragraph("Feature engineering", S["bullet"]),
         Paragraph("TF-IDF with 1–2 word grams, 5,000 features.", S["bullet"])],
        [Paragraph("Model + train/test", S["bullet"]),
         Paragraph("MLP neural-net classifier, stratified 80/20 split, class "
                   "weights for imbalance.", S["bullet"])],
        [Paragraph("Evaluation &amp; visuals", S["bullet"]),
         Paragraph("Accuracy, precision/recall/F1 per class, confusion matrix, "
                   "confidence histogram with the 0.75 threshold.", S["bullet"])],
        [Paragraph("Real-world value", S["bullet"]),
         Paragraph("Confidence-routing to authentic WHO guidance keeps answers "
                   "safe and grounded.", S["bullet"])],
    ], colWidths=[5.2 * cm, 11 * cm])
    t.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.5, GREY_L),
        ("BACKGROUND", (0, 0), (-1, 0), PURPLE_L),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(t)
    story.append(PageBreak())

    # ── 3. ML PIPELINE ──────────────────────────────────────────────────────────
    story += [
        Paragraph("3. The machine-learning pipeline (built offline, once)", S["h1"]),
        Paragraph(
            "This is the part that <i>learns</i>. It runs once on our machine, "
            "produces two saved files, and then the app just loads them — no "
            "training happens while a user chats.", S["body"]),
    ]
    story += fig(imgs["ml"], S,
                 "Figure 2 — Six stages from raw labelled text to a saved, "
                 "evaluated model.")
    story += [
        Paragraph("Stage by stage", S["h2"]),
        bullets([
            "<b>Real dataset</b> — dair-ai/emotion: ~16,000 short messages, each "
            "labelled with an emotion (joy, sadness, anger, fear, love, surprise). "
            "We map these onto mental-health intents.",
            "<b>EDA + cleaning</b> — we look at the data first: how many of each "
            "class, how long messages are, and remove blanks, duplicates and "
            "ultra-short rows that carry no signal.",
            "<b>NLP preprocessing</b> — lower-case, split into tokens, drop common "
            "stop-words ('the', 'is'), and lemmatise ('running' → 'run') so similar "
            "words count as one.",
            "<b>TF-IDF</b> — converts text to numbers. Each message becomes a vector "
            "where words that are frequent <i>here</i> but rare <i>overall</i> get a "
            "high weight. We use 1- and 2-word grams and keep the top 5,000 terms.",
            "<b>MLP classifier</b> — a small neural network (hidden layers 128→64) "
            "trained on an 80/20 stratified split, with class weights so rare "
            "emotions aren't ignored.",
            "<b>Evaluate + save</b> — about <b>93% test accuracy</b>; we save "
            "<font face='Courier'>model/intent_model.pkl</font> and "
            "<font face='Courier'>model/vectorizer.pkl</font>.",
        ], S),
        Paragraph(
            "<b>Why TF-IDF + MLP instead of a big model like BERT?</b> Because the "
            "proposal is a data-science exercise: TF-IDF + MLP is fast, runs on a "
            "laptop with no GPU, and — crucially — is <b>explainable</b>. We can "
            "point to exact words driving a prediction and show every metric. A huge "
            "pre-trained transformer would hide the data-science we're being graded "
            "on.", S["body"]),
        PageBreak(),
    ]

    # ── 4. CONFIDENCE ROUTING ───────────────────────────────────────────────────
    story += [
        Paragraph("4. Runtime decision flow (what happens per message)", S["h1"]),
        Paragraph(
            "When a real message arrives, it passes through a series of checks "
            "before any machine-learning guess is trusted. Safety comes first.",
            S["body"]),
    ]
    story += fig(imgs["routing"], S,
                 "Figure 3 — Each message is screened for safety and small-talk, "
                 "classified, then routed by confidence.")
    story += [
        bullets([
            "<b>Safety &amp; small-talk checks</b> — crisis words trigger a "
            "supportive escalation response immediately. Greetings, gibberish and "
            "very short messages get sensible canned replies instead of being "
            "force-fed to the model.",
            "<b>Classify</b> — TF-IDF + MLP produce an intent and a confidence "
            "(the highest class probability).",
            "<b>Confidence ≥ 0.75 → empathetic path</b> — fetch a response from the "
            "counseling dataset, correct its tone to match the user, trim it to a "
            "couple of sentences, and avoid repeating recent replies.",
            "<b>Confidence &lt; 0.75 → WHO path</b> — search the guideline index; if "
            "the best match scores ≥ 0.35 relevance, show a tidied excerpt; "
            "otherwise fall back to empathetic support.",
        ], S),
        Paragraph(
            "<b>Why a threshold at all?</b> A classifier is most useful when it "
            "knows when to be humble. If it isn't sure, guessing an emotion and "
            "replying confidently is worse than looking up an authoritative answer. "
            "0.75 was chosen from the confidence histogram — it's where correct "
            "predictions cluster above and shaky ones below.", S["body"]),
        PageBreak(),
    ]

    # ── 5. RAG / WHO KNOWLEDGE BASE ─────────────────────────────────────────────
    story += [
        Paragraph("5. The WHO knowledge base (RAG-Lite)", S["h1"]),
        Paragraph(
            "RAG means <b>Retrieval-Augmented Generation</b> — instead of inventing "
            "an answer, the system <i>retrieves</i> a real passage from trusted "
            "documents. Ours is 'RAG-Lite': we retrieve and present authentic WHO "
            "text rather than paraphrasing it, so nothing is fabricated.", S["body"]),
    ]
    story += fig(imgs["rag"], S,
                 "Figure 4 — Four official WHO PDFs become a searchable index of "
                 "554 passages.")
    story += [
        Paragraph("How the library is built", S["h2"]),
        bullets([
            "<b>4 WHO PDFs</b> — downloaded from WHO's official IRIS repository "
            "through its DSpace REST API (handles, not fragile direct links).",
            "<b>Extract text</b> — pdfplumber pulls the words out of each PDF.",
            "<b>Chunk</b> — text is split into ~256-word overlapping passages so a "
            "search can return a focused snippet.",
            "<b>Drop references</b> — citation/bibliography chunks are detected and "
            "removed (151 dropped) so searches don't surface reference lists.",
            "<b>Embed</b> — each chunk is turned into a 384-number vector with the "
            "all-MiniLM-L6-v2 sentence-transformer, so meaning (not just words) can "
            "be compared.",
            "<b>FAISS index</b> — 554 vectors stored in a FAISS index for instant "
            "cosine-similarity search.",
        ], S),
        Paragraph(
            "<b>Sources:</b> WHO Mental Health Action Plan 2013–2030, mhGAP "
            "Intervention Guide v2, Preventing Suicide: A Global Imperative, and "
            "Mental Health Atlas 2020 — all public WHO documents.", S["body"]),
        Paragraph(
            "<b>Relevance gate (0.35):</b> a search always returns its nearest "
            "passage, but 'nearest' can still be irrelevant. If the best score is "
            "below 0.35 we discard it and give empathetic support instead — this is "
            "what stopped the early bug where unrelated text (e.g. a passage about "
            "snakes) was shown for an off-topic message.", S["body"]),
        PageBreak(),
    ]

    # ── 6. CONVERSATION QUALITY GUARDRAILS ──────────────────────────────────────
    story += [
        Paragraph("6. Conversation-quality guardrails", S["h1"]),
        Paragraph(
            "A correct intent isn't enough — the reply has to <i>feel</i> right. "
            "Several rule-based guards sit on top of the model to keep the "
            "conversation natural and safe.", S["body"]),
        Paragraph("Tone correction (symmetric)", S["h2"]),
        Paragraph(
            "The emotion model can be fooled by surface words, so we correct it "
            "using context. The logic is symmetric and ordered by priority:",
            S["body"]),
        bullets([
            "<b>Negation / negative words win first</b> — 'I did <i>not</i> get the "
            "job' → sadness, never joy.",
            "<b>Clear positive cues → joy</b> — 'I <b>won</b> this competition', "
            "'I <b>got the job</b>', 'I <b>passed</b>' → congratulate, don't "
            "commiserate.",
            "<b>Stress topics → fear/stress</b> — 'I have <b>exams</b>', "
            "'<b>deadline</b> tomorrow' → acknowledge the pressure.",
            "<b>Positive sentence but no strong cue → neutral</b> — avoids "
            "over-claiming an emotion.",
        ], S),
        Paragraph(
            "This directly fixes two reported bugs: 'I got final exams' was being "
            "read as joy, and 'I won this competition' would have been called "
            "stressful. Now tone is judged from the whole situation, both ways.",
            S["body"]),
        Paragraph("Anti-repetition &amp; meta-complaints", S["h2"]),
        bullets([
            "Every reply remembers the <b>last 6</b> bot messages and won't repeat "
            "them.",
            "If the user complains ('you keep repeating', 'going in circles') we "
            "detect it, apologise, and deliberately change direction.",
            "After several upbeat turns the bot de-escalates instead of robotically "
            "asking the same probing question.",
        ], S),
        Paragraph("Length &amp; relevance trimming", S["h2"]),
        bullets([
            "Long forum-style answers are trimmed to ~2 sentences / 55 words.",
            "Filler openers ('Thank you for reaching out!') and self-references "
            "('As an American…') are stripped — this fixed the 'where did America "
            "come from' bug.",
            "Messages under 6 words get a short, intent-appropriate reply that asks "
            "for a little more detail, instead of a wall of text.",
        ], S),
        PageBreak(),
    ]

    # ── 7. FILE MAP ─────────────────────────────────────────────────────────────
    story += [
        Paragraph("7. File-by-file map", S["h1"]),
        Paragraph("Where each piece of the project lives:", S["body"]),
    ]
    files = [
        ("app.py", "The Streamlit web app: UI, session state, safety checks, "
                   "classification, confidence routing, guardrails, and the "
                   "two response paths."),
        ("model/train_intent.py", "Offline training: loads + cleans the emotion "
                                  "dataset, builds TF-IDF, trains the MLP, "
                                  "evaluates, and saves the .pkl files."),
        ("model/humanize.py", "Tone-correction logic (symmetric emotion fixing, "
                              "stress + positive indicators)."),
        ("model/faiss_rag.py", "Loads the FAISS index and retrieves the most "
                               "relevant WHO passage with the 0.35 relevance gate."),
        ("model/intent_model.pkl", "Saved trained classifier (committed so the app "
                                   "runs after cloning)."),
        ("model/vectorizer.pkl", "Saved TF-IDF vectoriser."),
        ("data/build_who_corpus.py", "Downloads the 4 WHO PDFs via the DSpace REST "
                                     "API, extracts + chunks text, drops references, "
                                     "embeds, and builds the FAISS index."),
        ("data/faiss_who.index", "The 554-vector FAISS similarity index."),
        ("data/faiss_who_chunks.pkl", "The text of each indexed passage."),
        ("data/who_corpus/*.pdf", "The 4 source WHO PDFs."),
        ("docs/diagrams.py", "Generates the four explanatory diagrams."),
        ("docs/build_docs.py", "Builds this PDF."),
    ]
    rows = [[Paragraph("<b>Path</b>", S["bullet"]),
             Paragraph("<b>What it does</b>", S["bullet"])]]
    for p, d in files:
        rows.append([Paragraph(f"<font face='Courier' size='8.5'>{p}</font>",
                               S["bullet"]),
                     Paragraph(d, S["bullet"])])
    ft = Table(rows, colWidths=[5 * cm, 11.2 * cm])
    ft.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.5, GREY_L),
        ("BACKGROUND", (0, 0), (-1, 0), PURPLE_L),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(ft)
    story.append(PageBreak())

    # ── 8. RESULTS ──────────────────────────────────────────────────────────────
    story += [
        Paragraph("8. Key results", S["h1"]),
        bullets([
            "<b>~93% test accuracy</b> on the held-out 20% — the model generalises, "
            "not just memorises.",
            "<b>Per-class precision / recall / F1</b> reported for every emotion, so "
            "weak classes are visible.",
            "<b>Confusion matrix</b> shows which emotions are mixed up (e.g. fear vs "
            "sadness) — expected and explainable.",
            "<b>Confidence histogram</b> with the 0.75 line justifies the routing "
            "threshold visually.",
            "<b>554 clean WHO passages</b> indexed (from 705 raw, 151 reference "
            "chunks removed).",
        ], S),
        Paragraph("9. Glossary (say these clearly in the viva)", S["h1"]),
    ]
    gloss = [
        ("Intent", "The underlying meaning/need behind a message (e.g. anxiety, "
                   "low mood, a factual question)."),
        ("TF-IDF", "Term Frequency–Inverse Document Frequency: turns text into "
                   "numbers, weighting words that are frequent in one message but "
                   "rare across all messages."),
        ("MLP", "Multi-Layer Perceptron: a small feed-forward neural network used "
                "here as the classifier."),
        ("Embedding", "A list of numbers (a vector) that captures the meaning of a "
                      "piece of text, so similar meanings sit close together."),
        ("FAISS", "Facebook AI Similarity Search: a library for finding the nearest "
                  "vectors very fast."),
        ("Cosine similarity", "Measures how close two vectors point in direction — "
                              "used to rank passages by relevance."),
        ("RAG", "Retrieval-Augmented Generation: ground answers in retrieved real "
                "documents instead of inventing them."),
        ("Confidence threshold", "The 0.75 cutoff that decides whether to trust the "
                                 "classifier or look something up."),
        ("Relevance gate", "The 0.35 cutoff a retrieved WHO passage must clear to be "
                           "shown."),
    ]
    grows = [[Paragraph("<b>Term</b>", S["bullet"]),
              Paragraph("<b>Plain meaning</b>", S["bullet"])]]
    for term, meaning in gloss:
        grows.append([Paragraph(f"<b>{term}</b>", S["bullet"]),
                      Paragraph(meaning, S["bullet"])])
    gt = Table(grows, colWidths=[3.8 * cm, 12.4 * cm])
    gt.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.5, GREY_L),
        ("BACKGROUND", (0, 0), (-1, 0), PURPLE_L),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(gt)
    story.append(PageBreak())

    # ── 10. VIVA Q&A ────────────────────────────────────────────────────────────
    story += [Paragraph("10. Viva question bank (with answers)", S["h1"])]
    qa = [
        ("Why TF-IDF + MLP instead of BERT?",
         "It's a data-science project graded on the pipeline, not raw accuracy. "
         "TF-IDF + MLP is fast, laptop-friendly (no GPU), and explainable — we can "
         "show the exact words and every metric. BERT would hide that work."),
        ("How do you handle class imbalance?",
         "A stratified 80/20 split keeps class proportions in both sets, and we pass "
         "class weights (via sample_weight) so rare emotions still influence "
         "training."),
        ("What does the confidence score mean?",
         "It's the highest class probability from the MLP's predict_proba. High means "
         "the model strongly favours one intent; low means the classes are close and "
         "it's unsure."),
        ("Why route on confidence?",
         "A humble classifier is more trustworthy. When unsure, retrieving an "
         "authoritative WHO passage beats confidently guessing an emotion."),
        ("How did you pick 0.75 and 0.35?",
         "0.75 from the confidence histogram — correct predictions cluster above it. "
         "0.35 is the empirical relevance floor below which retrieved passages were "
         "off-topic."),
        ("Isn't retrieval just keyword search?",
         "No — we embed text into 384-dim vectors with a sentence-transformer and "
         "compare by cosine similarity, so it matches by <i>meaning</i>, not exact "
         "words."),
        ("How do you avoid fabricated medical advice?",
         "The WHO path shows authentic excerpts verbatim (RAG-Lite), guarded by the "
         "0.35 relevance gate; crisis messages are caught first and always get a safe "
         "escalation reply."),
        ("How do you stop repetitive or off-tone replies?",
         "Last-6 memory prevents repeats, a meta-complaint detector changes direction "
         "when the user notices, symmetric tone-correction reads the whole situation, "
         "and trimming keeps replies short and on-topic."),
        ("What are the limitations?",
         "It's not a medical tool; the emotion dataset is general (not clinical); "
         "TF-IDF ignores word order beyond bigrams; and the knowledge base is only "
         "four documents."),
        ("Could you improve it?",
         "Add more WHO/clinical sources, try calibrated probabilities, expand to "
         "multi-turn context, and A/B test the thresholds with user feedback."),
    ]
    for q, a in qa:
        story.append(Paragraph(f"Q. {q}", S["q"]))
        story.append(Paragraph(f"A. {a}", S["a"]))
    story += [
        Spacer(1, 0.3 * cm), rule(),
        Paragraph("End of notes — MindSpace DS. Built with real data, standard "
                  "techniques, and explainable choices.", S["caption"]),
    ]

    doc = SimpleDocTemplate(
        OUT, pagesize=A4,
        leftMargin=2 * cm, rightMargin=2 * cm,
        topMargin=2 * cm, bottomMargin=2 * cm,
        title="MindSpace DS — Project Notes")
    doc.build(story, onFirstPage=_footer, onLaterPages=_footer)
    return OUT


if __name__ == "__main__":
    path = build()
    print(f"PDF written -> {path}")
