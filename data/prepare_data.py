"""
Data preparation pipeline for MindSpace.

Produces two files:
  data/mental_health_cleaned.csv  — Amod counseling conversations (used by RAG)
  data/intent_train.csv           — dair-ai/emotion dataset, 16k rows (used by MLP)

Run once before train.py:
    python data/prepare_data.py
"""

import re
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from datasets import load_dataset

# Download required NLTK data (silent if already present)
for pkg in ("punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"):
    nltk.download(pkg, quiet=True)

_lemmatizer  = WordNetLemmatizer()
_stop_words  = set(stopwords.words("english"))


def nlp_preprocess(text: str) -> str:
    """Lowercase → punctuation removal → tokenize → stopword removal → lemmatize."""
    text   = str(text).lower()
    text   = re.sub(r"[^a-zA-Z\s]", "", text)          # punctuation removal
    tokens = word_tokenize(text)                         # NLTK tokenization
    tokens = [t for t in tokens if t not in _stop_words] # stopword removal
    tokens = [_lemmatizer.lemmatize(t) for t in tokens]  # lemmatization
    return " ".join(tokens)

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

# ═══════════════════════════════════════════════════════════════════════════════
# PART A — Amod counseling conversations  →  mental_health_cleaned.csv  (RAG)
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 60)
print("PART A  —  Amod mental health counseling conversations")
print("=" * 60)

print("\nDownloading Amod/mental_health_counseling_conversations...")
amod = load_dataset("Amod/mental_health_counseling_conversations", split="train").to_pandas()

# ── EDA ───────────────────────────────────────────────────────────────────────
print("\n── EDA ──────────────────────────────────────────────────────")
print(f"Shape         : {amod.shape}")
print(f"Columns       : {amod.columns.tolist()}")
print(f"\nDtypes:\n{amod.dtypes}")
print(f"\nNull values:\n{amod.isnull().sum()}")
print(f"\nDuplicate rows: {amod.duplicated().sum()}")
print(f"\nSample rows:\n{amod.head(3).to_string()}")
for col in amod.columns:
    if amod[col].dtype == object:
        lengths = amod[col].dropna().str.len()
        print(f"\n'{col}' length — min:{lengths.min()}  max:{lengths.max()}  mean:{lengths.mean():.1f}")

# ── Clean ─────────────────────────────────────────────────────────────────────
print("\n── CLEANING ─────────────────────────────────────────────────")
amod = amod.rename(columns={"Context": "user_input", "Response": "bot_response"})
before = len(amod)
amod = amod.dropna(subset=["user_input", "bot_response"])
print(f"After dropna          : {len(amod):,} rows  (removed {before - len(amod):,})")
amod = amod.drop_duplicates(subset=["user_input"])
print(f"After drop_duplicates : {len(amod):,} rows  (removed {before - len(amod):,} total)")
amod["user_input"]   = amod["user_input"].str.strip()
amod["bot_response"] = amod["bot_response"].str.strip()
amod = amod[amod["user_input"].str.len()   >= 15]
amod = amod[amod["bot_response"].str.len() >= 20]
print(f"After length filter   : {len(amod):,} rows")
amod = amod.reset_index(drop=True)

# ── Intent labelling (keyword rules on real text) ─────────────────────────────
INTENT_RULES = [
    ("depression_symptoms", [
        r"\bdepress\w*\b", r"\bhopeless\b", r"\bworthless\b", r"\bempty inside\b",
        r"\bno point\b", r"\bgiving up\b", r"\bsuicid\w*\b", r"\bcan'?t go on\b",
        r"\bmeaningless\b", r"\bnumb\b", r"\bno reason to live\b", r"\bwant to die\b",
        r"\blost interest\b", r"\bno motivation\b", r"\bcan'?t get out of bed\b",
        r"\bfeeling low\b", r"\bfeeling down\b", r"\bfeel miserable\b",
        r"\bfeel like a burden\b", r"\bisolat\w*\b", r"\bwithdrawn\b",
        r"\bno energy\b", r"\bcrying\b", r"\bfeel empty\b", r"\bfeel dead inside\b",
    ]),
    ("anxiety_symptoms", [
        r"\banxious\b", r"\banxiety\b", r"\bpanic\b", r"\bworr\w+\b", r"\bnervous\b",
        r"\bscared\b", r"\boverthink\w*\b", r"\bheart racing\b", r"\bcan'?t breathe\b",
        r"\bphobia\b", r"\bdread\b", r"\buneasy\b", r"\btense\b",
        r"\bcan'?t stop thinking\b", r"\bwhat if\b", r"\bon edge\b",
        r"\brestless\b", r"\btrembl\w+\b", r"\bshak\w+\b", r"\bfreaking out\b",
        r"\bsocial anxiety\b", r"\bfear of\b", r"\bafraid of\b",
    ]),
    ("stress_reaction", [
        r"\bstress\w*\b", r"\bburnout\b", r"\bexhaust\w+\b", r"\btoo much\b",
        r"\bcan'?t cope\b", r"\bpressure\b", r"\boverload\w*\b", r"\bfrustrat\w+\b",
        r"\bangry\b", r"\brage\b", r"\birritat\w+\b", r"\boverwhelm\w*\b",
        r"\bcan'?t handle\b", r"\bfalling apart\b", r"\bbreaking down\b",
        r"\blosing it\b", r"\bcan'?t deal\b", r"\bdeadline\b",
        r"\btoo busy\b", r"\bexam\b", r"\bassignment\b", r"\bdrained\b",
    ]),
    ("mental_health_faq", [
        r"\bwhat is\b", r"\bhow do\b", r"\bwhy do\b", r"\btherapy\b",
        r"\bcounsel\w*\b", r"\bmedication\b", r"\bdiagnos\w*\b", r"\bdisorder\b",
        r"\btreatment\b", r"\bpsychiat\w*\b", r"\bpsycholog\w*\b", r"\bmental health\b",
        r"\bcbt\b", r"\bdbt\b", r"\bshould i see\b", r"\bdo i need\b",
        r"\bis it normal\b", r"\bsigns of\b", r"\bget help\b",
        r"\bprofessional help\b", r"\banti.?depress\w*\b", r"\bcoping strateg\w*\b",
    ]),
]

def assign_intent(text: str) -> str:
    t = text.lower()
    for intent, patterns in INTENT_RULES:
        for pat in patterns:
            if re.search(pat, t):
                return intent
    return "emotional_support"

amod["intent"] = amod["user_input"].apply(assign_intent)

# Apply full NLP preprocessing pipeline for the model training column
print("\nApplying NLP preprocessing (tokenize → stopword removal → lemmatize)...")
amod["user_input_processed"] = amod["user_input"].apply(nlp_preprocess)

print("\n── INTENT DISTRIBUTION (RAG dataset) ───────────────────────")
dist_amod = amod["intent"].value_counts()
print(dist_amod.to_string())

# ── EDA: Intent distribution bar chart ───────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
sns.barplot(x=dist_amod.values, y=dist_amod.index, hue=dist_amod.index, palette="Blues_r", legend=False, ax=ax)
ax.set_title("Intent Distribution — Counseling Dataset (Amod)", fontsize=13, fontweight="bold")
ax.set_xlabel("Number of Samples")
ax.set_ylabel("Intent")
for i, v in enumerate(dist_amod.values):
    ax.text(v + 2, i, str(v), va="center", fontsize=10)
plt.tight_layout()
plt.show()

rag_path = os.path.join(DATA_DIR, "mental_health_cleaned.csv")
amod[["user_input", "user_input_processed", "bot_response", "intent"]].to_csv(rag_path, index=False)
print(f"\nSaved RAG data → {rag_path}  ({len(amod):,} rows)")


# ═══════════════════════════════════════════════════════════════════════════════
# PART B — dair-ai/emotion  →  intent_train.csv  (MLP classifier training)
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("PART B  —  dair-ai/emotion  (classifier training data)")
print("=" * 60)

# Emotion label → intent mapping (same logic BERT was using)
EMOTION_TO_INTENT = {
    0: "depression_symptoms",  # sadness
    1: "emotional_support",    # joy
    2: "emotional_support",    # love
    3: "stress_reaction",      # anger
    4: "anxiety_symptoms",     # fear
    5: "mental_health_faq",    # surprise
}

print("\nDownloading dair-ai/emotion (train + validation + test)...")
splits = load_dataset("dair-ai/emotion")

frames = []
for split_name in ["train", "validation", "test"]:
    part = splits[split_name].to_pandas()
    frames.append(part)

emotion_df = pd.concat(frames, ignore_index=True)

# ── EDA ───────────────────────────────────────────────────────────────────────
print("\n── EDA ──────────────────────────────────────────────────────")
print(f"Shape         : {emotion_df.shape}")
print(f"Columns       : {emotion_df.columns.tolist()}")
print(f"\nDtypes:\n{emotion_df.dtypes}")
print(f"\nNull values:\n{emotion_df.isnull().sum()}")
print(f"\nDuplicate rows: {emotion_df.duplicated().sum()}")
print(f"\nLabel distribution (raw):\n{emotion_df['label'].value_counts().to_string()}")

# ── Clean ─────────────────────────────────────────────────────────────────────
print("\n── CLEANING ─────────────────────────────────────────────────")
before = len(emotion_df)
emotion_df = emotion_df.dropna(subset=["text", "label"])
emotion_df = emotion_df.drop_duplicates(subset=["text"])
emotion_df["text"] = emotion_df["text"].str.strip()
emotion_df = emotion_df[emotion_df["text"].str.len() >= 10]
print(f"After cleaning: {len(emotion_df):,} rows  (removed {before - len(emotion_df):,})")

# ── Map labels to intents ─────────────────────────────────────────────────────
emotion_df["intent"] = emotion_df["label"].map(EMOTION_TO_INTENT)
emotion_df = emotion_df.rename(columns={"text": "user_input"})

print("\nApplying NLP preprocessing to emotion dataset...")
emotion_df["user_input_processed"] = emotion_df["user_input"].apply(nlp_preprocess)

print("\n── INTENT DISTRIBUTION (classifier training data) ───────────")
dist_emotion = emotion_df["intent"].value_counts()
print(dist_emotion.to_string())

# ── EDA: Intent distribution bar chart for emotion dataset ────────────────────
fig2, ax2 = plt.subplots(figsize=(8, 5))
sns.barplot(x=dist_emotion.values, y=dist_emotion.index, hue=dist_emotion.index, palette="Purples_r", legend=False, ax=ax2)
ax2.set_title("Intent Distribution — Emotion Dataset (dair-ai)", fontsize=13, fontweight="bold")
ax2.set_xlabel("Number of Samples")
ax2.set_ylabel("Intent")
for i, v in enumerate(dist_emotion.values):
    ax2.text(v + 10, i, str(v), va="center", fontsize=10)
plt.tight_layout()
plt.show()

train_path = os.path.join(DATA_DIR, "intent_train.csv")
emotion_df[["user_input", "user_input_processed", "intent"]].to_csv(train_path, index=False)
print(f"\nSaved classifier training data → {train_path}  ({len(emotion_df):,} rows)")

print("\nDone. Run model/train.py next.")
