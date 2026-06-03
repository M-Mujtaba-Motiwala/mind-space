"""
Train the TF-IDF + MLP intent classifier on the cleaned HuggingFace dataset.

Run after data/prepare_data.py:
    python model/train.py

Outputs:
  model/intent_model.pkl        — trained MLP classifier
  model/vectorizer.pkl          — fitted TF-IDF vectorizer
  model/confusion_matrix.png    — test-set confusion matrix heatmap
  model/confidence_histogram.png — distribution of prediction confidence scores
"""

import os
import re
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
from sklearn.utils.class_weight import compute_class_weight

MODEL_DIR = os.path.dirname(__file__)

# ── LOAD ──────────────────────────────────────────────────────────────────────
DATA_PATH = os.path.join(MODEL_DIR, "..", "data", "intent_train.csv")
df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip().str.lower()

print(f"Loaded {len(df):,} rows from intent_train.csv (dair-ai/emotion)")
print(f"\nIntent distribution (full dataset):")
dist = df["intent"].value_counts()
print(dist.to_string())
print(f"\nClass balance — majority/minority ratio: "
      f"{dist.max() / dist.min():.1f}x\n")

# ── PREPROCESS ────────────────────────────────────────────────────────────────
# Use the NLTK-preprocessed column if available, else fall back to raw text
TEXT_COL = "user_input_processed" if "user_input_processed" in df.columns else "user_input"

def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text.strip()

df["cleaned"] = df[TEXT_COL].apply(clean_text)
X = df["cleaned"]
y = df["intent"]

# ── 80 / 20 SPLIT ─────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train size : {len(X_train):,}")
print(f"Test  size : {len(X_test):,}\n")

# ── VECTORISE ─────────────────────────────────────────────────────────────────
vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec  = vectorizer.transform(X_test)

# ── CLASS WEIGHTS ─────────────────────────────────────────────────────────────
classes = np.unique(y_train)
weights = compute_class_weight("balanced", classes=classes, y=y_train)
class_weight_map  = dict(zip(classes, weights))
sample_weights    = np.array([class_weight_map[label] for label in y_train])

# ── TRAIN ─────────────────────────────────────────────────────────────────────
model = MLPClassifier(
    hidden_layer_sizes=(128, 64),
    max_iter=500,
    random_state=42,
    alpha=1.0,
    learning_rate_init=0.001,
)
model.fit(X_train_vec, y_train, sample_weight=sample_weights)

# ── EVALUATE — accuracy ───────────────────────────────────────────────────────
y_train_pred = model.predict(X_train_vec)
y_test_pred  = model.predict(X_test_vec)

train_acc = accuracy_score(y_train, y_train_pred)
test_acc  = accuracy_score(y_test,  y_test_pred)

print("=" * 50)
print(f"  Training Accuracy : {train_acc:.4f}  ({train_acc*100:.2f}%)")
print(f"  Testing  Accuracy : {test_acc:.4f}  ({test_acc*100:.2f}%)")
print(f"  Difference        : {abs(train_acc - test_acc):.4f}  "
      f"({'overfit' if train_acc - test_acc > 0.05 else 'generalising well'})")
print("=" * 50)

print("\nTraining Classification Report:")
print(classification_report(y_train, y_train_pred))

print("Testing Classification Report:")
print(classification_report(y_test, y_test_pred))

# ── CONFUSION MATRIX ─────────────────────────────────────────────────────────
print("\nGenerating confusion matrix...")
cm = confusion_matrix(y_test, y_test_pred, labels=classes)
fig1, ax1 = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=classes, yticklabels=classes, ax=ax1,
            linewidths=0.5, linecolor="white")
ax1.set_title("Confusion Matrix — Test Set", fontsize=13, fontweight="bold")
ax1.set_ylabel("Actual Label")
ax1.set_xlabel("Predicted Label")
plt.xticks(rotation=30, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# ── CONFIDENCE HISTOGRAM ──────────────────────────────────────────────────────
print("Generating confidence histogram...")
test_proba      = model.predict_proba(X_test_vec)
max_confidences = test_proba.max(axis=1)

fig2, ax2 = plt.subplots(figsize=(8, 5))
ax2.hist(max_confidences, bins=20, color="#8b7cf8", edgecolor="white", alpha=0.85)
ax2.axvline(x=0.75, color="red", linestyle="--", linewidth=1.8,
            label="Confidence threshold (0.75)")
ax2.set_title("Distribution of Prediction Confidence Scores — Test Set",
              fontsize=13, fontweight="bold")
ax2.set_xlabel("Max Predicted Probability")
ax2.set_ylabel("Number of Samples")
ax2.legend()
plt.tight_layout()
plt.show()

below_threshold = (max_confidences < 0.75).sum()
print(f"\n  Samples below 0.75 threshold : {below_threshold} / {len(max_confidences)} "
      f"({below_threshold/len(max_confidences)*100:.1f}%) → routed to FAISS retrieval")

# ── SAVE MODEL ────────────────────────────────────────────────────────────────
pickle.dump(model,      open(os.path.join(MODEL_DIR, "intent_model.pkl"), "wb"))
pickle.dump(vectorizer, open(os.path.join(MODEL_DIR, "vectorizer.pkl"),   "wb"))
print("\nSaved intent_model.pkl and vectorizer.pkl")
