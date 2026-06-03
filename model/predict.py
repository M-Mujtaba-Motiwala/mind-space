import os
import re
import pickle
import streamlit as st

# Maps MLP intent labels back to a raw emotion string expected by humanize.py
INTENT_TO_EMOTION = {
    "depression_symptoms": "sadness",
    "anxiety_symptoms":    "fear",
    "stress_reaction":     "anger",
    "emotional_support":   "joy",
    "mental_health_faq":   "surprise",
}

_MODEL_DIR = os.path.dirname(__file__)


@st.cache_resource(show_spinner=False)
def load_model():
    model      = pickle.load(open(os.path.join(_MODEL_DIR, "intent_model.pkl"), "rb"))
    vectorizer = pickle.load(open(os.path.join(_MODEL_DIR, "vectorizer.pkl"),   "rb"))
    return model, vectorizer


def _clean(text: str) -> str:
    text = str(text).lower()
    return re.sub(r"[^a-zA-Z\s]", "", text).strip()


def predict_intent(text: str):
    """Return (intent, emotion, confidence) for the given user text using the local MLP model.
    confidence is the max predicted probability (0–1); values below 0.75 are routed
    to the FAISS WHO/UNICEF retrieval path in app.py.
    """
    model, vectorizer = load_model()
    vec        = vectorizer.transform([_clean(text)])
    intent     = model.predict(vec)[0]
    proba      = model.predict_proba(vec)[0]
    confidence = float(max(proba))
    emotion    = INTENT_TO_EMOTION.get(intent, "surprise")
    return intent, emotion, confidence
