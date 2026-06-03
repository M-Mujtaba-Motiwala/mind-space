import os
import re
import random
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import speech_recognition as sr

from model.predict import predict_intent, load_model
from model.humanize import humanize_response, correct_emotion
from model.rag import semantic_retrieve, load_encoder, build_index
from model.faiss_rag import retrieve_from_corpus
from model import db

CONFIDENCE_THRESHOLD = 0.75   # below this → route to WHO/UNICEF FAISS retrieval


# ── BRAND ASSETS ──────────────────────────────────────────────────────────────
LOGO_SVG = """
<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 64 64'>
  <defs>
    <linearGradient id='lg' x1='0' y1='0' x2='64' y2='64' gradientUnits='userSpaceOnUse'>
      <stop offset='0' stop-color='#8b7cf8'/>
      <stop offset='1' stop-color='#c4b5fd'/>
    </linearGradient>
  </defs>
  <rect width='64' height='64' rx='14' fill='url(#lg)'/>
  <path d='M16 46 V20 L24 28 L32 20 L40 28 L48 20 V46'
        fill='none' stroke='white' stroke-width='3.5'
        stroke-linecap='round' stroke-linejoin='round'/>
</svg>
""".strip()

FAVICON_DATA = (
    "data:image/svg+xml;utf8,"
    "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 64 64'>"
    "<defs><linearGradient id='g' x1='0' y1='0' x2='64' y2='64' gradientUnits='userSpaceOnUse'>"
    "<stop offset='0' stop-color='%238b7cf8'/><stop offset='1' stop-color='%23c4b5fd'/>"
    "</linearGradient></defs>"
    "<rect width='64' height='64' rx='14' fill='url(%23g)'/>"
    "<path d='M16 46 V20 L24 28 L32 20 L40 28 L48 20 V46' fill='none' stroke='white' "
    "stroke-width='3.5' stroke-linecap='round' stroke-linejoin='round'/></svg>"
)

# Chat avatars — SVG data URIs (Streamlit accepts URL strings, including data URIs)
BOT_AVATAR = (
    "data:image/svg+xml;utf8,"
    "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'>"
    "<defs><linearGradient id='ba' x1='0' y1='0' x2='32' y2='32' gradientUnits='userSpaceOnUse'>"
    "<stop offset='0' stop-color='%238b7cf8'/><stop offset='1' stop-color='%23a78bfa'/>"
    "</linearGradient></defs>"
    "<circle cx='16' cy='16' r='16' fill='url(%23ba)'/>"
    "<text x='16' y='21' font-family='Inter,Arial,sans-serif' font-size='13' "
    "font-weight='700' fill='white' text-anchor='middle'>M</text></svg>"
)
USER_AVATAR = (
    "data:image/svg+xml;utf8,"
    "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'>"
    "<circle cx='16' cy='16' r='15' fill='%231f2235' stroke='%23292d45'/>"
    "<text x='16' y='21' font-family='Inter,Arial,sans-serif' font-size='13' "
    "font-weight='600' fill='%23737899' text-anchor='middle'>U</text></svg>"
)

# ── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MindSpace",
    layout="wide",
    page_icon=FAVICON_DATA,
    initial_sidebar_state="expanded",
)

# ── INITIALIZE DB ────────────────────────────────────────────────────────────
db.init_db()

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@24,400,0,0');

:root {
    --bg:           #181b27;
    --sidebar:      #13151f;
    --card:         #1f2235;
    --card-border:  #292d45;
    --accent:       #8b7cf8;
    --accent-dim:   #5a50c4;
    --accent-light: #a78bfa;
    --text:         #dde1f0;
    --text-muted:   #737899;
    --text-dim:     #383d5e;
    --user-bg:      #1b3254;
    --user-border:  #254876;
    --bot-bg:       #1d2038;
    --bot-border:   #282d50;
    --danger-soft:  #fca5a5;
    --radius:       16px;
    --radius-sm:    10px;
    --transition:   0.25s ease;
}

*, *::before, *::after { box-sizing: border-box; }
html, body { font-family: 'Inter', sans-serif !important; background-color: var(--bg) !important; }
body { color: var(--text); }
p, span, label, div, h1, h2, h3, h4, li, a {
    font-family: 'Inter', sans-serif !important;
}

/* ── Streamlit chrome — header collapsed but kept for sidebar toggle ── */
header[data-testid="stHeader"],
[data-testid="stHeader"] {
    background: transparent !important;
    box-shadow: none !important;
    height: 0 !important;
    min-height: 0 !important;
    overflow: visible !important;
}
footer, #MainMenu { display: none !important; visibility: hidden !important; }
[data-testid="stDecoration"],
[data-testid="stToolbar"],
[data-testid="stToolbarActions"],
[data-testid="stAppDeployButton"],
[data-testid="stStatusWidget"],
button[kind="header"] { display: none !important; }
[data-testid="stMain"] { background-color: var(--bg) !important; }

/* ── Sidebar — always visible at fixed width ── */
section[data-testid="stSidebar"] {
    width: 270px !important;
    min-width: 270px !important;
    max-width: 270px !important;
    background-color: var(--sidebar) !important;
    border-right: 1px solid var(--card-border) !important;
    visibility: visible !important;
    transform: translateX(0) !important;
}
section[data-testid="stSidebar"] > div:first-child {
    padding: 18px 14px 16px 14px !important;
}

/* Sidebar toggle buttons — fully hidden (sidebar always shown) */
[data-testid="stSidebarCollapseButton"],
[data-testid="stSidebarCollapsedControl"],
button[data-testid="collapsedControl"] { display: none !important; }

/* ── Brand block ── */
.brand-block {
    display: flex; align-items: center; gap: 10px;
    padding: 4px 8px 16px 8px;
    border-bottom: 1px solid var(--card-border);
    margin-bottom: 14px;
}
.brand-logo {
    width: 36px; height: 36px;
    border-radius: 10px;
    flex-shrink: 0;
    background: linear-gradient(135deg, var(--accent), var(--accent-light));
    display: flex; align-items: center; justify-content: center;
    overflow: hidden;
}
.brand-logo svg { width: 100%; height: 100%; display: block; }
.brand-name {
    font-size: 16px; font-weight: 700;
    color: var(--text); letter-spacing: -0.3px;
    line-height: 1.1;
}
.brand-tagline {
    font-size: 11px; color: var(--text-muted); margin-top: 2px;
}

/* User chip in sidebar */
.user-chip {
    display: flex; align-items: center; gap: 10px;
    padding: 10px 12px;
    background: var(--card);
    border: 1px solid var(--card-border);
    border-radius: var(--radius-sm);
    margin-bottom: 14px;
}
.user-chip-avatar {
    width: 30px; height: 30px;
    border-radius: 50%;
    background: linear-gradient(135deg, var(--accent), var(--accent-light));
    color: white; font-weight: 700; font-size: 12px;
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0;
}
.user-chip-name {
    font-size: 13px; font-weight: 600; color: var(--text);
    overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
}

/* Nav section labels */
.nav-label {
    font-size: 10px; font-weight: 600;
    color: var(--text-dim);
    text-transform: uppercase; letter-spacing: 0.8px;
    padding: 0 8px; margin: 14px 0 4px 0;
}

/* Sidebar buttons */
[data-testid="stSidebar"] .stButton > button {
    width: 100%;
    background: transparent !important;
    border: none !important;
    border-radius: var(--radius-sm) !important;
    color: var(--text-muted) !important;
    font-size: 13px !important; font-weight: 400 !important;
    text-align: left !important;
    padding: 8px 12px !important;
    margin-bottom: 1px;
    transition: all var(--transition);
    cursor: pointer;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: var(--card) !important;
    color: var(--text) !important;
}
[data-testid="stSidebar"] .stButton > button:focus { box-shadow: none !important; outline: none !important; }

.sdiv { border: none; border-top: 1px solid var(--card-border); margin: 12px 0; }

/* Conversations list */
.conv-empty {
    font-size: 12px; color: var(--text-dim);
    padding: 6px 12px; font-style: italic;
}

/* ── Main content ── */
.main .block-container,
[data-testid="stMain"] .block-container {
    padding: 24px 32px 130px 32px !important;
    max-width: 880px !important;
    margin: 0 auto !important;
}

/* ── Chat header — centered big brand ── */
.chat-header {
    display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    padding: 20px 0 22px 0;
    border-bottom: 1px solid var(--card-border);
    margin-bottom: 26px;
    text-align: center;
}
.chat-header-logo {
    width: 56px; height: 56px;
    border-radius: 14px;
    overflow: hidden;
    margin-bottom: 10px;
    flex-shrink: 0;
}
.chat-header-logo svg { width: 100%; height: 100%; display: block; }
.chat-header-name {
    font-size: 30px; font-weight: 700;
    color: var(--text); margin: 0;
    letter-spacing: -0.6px;
    line-height: 1.1;
}
.chat-header-sub {
    font-size: 12px; color: var(--text-muted);
    margin-top: 4px; font-weight: 400;
}
/* Compact header for wellness pages — keep them small/secondary */
.page-header {
    display: flex; align-items: center;
    padding: 14px 0 16px 0;
    border-bottom: 1px solid var(--card-border);
    margin-bottom: 22px;
    gap: 12px;
}
.page-header-icon {
    width: 36px; height: 36px;
    background: linear-gradient(135deg, var(--accent), var(--accent-light));
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 14px; font-weight: 700; color: white;
    flex-shrink: 0;
}
.page-header-title { font-size: 16px; font-weight: 600; color: var(--text); }
.page-header-sub { font-size: 11px; color: var(--text-muted); margin-top: 2px; }

/* ── Welcome screen ── */
.welcome-card {
    background: var(--card);
    border: 1px solid var(--card-border);
    border-radius: var(--radius);
    padding: 36px 32px;
    text-align: center;
    margin: 20px 0 24px 0;
}
.welcome-title {
    font-size: 24px; font-weight: 700;
    color: var(--text); margin: 0 0 10px 0;
    letter-spacing: -0.4px;
}
.welcome-sub {
    font-size: 14px; color: var(--text-muted);
    line-height: 1.6; max-width: 420px; margin: 0 auto;
}

/* ── Crisis banner ── */
.crisis-banner {
    background: rgba(252,165,165,0.08);
    border: 1px solid rgba(252,165,165,0.25);
    border-radius: var(--radius);
    padding: 16px 20px;
    margin-bottom: 18px;
    animation: fadeIn 0.5s ease;
}
.crisis-banner-title {
    font-size: 14px; font-weight: 600;
    color: var(--danger-soft); margin: 0 0 6px 0;
}
.crisis-banner-text {
    font-size: 13px; color: #f0a0a0; margin: 0 0 12px 0;
    line-height: 1.6;
}

/* ── Chat messages ── */
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: translateY(0); }
}
[data-testid="stChatMessage"] {
    background: transparent !important;
    border: none !important;
    padding: 6px 0 !important;
    margin-bottom: 6px !important;
    animation: fadeIn 0.4s ease;
    gap: 10px !important;
}

/* Avatar containers — render actual SVG images */
[data-testid="stChatMessage"] [data-testid^="chatAvatarIcon"] {
    width: 30px !important; height: 30px !important;
    border-radius: 50% !important;
    flex-shrink: 0 !important;
    overflow: hidden !important;
    background: transparent !important;
    padding: 0 !important;
}
[data-testid="stChatMessage"] [data-testid^="chatAvatarIcon"] img {
    width: 100% !important;
    height: 100% !important;
    object-fit: cover !important;
    display: block !important;
}

/* Message bubbles */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) > div:nth-child(2),
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) [data-testid="stChatMessageContent"] {
    background: var(--bot-bg) !important;
    border: 1px solid var(--bot-border) !important;
    border-radius: 4px var(--radius) var(--radius) var(--radius) !important;
    padding: 14px 18px !important;
    max-width: 78% !important;
}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    flex-direction: row-reverse !important;
    justify-content: flex-start !important;
}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) > div:nth-child(2),
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) [data-testid="stChatMessageContent"] {
    background: var(--user-bg) !important;
    border: 1px solid var(--user-border) !important;
    border-radius: var(--radius) 4px var(--radius) var(--radius) !important;
    padding: 10px 16px !important;
    max-width: 65% !important;
}
[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p {
    font-size: 14px !important; line-height: 1.7 !important;
    color: var(--text) !important; margin: 0 !important;
}

/* Typing indicator */
.typing-card {
    background: var(--bot-bg);
    border: 1px solid var(--bot-border);
    border-radius: 4px var(--radius) var(--radius) var(--radius);
    padding: 14px 20px;
    display: inline-flex; align-items: center; gap: 10px;
    animation: fadeIn 0.3s ease;
}
.typing-dots { display: flex; gap: 4px; }
.typing-dots span {
    width: 6px; height: 6px;
    background: var(--accent); border-radius: 50%;
    animation: dotBounce 1.2s infinite ease-in-out;
}
.typing-dots span:nth-child(2) { animation-delay: 0.2s; }
.typing-dots span:nth-child(3) { animation-delay: 0.4s; }
@keyframes dotBounce {
    0%, 80%, 100% { transform: scale(0.7); opacity: 0.5; }
    40%           { transform: scale(1.1); opacity: 1; }
}
.typing-text { font-size: 13px; color: var(--text-muted); }

/* ── Input area ── */
[data-testid="stBottom"] {
    background: linear-gradient(to top, var(--bg) 75%, transparent) !important;
    padding: 8px 0 16px !important;
}
[data-testid="stChatInput"] {
    background: var(--card) !important;
    border: 1px solid var(--card-border) !important;
    border-radius: 14px !important;
    box-shadow: 0 4px 24px rgba(0,0,0,0.3) !important;
    max-width: 740px !important; margin: 0 auto !important;
    position: relative !important;
}
[data-testid="stChatInput"]:focus-within {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px rgba(139,124,248,0.2), 0 4px 24px rgba(0,0,0,0.3) !important;
}
[data-testid="stChatInput"] textarea {
    color: var(--text) !important;
    font-size: 14px !important; line-height: 1.6 !important;
    padding-right: 96px !important;
}
[data-testid="stChatInput"] textarea::placeholder { color: var(--text-dim) !important; }
[data-testid="stChatInputSubmitButton"] button {
    background: var(--accent) !important; border-radius: 8px !important;
}
[data-testid="stChatInputSubmitButton"] button:hover { background: var(--accent-dim) !important; }

/* ── Mic button — float alongside chat input ── */
.st-key-mic_fab,
div[class*="st-key-mic_fab"] {
    position: fixed !important;
    bottom: 14px !important;
    right: 80px !important;
    z-index: 1001 !important;
    width: 44px !important;
    height: 44px !important;
    margin: 0 !important;
    padding: 0 !important;
    background: transparent !important;
    border: none !important;
}
.st-key-mic_fab > div,
.st-key-mic_fab .stButton {
    margin: 0 !important; padding: 0 !important;
    width: 44px !important; height: 44px !important;
}
.st-key-mic_fab button {
    width: 44px !important; height: 44px !important;
    border-radius: 50% !important;
    padding: 0 !important;
    font-size: 0 !important;
    background: var(--card) !important;
    border: 1px solid var(--card-border) !important;
    color: transparent !important;
    line-height: 1 !important;
    display: flex !important;
    align-items: center !important; justify-content: center !important;
    transition: all var(--transition) !important;
    cursor: pointer !important;
    position: relative !important;
}
.st-key-mic_fab button::before {
    content: 'mic' !important;
    font-family: 'Material Symbols Rounded' !important;
    font-size: 22px !important;
    color: var(--text-muted) !important;
    font-weight: 400 !important;
    line-height: 1 !important;
    -webkit-font-feature-settings: 'liga';
}
.st-key-mic_fab button:hover {
    background: var(--accent) !important;
    border-color: var(--accent) !important;
    transform: scale(1.05) !important;
}
.st-key-mic_fab button:hover::before { color: white !important; }

/* ── Wellness pages ── */
.page-card {
    background: var(--card);
    border: 1px solid var(--card-border);
    border-radius: var(--radius);
    padding: 26px 26px;
    margin-bottom: 14px;
}
.page-card h2 { font-size: 18px; font-weight: 600; color: var(--text); margin: 0 0 8px 0; }
.page-card p { font-size: 14px; color: var(--text-muted); line-height: 1.7; margin: 0; }

@keyframes breathe {
    0%, 100% { transform: scale(1);   opacity: 0.7; }
    50%      { transform: scale(1.4); opacity: 1; }
}
.breath-circle {
    width: 120px; height: 120px;
    background: radial-gradient(circle, var(--accent), var(--accent-dim));
    border-radius: 50%;
    margin: 32px auto;
    animation: breathe 6s ease-in-out infinite;
    display: flex; align-items: center; justify-content: center;
    font-size: 13px; color: white; font-weight: 500;
}

.grounding-step {
    background: var(--bg);
    border: 1px solid var(--card-border);
    border-radius: var(--radius-sm);
    padding: 14px 16px;
    margin-bottom: 10px;
    display: flex; align-items: flex-start; gap: 12px;
}
.grounding-num {
    font-size: 20px; font-weight: 700;
    color: var(--accent); min-width: 28px;
}
.grounding-text { font-size: 13px; color: var(--text-muted); line-height: 1.6; }

/* ── Auth screens ── */
.auth-shell {
    min-height: 100vh; width: 100%;
    display: flex; align-items: center; justify-content: center;
    padding: 40px 20px;
}
.auth-card {
    width: 100%; max-width: 420px;
    background: var(--card);
    border: 1px solid var(--card-border);
    border-radius: var(--radius);
    padding: 36px 32px 28px 32px;
    text-align: center;
}
.auth-logo {
    width: 64px; height: 64px;
    margin: 0 auto 18px auto;
    border-radius: 16px;
    overflow: hidden;
    background: linear-gradient(135deg, var(--accent), var(--accent-light));
}
.auth-logo svg { width: 100%; height: 100%; display: block; }
.auth-title {
    font-size: 22px; font-weight: 700; color: var(--text);
    margin: 0 0 6px 0; letter-spacing: -0.3px;
}
.auth-sub {
    font-size: 13px; color: var(--text-muted);
    line-height: 1.55; margin: 0 0 24px 0;
}
[data-testid="stTabs"] [role="tablist"] {
    gap: 4px;
    border-bottom: 1px solid var(--card-border);
    margin-bottom: 18px;
}
[data-testid="stTabs"] [role="tab"] {
    color: var(--text-muted) !important;
    font-weight: 500 !important;
    background: transparent !important;
    padding: 10px 18px !important;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    color: var(--accent) !important;
    border-bottom: 2px solid var(--accent) !important;
}
.auth-card .stTextInput > label,
.auth-card label {
    color: var(--text-muted) !important;
    font-size: 12px !important; font-weight: 500 !important;
    text-align: left !important;
}
.auth-card input {
    background: var(--bg) !important;
    border: 1px solid var(--card-border) !important;
    color: var(--text) !important;
    border-radius: 10px !important;
    padding: 10px 14px !important;
    font-size: 14px !important;
}
.auth-card input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px rgba(139,124,248,0.2) !important;
}
.auth-card .stButton > button {
    width: 100% !important;
    background: var(--accent) !important;
    border: none !important;
    color: white !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    padding: 11px !important;
    border-radius: 10px !important;
    margin-top: 8px !important;
    transition: background var(--transition) !important;
}
.auth-card .stButton > button:hover { background: var(--accent-dim) !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--card-border); border-radius: 4px; }

/* ── Mobile ── */
@media (max-width: 768px) {
    .main .block-container,
    [data-testid="stMain"] .block-container {
        padding: 16px 14px 130px 14px !important;
    }
    section[data-testid="stSidebar"] {
        width: 220px !important;
        min-width: 220px !important;
        max-width: 220px !important;
    }
    .welcome-card { padding: 28px 20px !important; }
    .welcome-title { font-size: 20px !important; }
    .chat-header-name { font-size: 24px !important; }
    .chat-header-logo { width: 48px !important; height: 48px !important; }
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) > div:nth-child(2),
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) > div:nth-child(2) {
        max-width: 92% !important;
    }
    [data-testid="stChatInput"] textarea { padding-right: 90px !important; }
}
@media (max-width: 480px) {
    .welcome-title { font-size: 18px !important; }
    .chat-header-title { font-size: 14px !important; }
    .page-card { padding: 18px 16px !important; }
    .auth-title { font-size: 18px !important; }
}

/* ── Loading splash ── */
@keyframes spin { to { transform: rotate(360deg); } }
</style>
""", unsafe_allow_html=True)

# ── MODEL / DATA WARM-UP ──────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_dataset():
    d = pd.read_csv(os.path.join(os.path.dirname(__file__), "data", "mental_health_cleaned.csv"))
    d.columns = d.columns.str.strip().str.lower()
    return d

if "model_loaded" not in st.session_state:
    st.session_state.model_loaded = False

if not st.session_state.model_loaded:
    ph = st.empty()
    ph.markdown(f"""
    <div style="position:fixed;inset:0;background:#181b27;display:flex;
        flex-direction:column;align-items:center;justify-content:center;z-index:9999;">
        <div style="width:64px;height:64px;border-radius:16px;overflow:hidden;
            margin-bottom:20px;">{LOGO_SVG}</div>
        <div style="width:36px;height:36px;border:3px solid #292d45;
            border-top:3px solid #8b7cf8;border-radius:50%;
            animation:spin 0.9s linear infinite;margin-bottom:16px;"></div>
        <p style="color:#dde1f0;font-size:16px;font-weight:600;margin:0 0 4px;
            font-family:Inter,sans-serif;">MindSpace</p>
        <p style="color:#737899;font-size:13px;margin:0;font-family:Inter,sans-serif;">
            Preparing your safe space...</p>
    </div>""", unsafe_allow_html=True)
    load_model()
    load_encoder()
    df_warm = load_dataset()
    build_index(df_warm)
    st.session_state.model_loaded = True
    ph.empty()
    st.rerun()

df = load_dataset()

# ── AUTH SCREEN ───────────────────────────────────────────────────────────────
def render_auth():
    # Auth-screen-only CSS (applied globally; no other Streamlit elements here)
    st.markdown("""
    <style>
        section[data-testid="stSidebar"] {
            width: 0 !important; min-width: 0 !important;
            overflow: hidden !important;
            border: none !important;
        }
        [data-testid="stSidebarCollapseButton"],
        [data-testid="stSidebarCollapsedControl"],
        button[data-testid="collapsedControl"] { display: none !important; }
        [data-testid="stMain"] .block-container {
            max-width: 100% !important;
            padding: 60px 20px !important;
        }
        .stTextInput input {
            background: var(--bg) !important;
            border: 1px solid var(--card-border) !important;
            color: var(--text) !important;
            border-radius: 10px !important;
            padding: 10px 14px !important;
            font-size: 14px !important;
        }
        .stTextInput input:focus {
            border-color: var(--accent) !important;
            box-shadow: 0 0 0 2px rgba(139,124,248,0.2) !important;
            outline: none !important;
        }
        .stTextInput label {
            color: var(--text-muted) !important;
            font-size: 12px !important; font-weight: 500 !important;
        }
        .stButton > button {
            background: var(--accent) !important;
            border: none !important; color: white !important;
            font-weight: 600 !important; font-size: 14px !important;
            padding: 11px !important; border-radius: 10px !important;
            margin-top: 6px !important;
        }
        .stButton > button:hover { background: var(--accent-dim) !important; }
        [data-testid="stTabs"] [role="tablist"] {
            gap: 4px;
            border-bottom: 1px solid var(--card-border);
            margin-bottom: 14px;
            justify-content: center;
        }
        [data-testid="stTabs"] [role="tab"] {
            color: var(--text-muted) !important;
            font-weight: 500 !important;
            background: transparent !important;
            padding: 10px 24px !important;
        }
        [data-testid="stTabs"] [role="tab"][aria-selected="true"] {
            color: var(--accent) !important;
            border-bottom: 2px solid var(--accent) !important;
        }
        @media (max-width: 768px) {
            [data-testid="stMain"] .block-container { padding: 30px 14px !important; }
        }
    </style>
    """, unsafe_allow_html=True)

    _, mid, _ = st.columns([1, 2, 1])
    with mid:
        st.markdown(f"""
        <div style="background: var(--card); border: 1px solid var(--card-border);
            border-radius: var(--radius); padding: 32px 28px 16px 28px;
            text-align: center; margin-bottom: 0;">
            <div style="width:64px;height:64px;margin:0 auto 18px auto;
                border-radius:16px;overflow:hidden;">{LOGO_SVG}</div>
            <div style="font-size:22px;font-weight:700;color:var(--text);
                margin:0 0 6px 0;letter-spacing:-0.3px;">Welcome to MindSpace</div>
            <div style="font-size:13px;color:var(--text-muted);
                line-height:1.55;margin:0;">
                A safe, private space to talk about how you feel.<br>
                Sign in or create an account to continue.
            </div>
        </div>
        """, unsafe_allow_html=True)

        tab_login, tab_signup = st.tabs(["Sign In", "Create Account"])

        with tab_login:
            u = st.text_input("Username", key="login_user", placeholder="Your username")
            p = st.text_input("Password", type="password", key="login_pass",
                              placeholder="Your password")
            if st.button("Sign In", key="btn_login", use_container_width=True):
                uid = db.verify_user(u, p)
                if uid:
                    st.session_state.user_id = uid
                    st.session_state.username = u.strip()
                    st.session_state.active_conv = None
                    st.rerun()
                else:
                    st.error("Invalid username or password.")

        with tab_signup:
            nu = st.text_input("Choose a username", key="signup_user",
                               placeholder="Pick something memorable")
            np_ = st.text_input("Create a password", type="password",
                                key="signup_pass",
                                placeholder="At least 6 characters")
            nc = st.text_input("Confirm password", type="password",
                               key="signup_confirm",
                               placeholder="Re-enter password")
            if st.button("Create Account", key="btn_signup", use_container_width=True):
                if np_ != nc:
                    st.error("Passwords don't match.")
                else:
                    uid, err = db.create_user(nu, np_)
                    if err:
                        st.error(err)
                    else:
                        st.session_state.user_id = uid
                        st.session_state.username = nu.strip()
                        st.session_state.active_conv = None
                        st.rerun()


# ── ROUTING: not logged in → auth screen ──────────────────────────────────────
if "user_id" not in st.session_state:
    render_auth()
    st.stop()

# ── CONSTANTS ─────────────────────────────────────────────────────────────────
ANCHOR_KEYWORDS = {
    "mom":"your mom","dad":"your dad","mother":"your mom","father":"your dad",
    "parents":"your parents","family":"your family",
    "job":"your job","work":"work","boss":"your boss","career":"your career",
    "school":"school","university":"university","college":"college",
    "exam":"your exams","exams":"your exams","studies":"your studies",
    "homework":"your homework","assignment":"your assignments",
    "assignments":"your assignments","deadline":"your deadlines",
    "project":"your project","thesis":"your thesis",
    "friend":"your friend","friends":"your friends",
    "relationship":"your relationship","partner":"your partner",
    "boyfriend":"your boyfriend","girlfriend":"your girlfriend",
    "sleep":"your sleep","money":"money","debt":"your finances",
    "alone":"feeling alone","lonely":"loneliness",
}
SERIOUS_INTENTS = {"depression_symptoms","anxiety_symptoms","stress_reaction","emotional_support"}
ESCALATION_MESSAGES = [
    "You've been carrying a lot across our conversation. Have you had a chance to talk to someone you trust about any of this?",
    "I've noticed this has been weighing on you for a while. It might really help to speak with a professional — you deserve that kind of support.",
    "You've shared some heavy things today. I want to gently ask — is there someone in your life, or a counselor, you could reach out to?",
]
STRIP_PHRASES = [
    "Your feelings are valid.","You are not alone.",
    "It is okay to seek support from trusted people.",
    "Reaching out shows strength.",
    "Reaching out and talking about your emotions is an important step.",
]
CRISIS_WORDS = ["suicide","kill myself","end my life","hurt myself","don't want to live","want to die"]
FEELINGS_SHORTCUTS = ["I feel anxious","I feel sad","I can't sleep","I feel overwhelmed","I need to vent"]

POSITIVE_PHRASES = {
    "good","great","wonderful","amazing","fantastic","excellent",
    "doing well","feeling good","feeling great","really good","good actually",
    "doing great","pretty good","very good","super","happy","better",
    "much better","a lot better","feeling better","never better",
    "better today","better now","way better",
}
NEGATION_STARTS = {"not","nah","nope","no","never","don't","dont","can't","cant","haven't","havent"}
CASUAL_KEYWORDS = {
    "coffee","tea","food","lunch","dinner","breakfast","water",
    "walk","music","movie","show","game","book","nap","outside","chat","laugh","cook","bake",
}
HELP_SEEKING_PHRASES = [
    "want to talk","want to share","need to talk","need to share","need to vent",
    "want to vent","let me tell you","tell you about","talk about how",
    "talk about what","i'm feeling","im feeling","i feel","just wanted to talk",
    "just want to talk","need help","need advice","need someone","just chat",
    "wanted to talk","can we talk","help me","listen to me",
]

# ── SESSION DEFAULTS ──────────────────────────────────────────────────────────
def init_session():
    defaults = {
        "messages":[],"last_ack":None,"last_follow_up":None,
        "last_shortcut":None,"intent_history":[],"mentioned_topics":[],
        "pending_voice":None,"escalation_shown":False,
        "crisis_active":False,"page":"chat","active_conv":None,
        "last_confidence":None,"last_source":None,
        "positive_turns":0,"recent_bot":[],
    }
    for k,v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
init_session()

# ── HELPERS ───────────────────────────────────────────────────────────────────
def pick(options):
    # Avoid any response used in the last few turns so the bot doesn't repeat itself.
    recent = st.session_state.get("recent_bot", [])
    avail  = [o for o in options if o not in recent]
    choice = random.choice(avail or options)
    st.session_state.last_shortcut = choice
    st.session_state.recent_bot = (recent + [choice])[-6:]
    return choice

def extract_topics(text):
    found = []
    for word in text.lower().split():
        clean = re.sub(r"[^a-z]","",word)
        if clean in ANCHOR_KEYWORDS and clean not in st.session_state.mentioned_topics:
            found.append(clean)
    return found

def should_escalate():
    h = st.session_state.intent_history
    return (len(h)>=4 and not st.session_state.escalation_shown
            and sum(1 for i in h[-4:] if i in SERIOUS_INTENTS)>=3)

def topic_callback():
    topics = st.session_state.mentioned_topics
    if topics:
        label = ANCHOR_KEYWORDS.get(topics[-1])
        if label:
            return f"You mentioned {label} earlier — is that still on your mind?"
    return None

def clean_response(text):
    for p in STRIP_PHRASES:
        text = text.replace(p,"").strip()
    return " ".join(text.split())

# Filler openers and off-topic digressions common in the Amod forum answers.
_FILLER_OPENER = re.compile(
    r"^(thank you|thanks|that('s| is) a (great|good) question|great question|"
    r"good question|hi|hello|hey|welcome|i'?m glad you|i am glad you|"
    r"first of all|firstly|to answer your question)\b",
    re.IGNORECASE,
)
_SELF_REF = re.compile(
    r"\bas an? (american|aussie|brit)\b|\bin my country\b|"
    r"\bhere in (america|the us|the states)\b|\blive life to the fullest\b",
    re.IGNORECASE,
)

def trim_core(text, max_sentences=2, max_words=55):
    """
    The counseling dataset answers are long forum posts that open with filler
    ("That is a great question!") and wander off-topic ("As an American..."").
    Keep only the first couple of substantive, on-point sentences and cap length.
    """
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    keep = []
    for s in sentences:
        s = s.strip()
        if not s or len(s.split()) < 3:
            continue
        if _FILLER_OPENER.search(s) or _SELF_REF.search(s):
            continue
        keep.append(s)
        if len(keep) >= max_sentences:
            break
    out = " ".join(keep) if keep else text.strip()
    words = out.split()
    if len(words) > max_words:
        out = " ".join(words[:max_words]).rstrip(",;:") + "..."
    return out

# Concise, on-topic replies for short low-context messages ("i have exams").
# These probe for detail instead of dumping a long retrieved forum answer.
SHORT_SUPPORT = {
    "anxiety_symptoms": [
        "That sounds anxiety-provoking. What part of it worries you the most?",
        "I hear you — that can be a lot to sit with. What's making you most anxious about it?",
        "That's understandable. When you think about it, what's the worst-case fear that comes up?",
    ],
    "stress_reaction": [
        "That sounds stressful. What's feeling like the biggest pressure right now?",
        "I get that — it can pile up fast. What part is weighing on you most?",
        "That's a lot to carry. Is it the workload itself, or the worry about how it'll go?",
    ],
    "depression_symptoms": [
        "I'm sorry you're going through that. How long has it been feeling this way?",
        "That sounds heavy. What's been the hardest part lately?",
        "Thank you for telling me. Can you say a bit more about what's been going on?",
    ],
    "emotional_support": [
        "I'm here for you. Can you tell me a little more about what's going on?",
        "Thanks for sharing that. What's been on your mind the most?",
        "I'm listening. What would feel most helpful to talk through right now?",
    ],
    "mental_health_faq": [
        "Good question. Can you tell me a bit more about what you're experiencing?",
        "I can help with that. What specifically would you like to understand better?",
    ],
    "default": [
        "I hear you. Can you tell me a bit more about what's going on?",
        "Thanks for sharing. What's been weighing on you the most?",
        "I'm listening — what would you like to talk through?",
    ],
}

def normalize_text(text):
    return re.sub(r'(.)\1+',r'\1',text.lower().strip())

# Basic English words the bot should recognize — anything outside this is suspect
_COMMON_WORDS = {
    "i","me","my","we","you","he","she","it","they","the","a","an",
    "is","am","are","was","were","be","been","have","has","had",
    "do","does","did","will","would","could","should","can","may",
    "not","no","yes","and","or","but","if","so","to","of","in",
    "on","at","for","with","from","by","up","out","about","as",
    "feel","feeling","sad","happy","angry","scared","afraid","worried",
    "anxious","stressed","tired","lonely","alone","hurt","pain","low",
    "help","need","want","think","know","just","really","very","too",
    "bad","good","fine","okay","ok","hard","difficult","tough","rough",
    "work","school","family","friend","friends","life","day","time","thing",
    "love","hate","like","don't","dont","cant","can't","won't","wont",
    "im","i'm","ive","i've","its","it's","that","this","what","things",
    "how","why","when","where","who","much","many","some","all","lot",
    "nothing","everything","something","anything","more","less","most",
    "never","always","sometimes","maybe","still","also","even","only",
    "going","doing","being","having","getting","making","trying","talking",
    "because","than","then","here","there","now","today","lately",
    "night","sleep","eat","talk","tell","say","said","keep","kept",
    "people","someone","one","man","woman","mom","dad","mother","father",
    "depression","anxiety","panic","stress","overwhelmed","pressure",
    "cry","crying","tears","die","dead","death","kill",
    "sorry","please","thank","thanks","hello","hi","hey",
    "yeah","yep","nope","nah","sure","right","well","already",
    "oh","um","uh","hmm","hm","ah","idk","lol","omg",
    "been","ever","down","off","away","back","over","through",
    "better","worse","enough","stop","start","trying","again",
    "job","boss","career","money","debt","exam","exams","project",
    "relationship","partner","boyfriend","girlfriend","marriage",
    "brother","sister","parents","kids","child","children",
    "hopeless","worthless","useless","helpless","broken","empty",
    "numb","stuck","trapped","drained","frustrated","miserable",
    "exhausted","overwhelmed","burden","dark","lost","point",
    "sleep","sleeping","insomnia","rest","wake","waking",
    "eat","eating","appetite","weight","body","sick",
    "talk","talking","listen","hear","understand","care",
    "nobody","noone","anymore","nothing","myself","yourself",
    "everything","everyone","always","never","every","each",
    "home","house","room","bed","morning","evening","week",
    "month","year","years","long","since","ago","before",
    "after","during","while","until","when","whenever",
    "really","actually","literally","honestly","seriously",
    "kind","sort","type","way","bit","little","big","small",
    "new","old","same","different","other","another","own",
    "first","last","next","best","worst","end","done",
    "too","so","very","quite","pretty","rather","almost",
    "able","unable","enough","handle","manage","cope","deal",
    "problem","issue","situation","reason","cause","matter",
    "thought","idea","mind","head","heart","soul","world",
}

def _is_gibberish(text):
    """Detect nonsensical input like keyboard mashing."""
    cleaned = re.sub(r"[^a-zA-Z\s']", "", text).strip().lower()
    if not cleaned:
        return True
    # Vowel ratio — real English is ~38% vowels; gibberish has almost none
    vowels = sum(1 for c in cleaned if c in "aeiou")
    alpha_count = sum(1 for c in cleaned if c.isalpha())
    if alpha_count > 2 and vowels / alpha_count < 0.12:
        return True
    # Word recognition — at least 30% of words should be common English
    words = cleaned.split()
    if not words:
        return True
    recognized = sum(1 for w in words if w in _COMMON_WORDS)
    if len(words) >= 2 and recognized / len(words) < 0.25:
        return True
    # Single unrecognized word longer than 6 chars
    if len(words) == 1 and words[0] not in _COMMON_WORDS and len(words[0]) > 6:
        return True
    return False

def reset_chat_state():
    for k in ["messages","intent_history","mentioned_topics","recent_bot"]:
        st.session_state[k] = []
    for k in ["last_ack","last_follow_up","last_shortcut"]:
        st.session_state[k] = None
    st.session_state.positive_turns = 0
    st.session_state.escalation_shown = False
    st.session_state.crisis_active = False

def start_new_conversation():
    reset_chat_state()
    st.session_state.active_conv = None
    st.session_state.page = "chat"

def load_conversation(conv_id):
    reset_chat_state()
    st.session_state.active_conv = conv_id
    st.session_state.messages = db.get_messages(conv_id)
    st.session_state.page = "chat"

def logout():
    keys = list(st.session_state.keys())
    for k in keys:
        if k != "model_loaded":
            del st.session_state[k]

# ── EXPERT SYSTEM ─────────────────────────────────────────────────────────────
def expert_system_rule(text, row):
    t = text.lower()
    if any(w in t for w in CRISIS_WORDS):
        st.session_state.crisis_active = True
        return None
    # These columns only exist in the legacy labelled dataset. The current
    # HuggingFace-derived CSV doesn't carry them, so read defensively — crisis
    # handling is already covered by CRISIS_WORDS above and should_escalate().
    if str(row.get("risk_level", "")).strip().lower() == "high":
        return ("What you're sharing sounds really serious, and I want you to know I'm here. "
                "It might really help to talk to someone you trust or a professional — "
                "reaching out is one of the bravest things you can do.")
    if str(row.get("escalation_required", "")).strip().lower() in ["true", "yes"]:
        return ("This sounds like something that deserves real support beyond what I can offer. "
                "You're not alone in this — a mental health professional can truly help. "
                "Would you feel comfortable exploring that option?")
    return None

# ── RESPONSE LOGIC ────────────────────────────────────────────────────────────
def get_bot_response(user_text):
    text = user_text.lower().strip()
    normalized = normalize_text(user_text)
    words = text.split()
    word_set = set(words)

    if any(w in text for w in CRISIS_WORDS):
        st.session_state.crisis_active = True
        return ("I hear you, and I'm really glad you reached out. What you're feeling right now matters deeply. "
                "Please know you don't have to carry this alone — there are people ready to help you right now.")

    if normalized in ["hi","hello","hey","hi there","hello there","helo"]:
        return "Hello. I'm glad you're here. How are you feeling today?"

    # ── Gibberish detection ──────────────────────────────────────────────────
    if _is_gibberish(text):
        return pick([
            "I didn't quite catch that. Could you try saying it another way?",
            "I want to make sure I understand you. Could you rephrase that?",
            "I'm not sure I followed that. Take your time -- what's on your mind?",
            "I couldn't quite make that out. Try typing what you're feeling in your own words.",
        ])

    # ── Meta-complaint: user says the bot is repeating itself ────────────────
    _META_COMPLAINT = [
        "repetit","repeating","repeat","same question","same thing",
        "already asked","keep asking","asked that","said that already",
        "you keep saying","going in circles","broken record","stop asking",
    ]
    if any(p in text for p in _META_COMPLAINT):
        st.session_state.positive_turns = 0
        return pick([
            "You're right — I'm sorry for going in circles. Let's change tack: "
            "what's one thing that's actually been on your mind lately?",
            "Fair point, and I apologise for repeating myself. Tell me something "
            "concrete — what happened today that stuck with you?",
            "Good catch — I'll stop asking the same thing. If there's nothing "
            "pressing, that's completely fine. Otherwise, what would you like to dig into?",
            "You're right, I keep circling back. Let's reset — is there something "
            "specific you came here to talk about, or are we just checking in?",
        ])

    if text in {"yeah","yes","yep","yup","mhm","mmhm","uh huh","sure",
                "exactly","right","indeed","true","yeah it is","yes it is",
                "yeah still","yeah definitely","definitely"}:
        return pick([
            "I hear you. Do you want to talk more about what's been going on with that?",
            "That makes sense. What's been the hardest part of dealing with it?",
            "Got it. How long has it been weighing on you like this?",
            "I understand. Is there anything specific about it that's been most difficult?",
        ])

    starts_negative = words[0] in NEGATION_STARTS if words else False
    if not starts_negative:
        if text in POSITIVE_PHRASES or (len(words)<=8 and word_set & POSITIVE_PHRASES):
            return pick([
                "That's really good to hear. Is there anything on your mind you'd like to talk about?",
                "Glad to hear it — what's been making things feel better?",
                "That's great. Anything specific going well, or just a good day overall?",
                "Good to hear. What do you think has helped?",
            ])

    if word_set & CASUAL_KEYWORDS and len(words)<=8:
        return pick([
            "That sounds like a good call. Simple things like that really do help. How are you feeling otherwise?",
            "Sometimes that's exactly what's needed. How's everything else going for you?",
            "Good choice. Is there anything else on your mind today?",
        ])

    if text in {"not really","nah","nope","nothing","nothing much","not much",
                "no not really","i don't know","idk","not sure","don't know",
                "no idea","just checking in","just wanted to talk","bored"}:
        return pick([
            "That's totally fine. I'm here whenever something is on your mind.",
            "No worries at all. Feel free to come back anytime you want to talk.",
            "That's okay. Sometimes it helps just to have somewhere to check in. I'm here.",
        ])

    if text in {"fine","just fine","okay","ok","alright","i'm fine","im fine",
                "i'm okay","im okay","not bad","managing","surviving",
                "getting by","hanging in there","so so","so-so"}:
        return pick([
            "Just fine? Sometimes that's the best we can do. Is there anything on your mind today?",
            "Okay — is there something you've been thinking about lately, or did you just want to talk?",
            "Glad to hear it. Anything you'd like to talk through today?",
        ])

    if any(p in text for p in [
        "not good","not great","feeling bad","feel bad","feeling low",
        "not that good","not doing well","not okay","not ok","not fine",
        "not well","meh","could be better","not the best"
    ]):
        return ("I'm sorry to hear that. You don't have to carry that alone — I'm here to listen. "
                "Do you want to tell me a bit more about what's been going on?")

    is_help_seeking = any(p in text for p in HELP_SEEKING_PHRASES)
    user_msg_count = sum(1 for r, _ in st.session_state.messages if r == "user")
    # Only use generic "tell me more" responses on the FIRST message.
    # After that, let the full pipeline handle it so the bot actually addresses what was said.
    if is_help_seeking and len(words) <= 12 and user_msg_count <= 1:
        return pick([
            "I'm here. Take your time — what's been on your mind?",
            "Of course. I'm listening. What's been going on for you lately?",
            "I'm glad you reached out. Whenever you're ready, tell me what's weighing on you.",
            "Yes — let's talk. What would you like to start with?",
        ])

    for t in extract_topics(user_text):
        st.session_state.mentioned_topics.append(t)

    intent, emotion, confidence = predict_intent(user_text)
    st.session_state.last_confidence = confidence   # stored for UI display
    st.session_state.last_source     = None         # reset source each turn

    # ── CONFIDENCE THRESHOLD — proposal §4.4 ─────────────────────────────────
    # Only route to FAISS when:
    #   - confidence is low  AND
    #   - message is long enough to be a real query (not "bye", "ok", "thanks")
    #   - message is not a simple conversational reply
    _CASUAL = re.compile(
        r"^(bye|goodbye|ok|okay|thanks|thank\s*you|good|great|sure|"
        r"no|yes|yeah|nope|alright|fine|hello|hi|hey|see\s*you|later|"
        r"got\s*it|i\s*see|makes\s*sense|cool|nice|perfect)[\s!.?]*$",
        re.IGNORECASE,
    )
    is_casual   = bool(_CASUAL.match(user_text.strip()))
    is_too_short = len(user_text.split()) < 6

    if confidence < CONFIDENCE_THRESHOLD and not is_casual and not is_too_short:
        result = retrieve_from_corpus(user_text)
        if result:
            st.session_state.last_source = result["source"]
            return (
                f"Based on {result['source']}:\n\n"
                f"{result['text']}\n\n"
                f"*If you'd like to talk more about how you're feeling, I'm here.*"
            )

    effective_emotion = correct_emotion(user_text, emotion)

    if effective_emotion in ("joy","love") and not is_help_seeking:
        st.session_state.positive_turns += 1
        # After a couple of positive exchanges, stop fishing for more and
        # back off gracefully instead of asking the same question again.
        if st.session_state.positive_turns >= 3:
            st.session_state.positive_turns = 0
            return pick([
                "Sounds like things are genuinely in a good place — that's great. "
                "I'll leave it there; just know I'm around whenever you want to talk.",
                "It seems like you're doing well, which is really good to hear. "
                "No need to dig for problems — I'm here if anything comes up.",
                "Glad you're in a good spot. I won't keep prodding — feel free to "
                "come back anytime something's on your mind.",
            ])
        # Achievement words ("won", "passed", "got the job") deserve a real
        # congratulations rather than a generic "what's been helping?".
        if any(w in word_set for w in {
            "won","win","passed","aced","achieved","accomplished","nailed",
            "promoted","promotion","accepted","hired","graduated","succeeded"
        }):
            return pick([
                "That's wonderful — congratulations! How are you feeling about it?",
                "Congratulations, that's a real achievement! What did it take to get there?",
                "That's fantastic news. You should be proud — what's the win?",
                "Amazing, well done! How are you planning to celebrate?",
            ])
        return pick([
            "That's really good to hear. What's been going on that's keeping things feeling positive?",
            "Glad things are feeling okay. Anything on your mind you'd like to talk through?",
            "Good to hear. Is there something specific that's been going well?",
            "That's encouraging. What do you think has been helping?",
            "Love that for you. Is there anything you're looking forward to?",
            "Nice — it's good to hear something upbeat. What's been the highlight?",
        ])
    else:
        # Reset the positive streak whenever the mood shifts away from upbeat.
        st.session_state.positive_turns = 0

    st.session_state.intent_history.append(intent)
    if len(st.session_state.intent_history)>10:
        st.session_state.intent_history.pop(0)

    if should_escalate():
        st.session_state.escalation_shown = True
        return random.choice(ESCALATION_MESSAGES)

    # Short, low-context messages ("i have exams", "work is rough") don't carry
    # enough detail to answer well — and the counseling dataset's long forum
    # answers ramble off-topic. Respond concisely and probe for detail instead.
    if len(words) < 6 and not is_help_seeking:
        return pick(SHORT_SUPPORT.get(intent, SHORT_SUPPORT["default"]))

    row = semantic_retrieve(user_text, df, intent)
    if row is not None:
        rule = expert_system_rule(user_text, row)
        if rule:
            return rule

        callback = None
        if len(st.session_state.intent_history)%3==0:
            callback = topic_callback()

        core = trim_core(clean_response(row["bot_response"]))
        response, used_ack, used_fup = humanize_response(
            core, emotion, intent,
            last_ack=st.session_state.last_ack,
            raw_text=user_text,
            last_follow_up=st.session_state.last_follow_up,
            msg_count=user_msg_count,
        )
        st.session_state.last_ack = used_ack
        st.session_state.last_follow_up = used_fup
        if callback:
            response += " " + callback
        return response

    return pick([
        "I'm here to support you. Can you tell me a little more about what's been going on?",
        "I want to understand better. Can you share a bit more?",
        "I'm listening. What's been on your mind?",
    ])

# ── PERSISTENCE HELPER ────────────────────────────────────────────────────────
def persist_message(role, content):
    if st.session_state.active_conv is None:
        st.session_state.active_conv = db.create_conversation(
            st.session_state.user_id, "New Conversation"
        )
    db.add_message(st.session_state.active_conv, role, content)

def maybe_rename_conversation():
    if st.session_state.active_conv is None:
        return
    user_msg_count = sum(1 for r, _ in st.session_state.messages if r == "user")
    if user_msg_count == 2:
        title = db.auto_name(st.session_state.messages)
        db.rename_conversation(st.session_state.active_conv, title)

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    initial = (st.session_state.username[:1] or "U").upper()
    st.markdown(f"""
    <div class="brand-block">
        <div class="brand-logo">{LOGO_SVG}</div>
        <div>
            <div class="brand-name">MindSpace</div>
            <div class="brand-tagline">A private space to talk</div>
        </div>
    </div>
    <div class="user-chip">
        <div class="user-chip-avatar">{initial}</div>
        <div class="user-chip-name">{st.session_state.username}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="nav-label">Conversations</div>', unsafe_allow_html=True)
    if st.button("New Conversation", use_container_width=True, key="nav_new"):
        start_new_conversation()
        st.rerun()

    convs = db.list_conversations(st.session_state.user_id)
    if not convs:
        st.markdown('<div class="conv-empty">No past conversations yet.</div>',
                    unsafe_allow_html=True)
    else:
        for conv in convs[:15]:
            label = conv["title"]
            if len(label) > 28:
                label = label[:28] + "…"
            if st.button(label, use_container_width=True, key=f"conv_{conv['id']}"):
                load_conversation(conv["id"])
                st.rerun()

    st.markdown('<div class="nav-label">Wellness</div>', unsafe_allow_html=True)
    if st.button("Mood Journal", use_container_width=True, key="nav_journal"):
        st.session_state.page = "journal"; st.rerun()
    if st.button("Breathing Exercise", use_container_width=True, key="nav_breath"):
        st.session_state.page = "breathing"; st.rerun()
    if st.button("Sleep Help", use_container_width=True, key="nav_sleep"):
        st.session_state.page = "sleep"; st.rerun()
    if st.button("Grounding Exercise", use_container_width=True, key="nav_ground"):
        st.session_state.page = "grounding"; st.rerun()

    st.markdown('<hr class="sdiv">', unsafe_allow_html=True)
    st.markdown('<div class="nav-label">Account</div>', unsafe_allow_html=True)
    if st.button("Log Out", use_container_width=True, key="nav_logout"):
        logout()
        st.rerun()

# ── PAGES ─────────────────────────────────────────────────────────────────────
page = st.session_state.get("page", "chat")

if page == "breathing":
    st.markdown(f"""
    <div class="page-header">
        <div class="page-header-icon">B</div>
        <div><div class="page-header-title">Breathing Exercise</div>
        <div class="page-header-sub">Follow the circle</div></div>
    </div>
    <div class="page-card" style="text-align:center">
        <h2>Box Breathing</h2>
        <p>Watch the circle and breathe with it. Inhale as it grows, exhale as it shrinks.<br>
        Repeat for 2–5 minutes to calm your nervous system.</p>
        <div class="breath-circle">Breathe</div>
        <p style="margin-top:16px;font-size:12px;color:#3d4060;">
        4 seconds in &middot; 4 seconds hold &middot; 4 seconds out &middot; 4 seconds hold</p>
    </div>
    <div class="page-card">
        <h2>Why this helps</h2>
        <p>Controlled breathing activates your parasympathetic nervous system, reducing stress
        hormones and slowing your heart rate. Even 60 seconds can make a measurable difference.</p>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Back to chat", key="back_breath"):
        st.session_state.page = "chat"; st.rerun()

elif page == "grounding":
    st.markdown("""
    <div class="page-header">
        <div class="page-header-icon">G</div>
        <div><div class="page-header-title">Grounding Exercise</div>
        <div class="page-header-sub">5-4-3-2-1 technique</div></div>
    </div>
    <div class="page-card">
        <h2>The 5-4-3-2-1 Technique</h2>
        <p>This exercise brings you back to the present moment by engaging all five senses.
        Take your time with each step — there's no rush.</p>
    </div>
    <div class="grounding-step"><div class="grounding-num">5</div>
        <div class="grounding-text"><strong style="color:#dde1f0">things you can SEE</strong><br>
        Look around and name 5 things — a lamp, a window, your hands, anything.</div></div>
    <div class="grounding-step"><div class="grounding-num">4</div>
        <div class="grounding-text"><strong style="color:#dde1f0">things you can TOUCH</strong><br>
        Feel the texture of your chair, your clothes, the floor beneath your feet.</div></div>
    <div class="grounding-step"><div class="grounding-num">3</div>
        <div class="grounding-text"><strong style="color:#dde1f0">things you can HEAR</strong><br>
        Listen carefully. Maybe traffic, your breathing, a distant sound.</div></div>
    <div class="grounding-step"><div class="grounding-num">2</div>
        <div class="grounding-text"><strong style="color:#dde1f0">things you can SMELL</strong><br>
        Notice any scents in the air, or imagine a comforting smell.</div></div>
    <div class="grounding-step"><div class="grounding-num">1</div>
        <div class="grounding-text"><strong style="color:#dde1f0">thing you can TASTE</strong><br>
        Notice the taste in your mouth, or take a sip of water.</div></div>
    """, unsafe_allow_html=True)
    if st.button("Back to chat", key="back_ground"):
        st.session_state.page = "chat"; st.rerun()

elif page == "sleep":
    st.markdown("""
    <div class="page-header">
        <div class="page-header-icon">S</div>
        <div><div class="page-header-title">Sleep Help</div>
        <div class="page-header-sub">Wind down gently</div></div>
    </div>
    <div class="page-card">
        <h2>Can't sleep?</h2>
        <p>Sleep struggles are often tied to an overactive mind. These techniques help slow your
        thoughts and prepare your body for rest.</p>
    </div>
    <div class="page-card">
        <h2>The Military Sleep Method</h2>
        <p>Relax your face completely. Drop your shoulders, let your arms fall. Breathe out slowly
        and relax your chest. Release your legs from thigh to toe. Clear your mind for 10 seconds —
        imagine a warm, still lake. Most people fall asleep within 2 minutes.</p>
    </div>
    <div class="page-card">
        <h2>Before Bed Checklist</h2>
        <p>
        &middot; No screens 30 minutes before bed<br>
        &middot; Keep your room cool and dark<br>
        &middot; Write down tomorrow's worries so your brain can let them go<br>
        &middot; Try the 4-7-8 breath: inhale 4s, hold 7s, exhale 8s
        </p>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Back to chat", key="back_sleep"):
        st.session_state.page = "chat"; st.rerun()

elif page == "journal":
    st.markdown("""
    <div class="page-header">
        <div class="page-header-icon">J</div>
        <div><div class="page-header-title">Mood Journal</div>
        <div class="page-header-sub">Track how you feel over time</div></div>
    </div>
    <div class="page-card">
        <h2>Today's Entry</h2>
        <p>How are you feeling right now? Writing it down — even a few words — can bring clarity.</p>
    </div>
    """, unsafe_allow_html=True)
    mood = st.select_slider("Today's mood",
        options=["Very low","Low","Neutral","Good","Great"], value="Neutral")
    entry = st.text_area("Write about your day...", height=140,
        placeholder="What happened today? How did it make you feel?")
    if st.button("Save Entry", use_container_width=True, key="save_journal"):
        st.success("Entry saved. You're doing great by checking in with yourself.")
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Back to chat", key="back_journal"):
        st.session_state.page = "chat"; st.rerun()

else:
    # ── CHAT PAGE ─────────────────────────────────────────────────────────────
    if st.session_state.active_conv:
        active_title = next(
            (c["title"] for c in db.list_conversations(st.session_state.user_id)
             if c["id"] == st.session_state.active_conv),
            "Conversation",
        )
    else:
        active_title = "New Conversation"

    st.markdown(f"""
    <div class="chat-header">
        <div class="chat-header-logo">{LOGO_SVG}</div>
        <h1 class="chat-header-name">MindSpace</h1>
        <div class="chat-header-sub">{active_title} &middot; Private &amp; Safe</div>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.crisis_active:
        st.markdown("""
        <div class="crisis-banner">
            <div class="crisis-banner-title">You reached out, and that took courage.</div>
            <div class="crisis-banner-text">
                You don't have to go through this alone. Please consider reaching out to someone
                who can help right now — you matter.
            </div>
        </div>
        """, unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            st.link_button("Find a helpline near you",
                           "https://www.findahelpline.com", use_container_width=True)
        with c2:
            if st.button("Continue talking", use_container_width=True, key="cb_continue"):
                st.session_state.crisis_active = False
                st.rerun()

    if not st.session_state.messages:
        st.markdown("""
        <div class="welcome-card">
            <div class="welcome-title">I'm here to listen, not judge.</div>
            <div class="welcome-sub">
                Whatever you're carrying right now, you don't have to carry it alone.
                Type below or pick a feeling to start.
            </div>
        </div>
        """, unsafe_allow_html=True)

    for role, msg in st.session_state.messages:
        avatar = BOT_AVATAR if role == "assistant" else USER_AVATAR
        with st.chat_message(role, avatar=avatar):
            st.write(msg)

    if not st.session_state.messages:
        st.markdown("---")
        shortcut_cols = st.columns(len(FEELINGS_SHORTCUTS))
        for i, feeling in enumerate(FEELINGS_SHORTCUTS):
            with shortcut_cols[i]:
                if st.button(feeling, key=f"fs_{i}", use_container_width=True):
                    st.session_state.pending_voice = feeling
                    st.rerun()

    # Mic button — JS dynamically positions it inside chat input
    with st.container(key="mic_fab"):
        mic_clicked = st.button(" ", key="mic_main")

    # Inject script to anchor the mic button to the right edge of the chat input.
    # Runs inside a small iframe component but reaches into parent document.
    components.html("""
    <script>
    (function() {
        const parentDoc = window.parent.document;
        function placeMic() {
            const input = parentDoc.querySelector('[data-testid="stChatInput"]');
            let mic = parentDoc.querySelector('[class*="st-key-mic_fab"]');
            if (!mic) {
                // fallback: find the empty-label button just before stBottom
                const buttons = parentDoc.querySelectorAll('div.stButton');
                for (const b of buttons) {
                    const btn = b.querySelector('button');
                    if (btn && btn.textContent.trim() === '') { mic = b; break; }
                }
            }
            if (!input || !mic) return;
            const r = input.getBoundingClientRect();
            mic.style.position = 'fixed';
            mic.style.top = (r.top + (r.height - 44) / 2) + 'px';
            mic.style.left = (r.right - 96) + 'px';
            mic.style.right = 'auto';
            mic.style.bottom = 'auto';
            mic.style.width = '44px';
            mic.style.height = '44px';
            mic.style.zIndex = '1001';
            mic.style.margin = '0';
            mic.style.padding = '0';
            mic.classList.add('mic-anchored');
        }
        placeMic();
        setInterval(placeMic, 250);
        window.parent.addEventListener('resize', placeMic);
    })();
    </script>
    """, height=0)

    if mic_clicked:
        r = sr.Recognizer()
        r.energy_threshold = 300
        try:
            with sr.Microphone() as src:
                r.adjust_for_ambient_noise(src, duration=0.5)
                with st.spinner("Listening..."):
                    audio = r.listen(src, timeout=8, phrase_time_limit=10)
            recognized = r.recognize_google(audio)
            st.session_state.pending_voice = recognized
            st.rerun()
        except sr.WaitTimeoutError:
            st.toast("No speech detected. Try again.")
        except sr.UnknownValueError:
            st.toast("Couldn't understand. Please speak clearly.")
        except Exception:
            st.toast("Microphone unavailable. Check permissions.")

    user_input = st.chat_input("Share what you're feeling...")

    if st.session_state.pending_voice:
        user_input = st.session_state.pending_voice
        st.session_state.pending_voice = None

    if user_input:
        st.session_state.messages.append(("user", user_input))
        persist_message("user", user_input)

        with st.chat_message("assistant", avatar=BOT_AVATAR):
            typing_ph = st.empty()
            typing_ph.markdown("""
            <div class="typing-card">
                <div class="typing-dots"><span></span><span></span><span></span></div>
                <span class="typing-text">Listening...</span>
            </div>""", unsafe_allow_html=True)
            response = get_bot_response(user_input)
            typing_ph.empty()
            st.write(response)

            # ── Confidence badge + source (proposal §4.4 / §5.2) ─────────────
            conf = st.session_state.get("last_confidence")
            src  = st.session_state.get("last_source")
            if conf is not None:
                colour = "#22c55e" if conf >= CONFIDENCE_THRESHOLD else "#f97316"
                label  = "High confidence" if conf >= CONFIDENCE_THRESHOLD else "Low confidence → WHO/UNICEF source used"
                st.markdown(
                    f'<div style="font-size:11px;color:{colour};margin-top:4px;">'
                    f'🔍 {label} &nbsp;|&nbsp; score: <b>{conf:.2f}</b>'
                    + (f'&nbsp;|&nbsp; 📄 {src}' if src else '')
                    + '</div>',
                    unsafe_allow_html=True
                )

        st.session_state.messages.append(("assistant", response))
        persist_message("assistant", response)
        maybe_rename_conversation()
        st.rerun()
