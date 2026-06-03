import os
import re
import hashlib
import sqlite3
from datetime import datetime
from contextlib import contextmanager

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "mindspace.db")


@contextmanager
def _conn():
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    try:
        yield c
        c.commit()
    finally:
        c.close()


def init_db():
    with _conn() as c:
        c.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                title TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id INTEGER NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (conversation_id) REFERENCES conversations(id)
            )
        """)


# ── Password hashing (PBKDF2-SHA256, no external deps) ──────────────────────
def _hash_password(password: str, salt: bytes = None) -> str:
    if salt is None:
        salt = os.urandom(16)
    h = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 200_000)
    return f"{salt.hex()}:{h.hex()}"


def _verify_password(password: str, stored: str) -> bool:
    try:
        salt_hex, hash_hex = stored.split(":")
        salt = bytes.fromhex(salt_hex)
        h = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 200_000).hex()
        return h == hash_hex
    except Exception:
        return False


# ── Users ────────────────────────────────────────────────────────────────────
def create_user(username: str, password: str):
    username = username.strip()
    if not username or not password:
        return None, "Username and password required."
    if len(password) < 6:
        return None, "Password must be at least 6 characters."
    pw_hash = _hash_password(password)
    try:
        with _conn() as c:
            cur = c.execute(
                "INSERT INTO users (username, password_hash, created_at) VALUES (?, ?, ?)",
                (username, pw_hash, datetime.now().isoformat()),
            )
            return cur.lastrowid, None
    except sqlite3.IntegrityError:
        return None, "That username is already taken."


def verify_user(username: str, password: str):
    with _conn() as c:
        row = c.execute(
            "SELECT id, password_hash FROM users WHERE username = ?",
            (username.strip(),),
        ).fetchone()
    if not row:
        return None
    if _verify_password(password, row["password_hash"]):
        return row["id"]
    return None


# ── Conversations ────────────────────────────────────────────────────────────
def create_conversation(user_id: int, title: str = "New Conversation") -> int:
    now = datetime.now().isoformat()
    with _conn() as c:
        cur = c.execute(
            "INSERT INTO conversations (user_id, title, created_at, updated_at) "
            "VALUES (?, ?, ?, ?)",
            (user_id, title, now, now),
        )
        return cur.lastrowid


def list_conversations(user_id: int):
    with _conn() as c:
        rows = c.execute(
            "SELECT id, title, updated_at FROM conversations "
            "WHERE user_id = ? ORDER BY updated_at DESC",
            (user_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def get_messages(conversation_id: int):
    with _conn() as c:
        rows = c.execute(
            "SELECT role, content FROM messages "
            "WHERE conversation_id = ? ORDER BY id",
            (conversation_id,),
        ).fetchall()
    return [(r["role"], r["content"]) for r in rows]


def add_message(conversation_id: int, role: str, content: str):
    now = datetime.now().isoformat()
    with _conn() as c:
        c.execute(
            "INSERT INTO messages (conversation_id, role, content, created_at) "
            "VALUES (?, ?, ?, ?)",
            (conversation_id, role, content, now),
        )
        c.execute(
            "UPDATE conversations SET updated_at = ? WHERE id = ?",
            (now, conversation_id),
        )


def rename_conversation(conversation_id: int, title: str):
    with _conn() as c:
        c.execute(
            "UPDATE conversations SET title = ? WHERE id = ?",
            (title, conversation_id),
        )


def delete_conversation(conversation_id: int):
    with _conn() as c:
        c.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))
        c.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))


# ── Auto-naming ──────────────────────────────────────────────────────────────
TOPIC_RULES = [
    ("Sleep difficulties",  ["sleep", "insomnia", "tired", "rest", "wake", "exhausted"]),
    ("Anxiety",             ["anxious", "anxiety", "panic", "worry", "worried", "nervous", "scared"]),
    ("Work stress",         ["work", "job", "boss", "career", "deadline", "office", "burnout"]),
    ("Relationships",       ["relationship", "boyfriend", "girlfriend", "partner", "marriage", "breakup", "dating"]),
    ("Family",              ["mom", "dad", "mother", "father", "parents", "family", "sibling", "brother", "sister"]),
    ("Studies",             ["study", "studies", "exam", "exams", "school", "college", "university", "homework", "thesis"]),
    ("Loneliness",          ["alone", "lonely", "isolated", "no one", "nobody"]),
    ("Sadness",             ["sad", "depress", "down", "low", "empty", "hopeless", "crying"]),
    ("Anger",               ["angry", "anger", "frustrat", "rage", "mad", "furious"]),
    ("Self-doubt",          ["worthless", "useless", "failure", "not good enough", "broken"]),
    ("Grief",               ["loss", "lost", "died", "death", "grief", "miss them"]),
    ("Money",               ["money", "debt", "rent", "bills", "broke", "afford"]),
]


def auto_name(messages):
    """Pick a title from the first 1–3 user messages."""
    user_msgs = [m for r, m in messages if r == "user"]
    if not user_msgs:
        return "New Conversation"
    blob = " ".join(user_msgs[:3]).lower()
    for label, keywords in TOPIC_RULES:
        if any(k in blob for k in keywords):
            return label
    first = re.sub(r"\s+", " ", user_msgs[0]).strip()
    if len(first) > 32:
        first = first[:32].rstrip() + "…"
    return first.capitalize() if first else "New Conversation"
