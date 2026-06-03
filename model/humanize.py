import random
import re

# Words that signal negative emotional state — used to correct BERT misclassifications
NEGATIVE_INDICATORS = {
    "hard", "difficult", "struggling", "struggle", "pain", "hurt", "sad", "bad",
    "worse", "terrible", "awful", "hopeless", "lost", "alone", "empty", "tired",
    "exhausted", "overwhelmed", "scared", "afraid", "worried", "anxious", "depressed",
    "miserable", "worthless", "useless", "helpless", "broken", "crying", "tears",
    "heavy", "burden", "dark", "numb", "stuck", "trapped", "drained", "angry",
    "frustrated", "stressed", "pressure", "unbearable", "pointless", "lonely",
    "meh", "bleh", "ugh", "sucks", "horrible", "awful", "rough", "rough"
}

# Regex patterns that signal negativity even when individual words look positive
NEGATION_PATTERNS = [
    r"\bnot\s+(\w+\s+)?(good|great|okay|ok|fine|well|happy|alright|doing\s+well)\b",
    r"\bnot\s+feeling\s+(good|great|okay|ok|fine|well|happy)\b",
    r"\b(meh|bleh|so[\s-]so|not\s+really|not\s+much)\b",
    r"\b(can'?t|don'?t|won'?t|couldn'?t)\s+(cope|manage|sleep|function|feel)\b",
]

# High-pressure / stressful topics. A neutral-sounding statement that mentions
# these ("i got final exams") is almost never genuine "joy" — treat as stress.
STRESS_TOPICS = {
    "exam", "exams", "test", "tests", "midterm", "midterms", "final", "finals",
    "deadline", "deadlines", "assignment", "assignments", "homework", "project",
    "interview", "presentation", "viva", "thesis", "dissertation", "grades",
    "results", "boss", "work", "job", "money", "rent", "bills", "debt",
    "breakup", "divorce", "fight", "argument", "sick", "illness", "diagnosis",
}

# Words that signal genuine positive sentiment. Required for a message to be
# treated as "joy" — absence of negativity is NOT the same as being upbeat.
POSITIVE_INDICATORS = {
    "happy", "great", "good", "better", "wonderful", "amazing", "excited",
    "glad", "grateful", "thankful", "relieved", "proud", "calm", "peaceful",
    "hopeful", "love", "loved", "joy", "joyful", "fantastic", "awesome",
    "excellent", "well", "fine", "okay", "ok", "enjoy", "enjoyed", "fun",
    "celebrate", "celebrating", "win", "won", "winning", "passed", "passing",
    "achieved", "achievement", "improving", "improved", "aced", "nailed",
    "accomplished", "promoted", "promotion", "hired", "accepted", "graduated",
    "succeeded", "success", "best", "thrilled", "delighted", "ecstatic",
}

ACKNOWLEDGMENTS = {
    "sadness": [
        "That sounds really hard.",
        "I'm sorry you're carrying that.",
        "That takes courage to share.",
        "I can hear how heavy this feels.",
        "You didn't deserve to feel this way.",
    ],
    "anger": [
        "It makes complete sense that you're frustrated.",
        "That sounds genuinely overwhelming.",
        "I understand why that would feel that way.",
        "I can hear how much this has been affecting you.",
        "That sounds really draining to deal with.",
    ],
    "fear": [
        "It's okay to feel scared about this.",
        "Uncertainty can feel so heavy sometimes.",
        "That sounds really unsettling.",
        "It makes sense to feel anxious about that.",
        "I hear you -- that sounds frightening.",
    ],
    "joy": [
        "I'm really glad to hear that.",
        "That's genuinely wonderful.",
        "It's great that you're feeling that way.",
        "That sounds like a real positive step.",
        "I'm happy you shared that with me.",
    ],
    "love": [
        "That's really meaningful.",
        "It sounds like that matters a lot to you.",
        "I can feel how much that means to you.",
        "That's a beautiful thing to hold on to.",
        "It's clear that comes from a place of care.",
    ],
    "surprise": [
        "That must have caught you off guard.",
        "Unexpected things can be really hard to process.",
        "It makes sense you're still sitting with that.",
        "That sounds like a lot to take in.",
        "I can imagine that felt quite sudden.",
    ],
    "fallback": [
        "Thank you for sharing that with me.",
        "I hear you.",
        "That means a lot that you felt comfortable sharing.",
        "I'm glad you reached out.",
        "I'm here with you.",
    ],
}

# ── Probing follow-ups — used only for the FIRST user message ────────────────
FOLLOW_UPS = {
    "depression_symptoms": [
        "What's been weighing on you the most lately?",
        "Has anything in particular brought these feelings on?",
        "How long have you been feeling this way?",
        "Is there one thing that's been the hardest to deal with?",
    ],
    "anxiety_symptoms": [
        "When did you first start feeling this way?",
        "Is there a specific situation that tends to trigger it?",
        "What does it feel like when the anxiety hits hardest?",
        "Have you noticed anything that makes it better or worse?",
    ],
    "stress_reaction": [
        "What feels like the biggest source of pressure right now?",
        "How long have you been under this much pressure?",
        "Is there one thing you could take off your plate, even temporarily?",
        "What does a good day look like for you at the moment?",
    ],
    "coping_mechanism": [
        "Have you tried any of these before, or does this feel new to you?",
        "What has helped you most in the past when things got tough?",
        "Is there one small thing you could try today?",
        "What does support look like for you personally?",
    ],
    "emotional_support": [
        "What's been on your mind the most today?",
        "Would you like to talk more about what's been going on?",
        "What's been the hardest part lately?",
        "What would feel most helpful right now?",
    ],
    "professional_help": [
        "What's been making you consider reaching out for support?",
        "Have you spoken to anyone about this before?",
        "What feels like the biggest barrier to getting help right now?",
    ],
    "mental_health_faq": [
        "Is there something specific about this you've been wondering about?",
        "Does any of this connect to something you've been experiencing yourself?",
        "Is there anything else you'd like to understand better?",
    ],
    "suicide_warning": [],
}

# ── Consoling follow-ups — used AFTER user has already shared their problem ──
CONSOLING_FOLLOW_UPS = {
    "depression_symptoms": [
        "Even on the hardest days, you're still moving forward -- and that matters.",
        "Is there one small thing that has brought you any comfort lately, even briefly?",
        "You don't have to fix everything at once. Just getting through today is enough.",
        "Have you been able to talk to anyone about how you've been feeling?",
    ],
    "anxiety_symptoms": [
        "Sometimes just naming what you're anxious about can take some of its power away.",
        "You're not weak for feeling this way -- anxiety is your mind trying to protect you.",
        "What's one thing that usually helps you feel even a little grounded?",
        "Try to focus on just the next hour, not the whole picture.",
    ],
    "stress_reaction": [
        "It might help to pick just one thing to focus on first -- you don't have to tackle everything at once.",
        "Is there anything on your plate right now that you could let go of, even temporarily?",
        "Sometimes the pressure builds because we forget to give ourselves permission to rest.",
        "Have you been able to take any breaks lately, even small ones?",
    ],
    "coping_mechanism": [
        "Even tiny acts of self-care matter more than you think.",
        "What's one small thing you could do for yourself today?",
        "Finding what works takes time -- be patient with yourself.",
        "Is there something that has helped you before, even a little?",
    ],
    "emotional_support": [
        "You're not alone in this, even when it feels that way.",
        "It takes strength to sit with hard emotions. You're doing better than you think.",
        "What would feel most supportive for you right now?",
        "You deserve the same kindness you'd give to someone you care about.",
    ],
    "professional_help": [
        "Reaching out for support is one of the bravest things you can do.",
        "A professional can offer tools that really make a difference -- it's worth considering.",
        "You don't have to figure this out alone.",
    ],
    "mental_health_faq": [
        "Understanding what you're going through is already a meaningful step.",
        "Is there anything about this that connects to what you've been feeling?",
    ],
    "suicide_warning": [],
    "fallback": [
        "What would feel most helpful for you right now?",
        "You're already doing something brave by talking about this.",
        "You deserve support -- and it's okay to ask for it.",
    ],
}

# ── Patterns that detect "tell me more" type responses ───────────────────────
PROBING_PATTERNS = [
    r"tell me (more|what|about)",
    r"share (more|what|how)",
    r"what('s| is| has) (been )?(going on|happening|on your mind|weighing|bothering)",
    r"(can|could|would) you (tell|share|explain|like to talk)",
    r"whenever you'?re ready",
    r"what would you like to",
    r"do you want to (talk|share|tell|open)",
    r"i'?m (here )?(to )?(listen|hear you)",
    r"(talk|share|open up) (about|more)",
    r"what brings you here",
    r"let me know (what|how|when)",
    r"feel free to (share|talk|tell)",
]


def _is_probing(text):
    """Check if a response is just asking for more info rather than addressing the concern."""
    lower = text.lower()
    return any(re.search(p, lower) for p in PROBING_PATTERNS)


# ── Consoling core responses — replace probing dataset responses ─────────────
CONSOLING_CORES = {
    "stress_reaction": [
        "When everything piles up at once, it can feel impossible to see a way through -- but you're already coping better than you think by recognizing it.",
        "Feeling crushed under pressure doesn't mean you're failing. It means you care deeply, and that matters.",
        "It's okay to not have everything figured out right now. Sometimes just getting through the day is a real achievement.",
        "The weight you're carrying is real. You don't need to push through it all at once -- one step at a time is enough.",
    ],
    "depression_symptoms": [
        "These feelings are heavy, but they don't define you. Even on the hardest days, you're still here -- and that matters.",
        "It's okay to not be okay right now. What you're feeling is real, and you deserve gentleness.",
        "You don't have to pretend everything is fine. It's okay to feel exactly how you feel right now.",
        "The heaviness you're feeling won't last forever, even though it might feel that way right now.",
    ],
    "anxiety_symptoms": [
        "Anxiety can make everything feel urgent and impossible at the same time. You don't have to solve it all right now.",
        "Your mind is trying to protect you, even when it feels like it's working against you. That's not a flaw -- it's human.",
        "It's okay to take things one moment at a time. You don't need to have all the answers right now.",
        "The worry you're feeling is your mind working overtime. You deserve a break from that constant pressure.",
    ],
    "emotional_support": [
        "What you're going through sounds genuinely difficult, and it's okay to feel this way.",
        "You're not weak for feeling this way. It takes real strength to sit with hard emotions.",
        "These feelings are valid -- you're dealing with something real, and you deserve support.",
        "It's okay to struggle. It doesn't mean something is wrong with you -- it means you're human.",
    ],
    "coping_mechanism": [
        "Finding ways to cope takes time. Be patient with yourself -- there's no right or wrong way to heal.",
        "Even small acts of self-care matter more than you think. You deserve that kindness from yourself.",
        "Coping isn't about fixing everything. Sometimes it's just about getting through the next moment.",
    ],
    "professional_help": [
        "Wanting support is a sign of strength, not weakness. You deserve to have someone in your corner.",
        "Professional help can give you tools to manage what you're feeling -- and you deserve that kind of support.",
    ],
    "mental_health_faq": [
        "Understanding what you're experiencing is already a meaningful step forward.",
        "Learning about what you're going through shows real self-awareness. That matters.",
    ],
    "fallback": [
        "What you're going through sounds genuinely difficult, and it's okay to feel this way.",
        "It makes complete sense that you'd feel this way given everything you're dealing with.",
        "That's a lot to carry, and you shouldn't have to push through it alone.",
        "These feelings are valid -- you're dealing with something real and heavy.",
    ],
}


def correct_emotion(text, bert_emotion):
    """
    Decide the true emotional tone of `text`, overriding the classifier when the
    words clearly contradict it. The classifier's emotion is derived from the
    predicted intent, so it can be wrong in BOTH directions — this resolves tone
    symmetrically from the actual sentiment words in the message.

    Precedence (strongest signal wins):
      1. Explicit negative words or negation ("worried", "not good", "can't cope") → sadness
      2. Genuine positive sentiment ("won", "passed my exam", "feeling great")    → joy
      3. Neutral mention of a stressful topic ("i got final exams")               → fear
      4. Classifier said positive but there's no real positive sentiment          → neutral
    """
    text_lower   = text.lower()
    word_set     = set(re.findall(r"[a-z']+", text_lower))
    has_negative = bool(word_set & NEGATIVE_INDICATORS)
    has_negation = any(re.search(p, text_lower) for p in NEGATION_PATTERNS)
    # Positive idioms where no single word is obviously upbeat ("got the job").
    has_pos_phrase = bool(re.search(
        r"\bgot (the|a|an|my) (job|offer|role|position|place|spot)\b|"
        r"\bgot (in|accepted|engaged|married|promoted)\b|"
        r"\bgot into \w+|\baced (it|the|my)\b|\bgood news\b",
        text_lower,
    ))
    has_positive = bool(word_set & POSITIVE_INDICATORS) or has_pos_phrase
    has_stress   = bool(word_set & STRESS_TOPICS)

    # 1. Clear negativity always wins (a downbeat statement misread as positive).
    if has_negative or has_negation:
        return "sadness" if bert_emotion in ("joy", "love", "surprise") else bert_emotion

    # 2. Genuine positive sentiment, with no negativity → joy, even if the
    #    classifier guessed a negative intent ("i won this competition!").
    if has_positive:
        return "joy"

    # 3. Neutral statement mentioning a stressful topic → treat as anxiety/stress.
    if has_stress:
        return "fear"

    # 4. Classifier claimed positive but there's no actual positive cue →
    #    neutral, so flat factual statements don't get cheery replies.
    if bert_emotion in ("joy", "love", "surprise"):
        return "neutral"

    return bert_emotion


def humanize_response(core_response, emotion, intent, last_ack=None,
                      raw_text="", last_follow_up=None, msg_count=0):
    """
    Build a 3-part response: acknowledgment + core + follow-up.

    msg_count = number of user messages so far (BEFORE this one).
      0 or 1: user is just starting → use probing follow-ups to understand the problem
      2+: user has already shared → use consoling follow-ups to validate and support
    """
    effective_emotion = correct_emotion(raw_text, emotion)

    # ── 1. Pick acknowledgment ───────────────────────────────────────────────
    ack_pool = ACKNOWLEDGMENTS.get(effective_emotion, ACKNOWLEDGMENTS["fallback"])
    available_acks = [a for a in ack_pool if a != last_ack]
    if not available_acks:
        available_acks = ack_pool

    # Avoid acknowledgments whose words heavily overlap with the core response
    core_lower = core_response.lower()
    non_overlapping = [
        a for a in available_acks
        if sum(1 for w in a.lower().split() if w in core_lower) / max(len(a.split()), 1) < 0.5
    ]
    ack = random.choice(non_overlapping if non_overlapping else available_acks)

    # ── 2. Pick core response — replace probing dataset answers after msg 1 ──
    effective_core = core_response
    if msg_count >= 2 and _is_probing(core_response):
        consoling_pool = CONSOLING_CORES.get(intent, CONSOLING_CORES["fallback"])
        effective_core = random.choice(consoling_pool)

    # ── 3. Pick follow-up — consoling after first exchange, probing at start ─
    if msg_count >= 2:
        follow_up_pool = CONSOLING_FOLLOW_UPS.get(
            intent, CONSOLING_FOLLOW_UPS.get("fallback", [])
        )
    else:
        follow_up_pool = FOLLOW_UPS.get(intent, [])

    available_fups = [f for f in follow_up_pool if f != last_follow_up]
    if not available_fups:
        available_fups = follow_up_pool
    follow_up = random.choice(available_fups) if available_fups else ""

    # ── 4. Assemble ──────────────────────────────────────────────────────────
    parts = [ack, effective_core]
    if follow_up:
        parts.append(follow_up)

    full_response = " ".join(parts)
    return full_response, ack, follow_up
