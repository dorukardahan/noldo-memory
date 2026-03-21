"""Trigger patterns and importance scoring.

Ported from Mahmory v6 ``memory_skill.py`` / ``importance.py``, cleaned up
and adapted for Turkish+English environment.
"""

from __future__ import annotations

import os
import re
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Trigger patterns
# ---------------------------------------------------------------------------

TURKISH_TRIGGERS: List[str] = [
    # Memory / recall
    r"hatırl", r"hatirl",
    r"neydi",
    r"ne\s*konuştuk", r"ne\s*konustuk",
    # Time references
    r"dün", r"dun",
    r"geçen", r"gecen",
    r"önceki", r"onceki",
    r"ne\s*zaman",
    # Past discussion
    r"bahsetmiştik", r"bahsetmistik",
    r"söylemiştik", r"soylemistik",
    r"söylemiştim", r"soylemistim",
    # Decision / preference
    r"karar",
    r"sevdiğim", r"sevdigim", r"sevmediğim", r"sevmedigim",
    r"unutma",
    r"her\s*zaman", r"asla",
    # Identity
    r"benim\s*(?:adım|ismim)",
    r"benim\s*(?:adim|ismim)",
    # Context questions
    r"ne\s*yapıyorduk", r"ne\s*yapiyorduk",
    r"neredeydi",
    r"nerede\s*kaldık", r"nerede\s*kaldik",
    r"devam",
    r"son\s*durum",
    r"hakkında", r"hakkinda",
    r"ile\s*ilgili",
    r"konusunda",
]

ENGLISH_TRIGGERS: List[str] = [
    r"remember", r"recall",
    r"what\s*did\s*(?:we|[iI])\s*(?:say|talk|discuss)",
    r"last\s*time", r"previously",
    r"my\s*(?:favorite|favourite|name|preference)",
    r"[iI]\s*(?:like|prefer|hate|love|want|need)",
    r"always", r"never", r"important",
    r"where\s*were\s*we",
    r"what\s*were\s*we",
    r"continue", r"resume",
    r"left\s*off", r"working\s*on",
    r"about", r"regarding",
]

# Patterns for messages that should NOT trigger memory search
ANTI_TRIGGER_PATTERNS: List[str] = [
    # Single-word acknowledgements / reactions
    r"^(?:ok|tamam|evet|hayır|hayir|anladım|anladim|tamamdır|tamamdir|oldu|olur|güzel|harika|süper)[\s!.]*$",
    r"^(?:👍|😂|😊|🙏|👀|❤️|🔥|✅|💯|🤝|🎉)[\s]*$",
    # Greetings (no memory context needed)
    r"^(?:merhaba|selam|hey|hi|hello|nasılsın|nasilsin|naber|iyi\s*geceler|iyi\s*günler)[\s?!]*$",
    # Thanks (no memory context needed)
    r"^(?:teşekkür|tesekkur|sağol|sagol|eyvallah|thanks|thx|ty|thank\s*you)[\s!]*$",
    # Imperative commands (agent should just do it, no recall needed)
    r"^(?:yap|oluştur|olustur|gönder|gonder|aç|ac|kapat|başla|basla|bitir|sil|düzelt|duzelt|çalıştır|calistir)[\s!]*$",
    # Short confirmations / follow-ups
    r"^(?:onu|bunu|şunu|onu\s*yap|devam\s*et|git|gel|dur|bekle|sus)[\s!]*$",
    # Numeric responses
    r"^\d{1,4}[\s.!]*$",
    # Single letter / word too short for meaningful recall
    r"^[a-zA-ZçğıöşüÇĞİÖŞÜ]{1,2}[\s?!]*$",
]

# Past-tense heuristic (Turkish + English)
_PAST_TENSE_RE = re.compile(
    r"(?:mıştı|misti|mişti|muştu|mustu|müştü|dık|dik|duk|dük|aldı|aldi|yaptı|yapti|gitti|geldi|söyledi|soyledi"
    r"|was|were|did|had)",
    re.IGNORECASE,
)


def should_trigger(text: str) -> bool:
    """Determine whether *text* should trigger a memory search.

    Returns ``True`` if any trigger pattern matches and no anti-trigger fires.
    """
    text_stripped = text.strip()
    text_lower = text_stripped.lower()

    # Anti-triggers
    for pattern in ANTI_TRIGGER_PATTERNS:
        if re.match(pattern, text_stripped, re.IGNORECASE | re.UNICODE):
            return False

    # Too short
    if len(text_stripped) < 3:
        return False

    # Too generic (single common words)
    if text_lower in {"o", "şey", "sey", "bu", "şu", "su", "ne", "it", "that", "this"}:
        return False

    # Single emoji
    if len(text_stripped) <= 4 and not any(c.isalpha() for c in text_stripped):
        return False

    # Check Turkish triggers
    for pattern in TURKISH_TRIGGERS:
        if re.search(pattern, text_lower, re.IGNORECASE):
            return True

    # Check English triggers
    for pattern in ENGLISH_TRIGGERS:
        if re.search(pattern, text_lower, re.IGNORECASE):
            return True

    # Question mark heuristic (long enough question)
    if "?" in text and len(text) > 10:
        return True

    # Past tense heuristic
    if _PAST_TENSE_RE.search(text_lower):
        return True

    return False


# ---------------------------------------------------------------------------
# Importance scoring (0.0 – 1.0) — Recalibrated 2026-02-17 [S1]
#
# Design: base=0.20, most messages 0.20-0.50, only explicit markers/decisions
# push above 0.70. Previous version had base=0.50 causing 33% at 0.9+.
# ---------------------------------------------------------------------------

_IMPORTANCE_MARKERS: List[str] = [
    "hatırla", "hatirla", "unutma", "önemli", "onemli", "kritik",
    "acil", "kesinlikle", "mutlaka", "dikkat",
    "remember", "don't forget", "important", "critical",
    "urgent", "must", "definitely", "attention",
]

_DECISION_MARKERS: List[str] = [
    "karar", "kararlaştırdık", "kararlastirdik", "yapacağım", "yapacagim", "yapacağız", "yapacagiz",
    "söz", "soz", "tamamdır", "tamamdir", "anlaştık", "anlastik", "kabul",
    "decided", "will do", "agreed", "commitment",
    "plan is", "going to", "promise", "deal",
]

_TASK_MARKERS: List[str] = [
    "todo", "yapılacak", "yapilacak", "görev", "gorev", "task",
    "action item", "next step",
]

_OPS_MARKERS: List[str] = [
    "deploy", "deployed", "restart", "restarted", "migrate", "migrated",
    "taşıdık", "taşıdım", "deploy ettim", "restart ettim",
    "config", "konfigürasyon", ".env", "systemd", "systemctl",
    "merge", "merged", "commit", "pushed", "pull request",
    "sigkill", "oom", "crash", "hata", "error", "failed",
    "backup", "yedek", "version", "upgrade", "güncelle",
]

_NOISE_PATTERNS: List[str] = [
    r"^(?:ok|okay|tamam|evet|yes|no|hayır|hayir|hmm|haha|lol)$",
    r"^(?:thanks|teşekkür|tesekkur|sağol|sagol)[\s!.]*$",
    r"^\d+$",
    r"^[\W]+$",
]

# System noise — gateway connects, test msgs, cron boilerplate
_SYSTEM_NOISE_PATTERNS = [
    re.compile(r"whatsapp gateway (?:connected|disconnected)", re.IGNORECASE),
    re.compile(r"slack (?:socket mode )?(?:connected|disconnected)", re.IGNORECASE),
    re.compile(r"^GatewayRestart:", re.IGNORECASE),
    re.compile(r"^\[queued messages", re.IGNORECASE),
    re.compile(r"^say\s+(?:ok|hello|hi|test|something)\s*$", re.IGNORECASE),
    re.compile(r"^Conversation info \(untrusted metadata\)", re.IGNORECASE),
    re.compile(r"^Replied message \(untrusted", re.IGNORECASE),
]


# Cron / automated output patterns — cap at 0.30
_CRON_NOISE_PATTERNS = [
    re.compile(r"^\[cron:", re.IGNORECASE),
    re.compile(r"/steward-(?:engage|post|digest)", re.IGNORECASE),
    re.compile(r"Bureau Engage", re.IGNORECASE),
    re.compile(r"steward-engage", re.IGNORECASE),
    re.compile(r"^HEARTBEAT_OK\s*$", re.IGNORECASE),
    re.compile(r"\[cron:[a-f0-9-]+\s", re.IGNORECASE),
    re.compile(r"cron job .+ just completed", re.IGNORECASE),
    re.compile(r"Return your summary as plain text", re.IGNORECASE),
    re.compile(r"Current time: \w+day,", re.IGNORECASE),
]
_CRON_NOISE_MAX = 0.30

# Conversation source indicators — generic by default, override via env if needed.
_DEFAULT_SLACK_DM_SOURCE_REGEX = r"Slack DM from [^:\n]+:"
_CONVERSATION_SOURCE_REGEX = os.environ.get(
    "AGENT_MEMORY_CONVERSATION_SOURCE_REGEX",
    _DEFAULT_SLACK_DM_SOURCE_REGEX,
)

try:
    _SLACK_DM_SOURCE_PATTERN = re.compile(_CONVERSATION_SOURCE_REGEX, re.IGNORECASE)
except re.error:
    _SLACK_DM_SOURCE_PATTERN = re.compile(_DEFAULT_SLACK_DM_SOURCE_REGEX, re.IGNORECASE)

_CONVERSATION_SOURCE_PATTERNS = [
    _SLACK_DM_SOURCE_PATTERN,
    # Add your own WhatsApp/phone patterns here if needed:
    # re.compile(r"\[WhatsApp \+1234567890", re.IGNORECASE),
    re.compile(r"Conversation info.*\"conversation_label\"", re.IGNORECASE | re.DOTALL),
]
_CONVERSATION_BONUS = 0.15

_TURKISH_DECISION_PATTERNS = [
    r"(?:şöyle|böyle)\s+(?:yapalım|yapacağız|yapıyoruz|gidiyoruz)",
    r"tamam\s+(?:öyle|böyle|şöyle)\s+(?:yapalım|olsun)",
    r"(?:bence|bana göre|benim fikrim)",
    r"(?:planımız|plan\s+şu|strateji)",
    r"(?:bu\s+kısım|bu\s+şekilde|şu\s+şekilde).*(?:olacak|olsun|yapacağız)",
    r"(?:devam\s+edelim|başlayalım|geçelim)",
    r"(?:öncelik|priority|sıra)",
]
_TURKISH_DECISION_BONUS = 0.20

# Entity stopwords — false positive capitals that inflate entity count
_ENTITY_STOPWORDS: set = {
    "System", "User", "Assistant", "WhatsApp", "Slack", "Session",
    "Current", "Return", "Monday", "Tuesday", "Wednesday",
    "Thursday", "Friday", "Saturday", "Sunday",
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
    "The", "This", "That", "Here", "There", "What", "When", "Where",
    "How", "Why", "Who", "Which", "None", "True", "False",
}


def _is_system_noise(text_lower: str) -> bool:
    """Detect gateway connects, test messages, cron boilerplate."""
    return any(p.search(text_lower) for p in _SYSTEM_NOISE_PATTERNS)


def score_importance(text: str, metadata: Optional[Dict] = None) -> float:
    """Calculate importance score for a message (0.0 – 1.0).

    Recalibrated 2026-03-21: raised base from 0.20 to 0.35, lowered search floor
    to 0.05 so ranking (not filtering) controls recall visibility.
    """
    score = 0.35
    text_lower = text.lower().strip()
    metadata = metadata or {}

    # --- early exits for noise ---
    for pattern in _NOISE_PATTERNS:
        if re.match(pattern, text_lower):
            return 0.05

    if _is_system_noise(text_lower):
        return 0.10

    # --- cron penalty: detect and cap early ---
    is_cron = any(p.search(text) for p in _CRON_NOISE_PATTERNS)
    source = metadata.get("source", "")
    if source == "cron":
        is_cron = True

    # --- conversation boost ---
    is_conversation = any(p.search(text) for p in _CONVERSATION_SOURCE_PATTERNS)
    if source in ("slack-dm", "whatsapp"):
        is_conversation = True
    if is_conversation:
        score += _CONVERSATION_BONUS

    if "?" in text and len(text) > 10:
        score += 0.05

    if any(m in text_lower for m in _IMPORTANCE_MARKERS):
        score += 0.25

    if any(m in text_lower for m in _DECISION_MARKERS):
        score += 0.20

    if any(re.search(p, text_lower) for p in _TURKISH_DECISION_PATTERNS):
        score += _TURKISH_DECISION_BONUS

    if any(m in text_lower for m in _TASK_MARKERS):
        score += 0.15

    if any(m in text_lower for m in _OPS_MARKERS):
        score += 0.20

    caps = set(re.findall(r"[A-ZÇĞİÖŞÜ][a-zçğıöşü]{2,}", text))
    caps -= _ENTITY_STOPWORDS
    score += min(0.10, len(caps) * 0.02)

    word_count = len(text.split())
    if word_count > 150:
        score += 0.08
    elif word_count > 80:
        score += 0.04
    elif word_count < 8:
        score -= 0.05

    role = metadata.get("role", "")
    if role == "user":
        score += 0.05
    elif role == "qa_pair":
        score += 0.08

    if is_cron:
        score = min(score, _CRON_NOISE_MAX)

    has_decision = (
        any(m in text_lower for m in _DECISION_MARKERS)
        or any(re.search(p, text_lower) for p in _TURKISH_DECISION_PATTERNS)
    )
    if has_decision and not is_cron:
        score = max(score, 0.70)

    return max(0.05, min(1.0, score))


# ---------------------------------------------------------------------------
# Confidence tiers
# ---------------------------------------------------------------------------

def get_confidence_tier(score: float) -> str:
    """Map an importance / confidence score to a tier label.

    * ``HIGH``   — score > 0.85
    * ``MEDIUM`` — 0.60 ≤ score ≤ 0.85
    * ``LOW``    — score < 0.60
    """
    if score > 0.85:
        return "HIGH"
    if score >= 0.60:
        return "MEDIUM"
    return "LOW"
