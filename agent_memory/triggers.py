"""Trigger patterns and importance scoring.

Ported from Mahmory v6 ``memory_skill.py`` / ``importance.py``, cleaned up
and adapted for Turkish+English environment.
"""

from __future__ import annotations

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
    r"^(?:ok|tamam|evet|hayır|hayir|anladım|anladim|👍|😂|😊|🙏)$",
    r"^(?:merhaba|selam|hey|hi|hello|nasılsın|nasilsin|naber)[\s?!]*$",
    r"^(?:teşekkür|tesekkur|sağol|sagol|thanks|thx)[\s!]*$",
    r"^(?:yap|oluştur|olustur|gönder|gonder|aç|ac|kapat|başla|basla|bitir)[\s!]*$",
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
    re.compile(r"^GatewayRestart:", re.IGNORECASE),
    re.compile(r"^\[queued messages", re.IGNORECASE),
    re.compile(r"^say\s+(?:ok|hello|hi|test|something)\s*$", re.IGNORECASE),
    re.compile(r"^Conversation info \(untrusted metadata\)", re.IGNORECASE),
    re.compile(r"^Replied message \(untrusted", re.IGNORECASE),
]

# Entity stopwords — false positive capitals that inflate entity count
_ENTITY_STOPWORDS: set = {
    "System", "User", "Assistant", "WhatsApp", "Session",
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

    Recalibrated 2026-02-17: base lowered to 0.20, system noise floors at 0.10,
    operational event markers added, entity stopwords filter, reduced role bonuses.
    Target distribution: 15% at 0.0-0.2, 35% at 0.2-0.4, 25% at 0.4-0.6,
    15% at 0.6-0.8, 10% at 0.8-1.0.
    """
    score = 0.20  # was 0.50
    text_lower = text.lower().strip()

    # --- early exits for noise ---
    for pattern in _NOISE_PATTERNS:
        if re.match(pattern, text_lower):
            return 0.05  # noise floor

    if _is_system_noise(text_lower):
        return 0.10  # system noise floor

    # --- positive signals ---

    # Questions (reduced from +0.10 to +0.05)
    if "?" in text and len(text) > 10:
        score += 0.05

    # Importance markers (keep +0.25 — explicit signals)
    if any(m in text_lower for m in _IMPORTANCE_MARKERS):
        score += 0.25

    # Decision markers (keep +0.20)
    if any(m in text_lower for m in _DECISION_MARKERS):
        score += 0.20

    # Task markers (+0.15)
    if any(m in text_lower for m in _TASK_MARKERS):
        score += 0.15

    # Operational event markers (NEW)
    if any(m in text_lower for m in _OPS_MARKERS):
        score += 0.20

    # Named entities — with stopword filter, min 3 chars
    caps = set(re.findall(r"\b[A-ZÇĞİÖŞÜ][a-zçğıöşü]{2,}\b", text))
    caps -= _ENTITY_STOPWORDS
    score += min(0.10, len(caps) * 0.02)  # was 0.15/0.03

    # Substantive length (raised thresholds)
    word_count = len(text.split())
    if word_count > 150:
        score += 0.08
    elif word_count > 80:
        score += 0.04
    elif word_count < 8:
        score -= 0.05

    # Role bonus — REDUCED
    if metadata:
        role = metadata.get("role", "")
        if role == "user":
            score += 0.05    # was 0.10
        elif role == "qa_pair":
            score += 0.08    # was 0.15

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
