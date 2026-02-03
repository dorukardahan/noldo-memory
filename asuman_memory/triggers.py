"""Trigger patterns and importance scoring.

Ported from Mahmory v6 ``memory_skill.py`` / ``importance.py``, cleaned up
and adapted for Asuman's Turkish+English environment.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

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
    r"sevdiğim", r"sevmediğim",
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
    r"(?:mıştı|mişti|muştu|müştü|dık|dik|duk|dük|aldı|yaptı|gitti|geldi|söyledi"
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
# Importance scoring (0.0 – 1.0)
# ---------------------------------------------------------------------------

_IMPORTANCE_MARKERS: List[str] = [
    "hatırla", "hatirla", "unutma", "önemli", "onemli", "kritik",
    "acil", "kesinlikle", "mutlaka", "dikkat",
    "remember", "don't forget", "important", "critical",
    "urgent", "must", "definitely", "attention",
]

_DECISION_MARKERS: List[str] = [
    "karar", "kararlaştırdık", "yapacağım", "yapacağız",
    "söz", "tamamdır", "anlaştık", "kabul",
    "decided", "will do", "agreed", "commitment",
    "plan is", "going to", "promise", "deal",
]

_TASK_MARKERS: List[str] = [
    "todo", "yapılacak", "görev", "task",
    "action item", "next step",
]

_NOISE_PATTERNS: List[str] = [
    r"^(?:ok|okay|tamam|evet|yes|no|hayır|hayir|hmm|haha|lol)$",
    r"^(?:thanks|teşekkür|tesekkur|sağol|sagol)[\s!.]*$",
    r"^\d+$",
    r"^[\W]+$",
]


def score_importance(text: str, metadata: Optional[Dict] = None) -> float:
    """Calculate importance score for a message (0.0 – 1.0).

    Considers question marks, explicit importance markers, decisions,
    named entities, length, and role.
    """
    score = 0.5
    text_lower = text.lower().strip()

    # --- positive signals ---

    # Questions
    qcount = text.count("?")
    if qcount > 0:
        score += 0.1
    if qcount > 1:
        score += 0.05 * min(qcount - 1, 3)

    # Importance markers
    if any(m in text_lower for m in _IMPORTANCE_MARKERS):
        score += 0.25

    # Decision markers
    if any(m in text_lower for m in _DECISION_MARKERS):
        score += 0.20

    # Task markers
    if any(m in text_lower for m in _TASK_MARKERS):
        score += 0.15

    # Named entities (simple capital-word heuristic)
    caps = set(re.findall(r"\b[A-ZÇĞİÖŞÜ][a-zçğıöşü]+\b", text))
    score += min(0.15, len(caps) * 0.03)

    # Substantive length
    word_count = len(text.split())
    if word_count > 100:
        score += 0.10
    elif word_count > 50:
        score += 0.05
    elif word_count < 10:
        score -= 0.10

    # Role bonus
    if metadata:
        role = metadata.get("role", "")
        if role == "user":
            score += 0.10
        elif role == "qa_pair":
            score += 0.15

    # --- negative signals ---
    for pattern in _NOISE_PATTERNS:
        if re.match(pattern, text_lower):
            score -= 0.30
            break

    return max(0.0, min(1.0, score))


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
