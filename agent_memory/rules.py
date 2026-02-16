"""Instruction/rule detection for automatic memory capture.

Detects concise directive-style messages (Turkish + English) and safewords.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, Pattern


_WORD_RE = re.compile(r"\S+")

# Safeword markers for explicit rule capture
_SAFEWORD_RE: Pattern[str] = re.compile(
    r"(?:📌|💾|(?:^|\s)/rule(?:\s|$)|(?:^|\s)/save(?:\s|$)|(?:^|\s)/remember(?:\s|$))",
    re.IGNORECASE | re.UNICODE,
)

# Narrative anti-patterns (non-directive "always" statements)
_ANTIPATTERNS: tuple[Pattern[str], ...] = (
    re.compile(r"^\s*i\s+always\s+(?:go|went|walk|work|eat|drink|sleep|watch|play|study)\b", re.IGNORECASE),
    re.compile(r"^\s*ben\s+her\s+zaman\s+(?:giderim|gidiyorum|yaparım|yapiyorum|yerim|içerim|icerim|uyurum)\b", re.IGNORECASE),
)

# Turkish directive patterns (4+)
_TR_RULE_PATTERNS: tuple[Pattern[str], ...] = (
    re.compile(
        r"""
        ^\s*
        (?:artık|artik|bundan\s+sonra|bundan\s+böyle|bundan\s+boyle)
        \s+
        .*(?:
            \b(?:ver|yaz|konuş|konus|kullan|ol|davran|yanıtla|yanitla|cevapla)\b
            |
            \b\w+(?:in|ın|un|ün|yin|yın|yun|yün)\b
        )
        """,
        re.IGNORECASE | re.VERBOSE,
    ),
    re.compile(
        r"""
        \b(?:her\s+zaman|daima|asla|hiçbir\s+zaman|hicbir\s+zaman)\b
        \s+.*
        \b(?:
            kullan|yaz|konuş|konus|cevap|yanıt|yanit|ver|ol|yap|söyle|soyle|belirt|ekle|gönder|gonder|tut|kes|bırak|birak|at|göster|goster
        )\w*\b
        """,
        re.IGNORECASE | re.VERBOSE,
    ),
    re.compile(
        r"""
        \b(?:cevapları|cevaplari|yanıtları|yanitlari)\b
        \s+.*\s+
        \b(?:formatında|formatinda|şeklinde|seklinde)\b
        """,
        re.IGNORECASE | re.VERBOSE,
    ),
    re.compile(
        r"""
        \b(?:benim\s+için|benim\s+icin|bana\s+göre|bana\s+gore)\b
        \s+.*\s+
        \b(?:önemli|onemli|tercihimdir)\b
        """,
        re.IGNORECASE | re.VERBOSE,
    ),
)

# English directive patterns (4+)
_EN_RULE_PATTERNS: tuple[Pattern[str], ...] = (
    re.compile(r"^\s*(?:always|never|from\s+now\s+on)\s+(?:use|speak|write|reply|respond|act|be|include|add|show|give|provide|keep|make|send|format)\b", re.IGNORECASE),
    re.compile(r"^\s*(?:do\s+not|don't|stop)\s+(?:using|doing|saying)\b", re.IGNORECASE),
    re.compile(r"^\s*i\s+(?:prefer|want|need)\s+(?:you\s+to|the\s+answers\s+to)\b", re.IGNORECASE),
    re.compile(r"\b(?:output|response)\s+(?:format|style)\s*:", re.IGNORECASE),
)


@dataclass
class DetectedInstruction:
    """Normalized instruction detection result."""

    text: str
    category: str
    confidence: float
    original_text: str


class RuleDetector:
    """Detect concise user instructions/rules from free-form text.

    Singleton-like construction is used so compiled regexes are reused.
    """

    _instance: Optional["RuleDetector"] = None

    def __new__(cls) -> "RuleDetector":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def check_safeword(self, text: str) -> bool:
        """Return True if message contains explicit rule safeword markers."""
        if not text:
            return False
        return bool(_SAFEWORD_RE.search(text))

    def detect(self, text: str) -> Optional[DetectedInstruction]:
        """Detect whether text is a concise directive instruction.

        Returns:
            DetectedInstruction when a pattern matches; otherwise ``None``.
        """
        if not text:
            return None

        raw = text.strip()
        if not raw:
            return None

        if len(_WORD_RE.findall(raw)) > 50:
            return None

        for anti in _ANTIPATTERNS:
            if anti.search(raw):
                return None

        for pattern in _TR_RULE_PATTERNS:
            if pattern.search(raw):
                return DetectedInstruction(
                    text=raw,
                    category="rule",
                    confidence=0.92,
                    original_text=text,
                )

        for pattern in _EN_RULE_PATTERNS:
            if pattern.search(raw):
                return DetectedInstruction(
                    text=raw,
                    category="rule",
                    confidence=0.90,
                    original_text=text,
                )

        return None
