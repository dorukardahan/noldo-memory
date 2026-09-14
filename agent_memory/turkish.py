"""Turkish NLP utilities.

* Dependency-free lexical normalisation; caller-supplied morphology is optional
* **dateparser** — temporal expression parsing (Turkish + English)
* ASCII folding for Turkish special characters
* Turkish stopwords
* Text normalisation pipeline
"""

from __future__ import annotations

import logging
import re
import warnings
from datetime import datetime, timedelta
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Turkish stopwords
# ---------------------------------------------------------------------------

TURKISH_STOPWORDS: set[str] = {
    "ve", "bir", "bu", "da", "de", "ile", "için", "icin", "o", "çok", "cok",
    "gibi", "var", "yok", "ne", "ben", "sen", "biz", "siz", "onlar", "ama",
    "ki", "mi", "mu", "mı", "mü", "ya", "hem", "daha", "en", "her",
    "şu", "su", "şey", "sey", "olan", "olarak", "oldu", "olur", "olmuş",
    "olan", "bunu", "buna", "bunun", "şunu", "sunu", "onun", "ona",
    "benim", "senin", "bizim", "sizin", "onların", "ise", "kadar",
    "sonra", "önce", "once", "ayrıca", "ayrica", "fakat", "ise",
    "üzere", "uzere", "gibi", "göre", "gore", "bile", "ise",
    "ancak", "dolayı", "dolayi", "rağmen", "ragmen", "çünkü", "cunku",
    "eğer", "eger", "hatta", "üstelik", "ustelik",
    "the", "a", "an", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "can", "to", "of",
    "in", "for", "on", "with", "at", "by", "from", "it", "this",
    "that", "and", "or", "but", "if", "not", "so", "very",
}

# ---------------------------------------------------------------------------
# ASCII folding
# ---------------------------------------------------------------------------

_TR_FOLD_TABLE = str.maketrans({
    "ç": "c", "Ç": "C",
    "ğ": "g", "Ğ": "G",
    "ı": "i", "I": "I",  # Turkish dotless-ı → i
    "İ": "I",              # Turkish dotted-İ → I
    "ö": "o", "Ö": "O",
    "ş": "s", "Ş": "S",
    "ü": "u", "Ü": "U",
})


def ascii_fold(text: str) -> str:
    """Fold Turkish special characters to ASCII equivalents.

    >>> ascii_fold("çalışıyor")
    'calisiyor'
    """
    return text.translate(_TR_FOLD_TABLE)


# ---------------------------------------------------------------------------
# Lemmatisation
# ---------------------------------------------------------------------------

def lemmatize(text: str, *, analyzer=None) -> str:
    """Legacy morphology adapter; no bundled analyzer or model downloads.

    Without an explicit analyzer, returns the input unchanged and warns. This
    preserves the former dependency-unavailable fallback, not successful
    morphological analysis. Supply an object with ``lemmatize(token)`` returning
    ``[(token, [lemma, ...])]`` to retain that caller-owned integration.

    The legacy algorithm strips infinitive endings: its output can be a stem,
    not a linguistic lemma. The name is retained for source compatibility.
    """
    if analyzer is None:
        warnings.warn(
            "NoldoMem no longer supplies morphological analysis; lemmatize returns "
            "unchanged text. Use normalize_text for lexical normalization or pass "
            "a caller-owned analyzer explicitly.",
            FutureWarning, stacklevel=2,
        )
        return text

    tokens = re.findall(r"[\w']+", text, re.UNICODE)
    lemmas: list[str] = []
    for token in tokens:
        try:
            results = analyzer.lemmatize(token)
            if results and results[0] and len(results[0]) > 1:
                # results is list of (word, [lemma1, lemma2, ...])
                # e.g. ('hatırlıyorum', ['hatırlamak'])
                lemma = results[0][1][0] if results[0][1] else token
                # Strip Turkish infinitive suffixes to get the stem
                # hatırlamak → hatırla, sevmek → sev
                for suffix in ("mak", "mek"):
                    if lemma.lower().endswith(suffix) and len(lemma) > len(suffix) + 1:
                        lemma = lemma[: -len(suffix)]
                        break
                lemmas.append(lemma.lower())
            else:
                lemmas.append(token.lower())
        except Exception:
            lemmas.append(token.lower())
    return " ".join(lemmas)


def lemmatize_tokens(text: str, *, analyzer=None) -> List[str]:
    """Split the legacy adapter output; see :func:`lemmatize` migration notes."""
    return lemmatize(text, analyzer=analyzer).split()


# ---------------------------------------------------------------------------
# Temporal parsing
# ---------------------------------------------------------------------------

# Custom patterns that dateparser may not handle well
_CUSTOM_TEMPORAL: dict[str, callable] = {}


def _register_custom_temporal() -> None:
    """Register Turkish temporal expressions that need custom handling."""
    now_fn = datetime.now

    def _obur_gun() -> Tuple[datetime, datetime]:
        """öbür gün = day after tomorrow"""
        d = now_fn() + timedelta(days=2)
        return (d.replace(hour=0, minute=0, second=0),
                d.replace(hour=23, minute=59, second=59))

    def _bu_sabah() -> Tuple[datetime, datetime]:
        """bu sabah = this morning"""
        d = now_fn().replace(hour=6, minute=0, second=0)
        e = now_fn().replace(hour=12, minute=0, second=0)
        return (d, e)

    def _evvelsi_gun() -> Tuple[datetime, datetime]:
        """evvelsi gün = day before yesterday"""
        d = now_fn() - timedelta(days=2)
        return (d.replace(hour=0, minute=0, second=0),
                d.replace(hour=23, minute=59, second=59))

    def _dun_aksam() -> Tuple[datetime, datetime]:
        """dün akşam = yesterday evening"""
        d = (now_fn() - timedelta(days=1)).replace(hour=18, minute=0, second=0)
        e = (now_fn() - timedelta(days=1)).replace(hour=23, minute=59, second=59)
        return (d, e)

    _CUSTOM_TEMPORAL["öbür gün"] = _obur_gun
    _CUSTOM_TEMPORAL["obur gun"] = _obur_gun
    _CUSTOM_TEMPORAL["bu sabah"] = _bu_sabah
    _CUSTOM_TEMPORAL["evvelsi gün"] = _evvelsi_gun
    _CUSTOM_TEMPORAL["evvelsi gun"] = _evvelsi_gun
    _CUSTOM_TEMPORAL["dün akşam"] = _dun_aksam
    _CUSTOM_TEMPORAL["dun aksam"] = _dun_aksam


# Single-word patterns (dateparser fails on these inside longer sentences)
def _make_day_fn(offset):
    def fn():
        d = datetime.now() + timedelta(days=offset)
        return (d.replace(hour=0, minute=0, second=0), d.replace(hour=23, minute=59, second=59))
    return fn

_CUSTOM_TEMPORAL["dün"] = _make_day_fn(-1)
_CUSTOM_TEMPORAL["dun"] = _make_day_fn(-1)
_CUSTOM_TEMPORAL["bugün"] = _make_day_fn(0)
_CUSTOM_TEMPORAL["bugun"] = _make_day_fn(0)
_CUSTOM_TEMPORAL["yarın"] = _make_day_fn(1)
_CUSTOM_TEMPORAL["yarin"] = _make_day_fn(1)

_register_custom_temporal()


def parse_temporal(
    text: str,
) -> Optional[Tuple[datetime, datetime]]:
    """Parse a Turkish or English temporal expression into a (start, end) range.

    Returns ``None`` if no temporal expression is detected.

    >>> parse_temporal("geçen hafta")  # doctest: +SKIP
    (datetime(2026, 1, 26, ...), datetime(2026, 2, 1, ...))
    """
    text_lower = text.lower().strip()

    # 1. Check custom patterns first
    for pattern, fn in _CUSTOM_TEMPORAL.items():
        if re.search(r'(?:^|\s)' + re.escape(pattern) + r'(?:\s|$)', text_lower):
            return fn()

    # 2. dateparser
    try:
        import dateparser

        settings = {
            "PREFER_DATES_FROM": "past",
            "RELATIVE_BASE": datetime.now(),
        }

        # Try range-like expressions
        range_keywords = {
            "geçen hafta": 7, "gecen hafta": 7,
            "geçen ay": 30, "gecen ay": 30,
            "last week": 7, "last month": 30,
            "bu hafta": 7, "this week": 7,
            "bu ay": 30, "this month": 30,
        }
        for kw, days in range_keywords.items():
            if kw in text_lower:
                parsed = dateparser.parse(kw, languages=["tr", "en"], settings=settings)
                if parsed:
                    start = parsed.replace(hour=0, minute=0, second=0)
                    end = start + timedelta(days=days) - timedelta(seconds=1)
                    return (start, end)

        # Single-point parsing — only if text contains a recognisable temporal signal
        # (prevents dateparser from matching version numbers, random digits, etc.)
        _TEMPORAL_SIGNALS = re.compile(
            r'\b(dün|dun|bugün|bugun|yarın|yarin|gün|gun|hafta|ay|yıl|yil|'
            r'önce|once|geçen|gecen|sonra|'
            r'yesterday|today|tomorrow|ago|last|next|'
            r'monday|tuesday|wednesday|thursday|friday|saturday|sunday|'
            r'pazartesi|salı|sali|çarşamba|carsamba|perşembe|persembe|'
            r'cuma|cumartesi|pazar|'
            r'ocak|şubat|subat|mart|nisan|mayıs|mayis|haziran|'
            r'temmuz|ağustos|agustos|eylül|eylul|ekim|kasım|kasim|aralık|aralik|'
            r'january|february|march|april|may|june|july|august|'
            r'september|october|november|december)\b',
            re.IGNORECASE
        )
        if _TEMPORAL_SIGNALS.search(text_lower):
            parsed = dateparser.parse(text, languages=["tr", "en"], settings=settings)
            if parsed:
                start = parsed.replace(hour=0, minute=0, second=0)
                end = parsed.replace(hour=23, minute=59, second=59)
                return (start, end)

    except ImportError:
        logger.warning("dateparser not installed — temporal parsing disabled")
    except Exception as exc:
        logger.debug("dateparser failed for %r: %s", text, exc)

    return None


# ---------------------------------------------------------------------------
# Normalisation pipeline
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[\w']+", re.UNICODE)
_TR_LOWER_TABLE = str.maketrans({"I": "ı", "İ": "i"})


def normalize_text(text: str, use_lemma: bool = False, *, analyzer=None) -> str:
    """Lexical normalisation: lowercase, ASCII fold and remove stopwords.

    No stemming/lemmatisation is implied. Legacy ``use_lemma=True`` requires a
    caller-owned analyzer for morphology; otherwise it warns and passes through.
    """
    text = text.translate(_TR_LOWER_TABLE).lower()

    if use_lemma:
        text = lemmatize(text, analyzer=analyzer)

    # ASCII fold produces an extra copy for matching
    folded = ascii_fold(text)
    combined = f"{text} {folded}"

    # Remove stopwords
    tokens = _TOKEN_RE.findall(combined)
    filtered = [t for t in tokens if t not in TURKISH_STOPWORDS and len(t) > 1]
    return " ".join(dict.fromkeys(filtered))  # dedupe preserving order


def tokenize_for_search(text: str, *, analyzer=None) -> List[str]:
    """Tokenize and normalize for search queries (no dedup)."""
    text = text.translate(_TR_LOWER_TABLE).lower()
    if analyzer is not None:
        text = lemmatize(text, analyzer=analyzer)
    folded = ascii_fold(text)
    combined = f"{text} {folded}"
    tokens = _TOKEN_RE.findall(combined)
    return [t for t in tokens if t not in TURKISH_STOPWORDS and len(t) > 1]
