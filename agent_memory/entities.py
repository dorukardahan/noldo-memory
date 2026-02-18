"""Knowledge-graph entity extraction.

Regex + heuristic NER (NO torch / spaCy / heavy ML):
* Turkish name patterns
* Phone / email extraction
* Known entity lists
* Context-based extraction ("X ile görüştüm" → person)
* Entity dedup (case-insensitive, alias matching)
* Relationship extraction from conversation context
* Temporal fact tracking with validity periods

Adapted from Mahmory's ``entity_extractor.py`` + ``knowledge_graph.py``.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

from .storage import MemoryStorage
from .conflict_detector import ConflictDetector

try:
    from rapidfuzz import fuzz, process as rfprocess
    HAS_RAPIDFUZZ = True
except ImportError:
    HAS_RAPIDFUZZ = False

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Entity:
    """A single extracted entity."""
    text: str
    label: str  # person, place, org, tech, product, date, concept
    confidence: float = 1.0
    source: str = ""
    timestamp: str = ""


@dataclass
class ExtractedEntities:
    """Container for entities from a piece of text."""
    people: List[Entity] = field(default_factory=list)
    places: List[Entity] = field(default_factory=list)
    organizations: List[Entity] = field(default_factory=list)
    tech_terms: List[Entity] = field(default_factory=list)
    products: List[Entity] = field(default_factory=list)
    dates: List[Entity] = field(default_factory=list)
    concepts: List[Entity] = field(default_factory=list)

    def all_entities(self) -> List[Entity]:
        return (
            self.people + self.places + self.organizations
            + self.tech_terms + self.products + self.dates + self.concepts
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            k: [{"text": e.text, "label": e.label, "confidence": e.confidence}
                 for e in getattr(self, k)]
            for k in ("people", "places", "organizations", "tech_terms",
                       "products", "dates", "concepts")
        }


# ---------------------------------------------------------------------------
# Known entity dictionaries (expandable)
# ---------------------------------------------------------------------------

KNOWN_PEOPLE: Set[str] = {
    # Add your known people here, e.g.: "alice", "bob"
}

KNOWN_PLACES: Set[str] = {
    "istanbul", "izmir", "ankara", "berlin", "singapore", "singapur",
    "türkiye", "turkey", "london", "new york",
}

KNOWN_ORGS: Set[str] = {
    "anthropic", "openai", "google", "apple", "microsoft", "meta",
    "openclaw",
}

KNOWN_TECH: Set[str] = {
    # Languages & runtimes
    "python", "javascript", "typescript", "node.js", "react", "vue", "bash",
    # AI/ML
    "claude", "gpt", "llm", "ai", "ml", "api", "sdk", "opus", "sonnet",
    "codex", "gemini", "kimi", "qwen", "openai", "anthropic",
    # Databases & storage
    "sqlite", "chromadb", "embedding", "vector", "rag", "postgresql",
    "postgres", "redis", "fts5", "sqlite-vec",
    # Infrastructure
    "docker", "kubernetes", "linux", "git", "github", "systemd", "nginx",
    "traefik", "cloudflare", "certbot", "ufw", "iptables", "pm2",
    # Frameworks & tools
    "whatsapp", "telegram", "discord", "fastapi", "uvicorn", "celery",
    "openclaw", "llama-server", "tmux", "ssh",
    # Project-specific
    "senti", "track", "myapp", "bureau", "project",
    "asuman", "x-accounts", "domain-search",
}  # expanded [S14, 2026-02-17]

# ---------------------------------------------------------------------------
# Entity aliases — informal names → canonical names
# ---------------------------------------------------------------------------

ENTITY_ALIASES: Dict[str, str] = {
    # Project aliases [S14, 2026-02-17]
    "oc": "OpenClaw",
    "wp": "WhatsApp",
    "tg": "Telegram",
    "vps": "VPS",
    "db": "SQLite",
    "pg": "PostgreSQL",
    "k8s": "Kubernetes",
    "cf": "Cloudflare",
    "gh": "GitHub",
}


def resolve_alias(name: str, threshold: int = 80) -> str:
    """Resolve an informal name to its canonical form.

    Uses exact alias lookup first, then rapidfuzz fuzzy matching.
    Returns the canonical name if found, otherwise the original name.
    """
    name_lower = name.lower().strip()

    # Exact alias match
    if name_lower in ENTITY_ALIASES:
        return ENTITY_ALIASES[name_lower]

    # Fuzzy match against alias keys
    if HAS_RAPIDFUZZ and len(name_lower) >= 3:
        alias_keys = list(ENTITY_ALIASES.keys())
        match = rfprocess.extractOne(
            name_lower, alias_keys, scorer=fuzz.ratio, score_cutoff=threshold
        )
        if match:
            return ENTITY_ALIASES[match[0]]

    return name

# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

# Turkish name pattern: capital letter start, at least 2 words
_NAME_RE = re.compile(
    r"\b([A-ZÇĞİÖŞÜ][a-zçğıöşü]+(?:\s+[A-ZÇĞİÖŞÜ][a-zçğıöşü]+)+)\b"
)

# Phone numbers (international)
_PHONE_RE = re.compile(
    r"(?:\+?\d{1,3}[\s-]?)?\(?\d{3}\)?[\s-]?\d{3}[\s-]?\d{2}[\s-]?\d{2}"
)

# Email
_EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")

# Date patterns
_DATE_PATTERNS = [
    re.compile(r"\d{4}-\d{2}-\d{2}"),
    re.compile(r"\d{1,2}/\d{1,2}/\d{4}"),
    re.compile(r"\d{1,2}\.\d{1,2}\.\d{4}"),
    re.compile(
        r"\d{1,2}\s+(?:ocak|şubat|mart|nisan|mayıs|haziran|temmuz|ağustos|eylül|ekim|kasım|aralık)\s+\d{4}",
        re.IGNORECASE,
    ),
    re.compile(
        r"\d{1,2}\s+(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{4}",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:bugün|dün|yarın|today|yesterday|tomorrow)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:geçen|bu|önümüzdeki)\s+(?:hafta|ay|yıl)",
        re.IGNORECASE,
    ),
]

# Context-based patterns (Turkish)
_PERSON_CONTEXT_RE = re.compile(
    r"(\w+)\s+(?:ile\s+görüştüm|ile\s+gorusdum|ile\s+konuştum|ile\s+konustum"
    r"|aradi|aradı|yazdı|yazdi|dedi|söyledi|soyledi)",
    re.IGNORECASE | re.UNICODE,
)
_PLACE_CONTEXT_RE = re.compile(
    r"(\w+)'?[eaıiuü]?\s+(?:gittim|geldim|gidiyorum|geliyorum|taşındım|tasindim|uçtum|uctum)",
    re.IGNORECASE | re.UNICODE,
)

# Product patterns
_PRODUCT_PATTERNS = [
    re.compile(r"iPhone\s*\d+\s*(?:Pro|Max|Plus)?", re.IGNORECASE),
    re.compile(r"MacBook\s*(?:Pro|Air)?", re.IGNORECASE),
    re.compile(r"Mac\s*Studio", re.IGNORECASE),
    re.compile(r"iPad\s*(?:Pro|Air|Mini)?", re.IGNORECASE),
]

# Typed relation patterns
# NOTE: Keep patterns conservative (false positives are costlier than misses).
# Capturing groups are expected as: (1)=subject, (2)=object/value.
_RELATION_PATTERNS = {
    "lives_in": [
        re.compile(
            r"\b([A-ZÇĞİÖŞÜ][\w.-]{1,40})\b\s+(?:şu\s+an\s+)?(?:[\w\s]{0,12})?\b([A-ZÇĞİÖŞÜ][\w\s.-]{1,60})\b(?:'?(?:de|da)|\s+(?:de|da))\s+(?:yaşıyor|oturuyor|ikamet\s+ediyor)",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b([A-ZÇĞİÖŞÜ][\w.-]{1,40})\b\s+(?:lives?|is\s+based)\s+in\s+\b([A-Z][\w\s.-]{1,60})\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b([A-ZÇĞİÖŞÜ][\w.-]{1,40})\b\s+(?:in\s+)?\b([A-ZÇĞİÖŞÜ][\w\s.-]{1,60})\b(?:'?(?:de|da)|\s+(?:de|da))\s+(?:takılıyor|takiliyo|kalıyor)",
            re.IGNORECASE,
        ),
    ],
    "works_at": [
        re.compile(
            r"\b([A-ZÇĞİÖŞÜ][\w.-]{1,40})\b\s+\b([A-Z0-9][\w&./\-\s]{2,70})\b(?:'?(?:de|da)|\s+(?:de|da))\s+çalışıyor",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b([A-ZÇĞİÖŞÜ][\w.-]{1,40})\b\s+works?\s+(?:at|for)\s+\b([A-Z0-9][\w&./\-\s]{2,70})\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b([A-ZÇĞİÖŞÜ][\w.-]{1,40})\b\s+\b([A-Z0-9][\w&./\-\s]{2,70})\b\s+(?:bünyesinde|ekibinde)\s+(?:çalışıyor|görev\s+yapıyor)",
            re.IGNORECASE,
        ),
    ],
    "status": [
        re.compile(
            r"\b([A-ZÇĞİÖŞÜ][\w.-]{1,40})\b\s+(?:durumu|statüsü|statusu)\s*(?:is|=|:)\s*(aktif|pasif|beklemede|tamamlandı|iptal|çözüldü|açık|kapalı)",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b([A-ZÇĞİÖŞÜ][\w.-]{1,40})\b\s+(?:status|state)\s*(?:is|=|:)\s*(active|inactive|pending|completed|cancelled|resolved|open|closed)",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b([A-ZÇĞİÖŞÜ][\w.-]{1,40})\b\s+(?:şu\s+an\s+)?(aktif|pasif|beklemede|açık|kapalı|online|offline)",
            re.IGNORECASE,
        ),
    ],
    "prefers": [
        re.compile(
            r"\b([A-ZÇĞİÖŞÜ][\w.-]{1,40})\b\s+\b([\w#+./\-\s]{2,60})\b\s+tercih\s+ediyor",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b([A-ZÇĞİÖŞÜ][\w.-]{1,40})\b\s+prefers?\s+\b([\w#+./\-\s]{2,60})\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b([A-ZÇĞİÖŞÜ][\w.-]{1,40})\b\s+için\s+tercih\s+(?:o|şu|bu)?\s*\b([\w#+./\-\s]{2,60})\b",
            re.IGNORECASE,
        ),
    ],
    "has_state": [
        re.compile(
            r"\b([A-ZÇĞİÖŞÜ][\w.-]{1,40})\b\s+(hasta|iyi|yorgun|müsait|meşgul|yoğun|online|offline)",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b([A-ZÇĞİÖŞÜ][\w.-]{1,40})\b\s+is\s+(sick|well|tired|available|busy|online|offline|overloaded)",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b([A-ZÇĞİÖŞÜ][\w.-]{1,40})\b\s+(?:şu\s+an\s+)?(müsait|meşgul|boşta|yoğun)",
            re.IGNORECASE,
        ),
    ],
    "created_by": [
        re.compile(
            r"\b([A-Z0-9][\w./\-\s]{2,70})\b\s+(?:,\s*)?(?:tarafından|tarafindan)\s+\b([A-ZÇĞİÖŞÜ][\w.-]{1,40})\b\s+(?:yaratıldı|oluşturuldu|geliştirildi)",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b([A-Z0-9][\w./\-\s]{2,70})\b\s+(?:was\s+)?(?:created|built|developed)\s+by\s+\b([A-Z][\w.-]{1,40})\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b([A-Z0-9][\w./\-\s]{2,70})\b\s+'?(?:ı|i|u|ü)?\s+\b([A-ZÇĞİÖŞÜ][\w.-]{1,40})\b\s+yarattı",
            re.IGNORECASE,
        ),
    ],
    "uses": [
        re.compile(
            r"\b([A-Z0-9][\w./\-\s]{2,70})\b\s+\b([A-Z0-9][\w#+./\-\s]{2,70})\b\s+(?:kullanıyor|kullaniyor)",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b([A-Z0-9][\w./\-\s]{2,70})\b\s+uses\s+\b([A-Z0-9][\w#+./\-\s]{2,70})\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b([A-Z0-9][\w./\-\s]{2,70})\b\s+\b([A-Z0-9][\w#+./\-\s]{2,70})\b\s+(?:ile|with)\s+(?:çalışıyor|run(?:ning)?|works?)",
            re.IGNORECASE,
        ),
    ],
    "deployed_on": [
        re.compile(
            r"\b([A-Z0-9][\w./\-\s]{2,70})\b\s+\b([A-Z0-9][\w./\-\s]{2,70})\b(?:'?(?:te|ta|de|da)|\s+(?:te|ta|de|da))\s+deploy\s+edildi",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b([A-Z0-9][\w./\-\s]{2,70})\b\s+(?:is\s+)?deployed\s+on\s+\b([A-Z0-9][\w./\-\s]{2,70})\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b([A-Z0-9][\w./\-\s]{2,70})\b\s+\b([A-Z0-9][\w./\-\s]{2,70})\b\s+üstünde\s+koşuyor",
            re.IGNORECASE,
        ),
    ],
    "depends_on": [
        re.compile(
            r"\b([A-Z0-9][\w./\-\s]{2,70})\b\s+\b([A-Z0-9][\w./\-\s]{2,70})\b(?:'?(?:e|a)|\s+(?:e|a))\s+bağlı",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b([A-Z0-9][\w./\-\s]{2,70})\b\s+depends\s+on\s+\b([A-Z0-9][\w./\-\s]{2,70})\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b([A-Z0-9][\w./\-\s]{2,70})\b\s+için\s+gerekli\s+olan\s+\b([A-Z0-9][\w./\-\s]{2,70})\b",
            re.IGNORECASE,
        ),
    ],
    "scheduled_at": [
        re.compile(
            r"\b([A-Z0-9][\w./\-\s]{2,70})\b\s+her\s+(\d+\s*(?:dakika|saat|gün|hafta))(?:te|ta|de|da)?\s+bir\s+(?:çalışıyor|çalıştırılıyor)",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b([A-Z0-9][\w./\-\s]{2,70})\b\s+(?:is\s+)?scheduled\s+(?:at|for|every)\s+(\d+\s*(?:min|mins|minutes?|hours?|days?|weeks?))",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b([A-Z0-9][\w./\-\s]{2,70})\b\s+(?:cron|schedule)\s*[:=]\s*(\S+)",
            re.IGNORECASE,
        ),
    ],
    "configured_as": [
        re.compile(
            r"\b([A-Z0-9][\w./\-\s]{2,70})\b\s+\b([\w./\-]{2,40})\b\s+olarak\s+konfigüre\s+edildi",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b([A-Z0-9][\w./\-\s]{2,70})\b\s+(?:is\s+)?configured\s+as\s+\b([\w./\-]{2,40})\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b([A-Z0-9][\w./\-\s]{2,70})\b\s+(?:mode|rol)\s*(?:is|=|:)\s*\b([\w./\-]{2,40})\b",
            re.IGNORECASE,
        ),
    ],
    "member_of": [
        re.compile(
            r"\b([A-ZÇĞİÖŞÜ][\w.-]{1,40})\b\s+\b([A-Z0-9][\w&./\-\s]{2,70})\b(?:'?(?:nin|nın|nun|nün)|\s+(?:nin|nın|nun|nün))\s+üyesi",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b([A-ZÇĞİÖŞÜ][\w.-]{1,40})\b\s+is\s+(?:a\s+)?member\s+of\s+\b([A-Z0-9][\w&./\-\s]{2,70})\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b([A-ZÇĞİÖŞÜ][\w.-]{1,40})\b\s+\b([A-Z0-9][\w&./\-\s]{2,70})\b\s+ekibinde",
            re.IGNORECASE,
        ),
    ],
    "version_of": [
        re.compile(
            r"\b([A-Z0-9][\w./\-]{1,40}\s+v?\d+(?:\.\d+){0,3})\b\s*,?\s*\b([A-Z0-9][\w./\-]{1,40})\b(?:'?(?:ın|in|un|ün)|\s+(?:ın|in|un|ün))\s+versiyonu",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b([A-Z0-9][\w./\-]{1,40}\s+v?\d+(?:\.\d+){0,3})\b\s+is\s+(?:a\s+)?version\s+of\s+\b([A-Z0-9][\w./\-]{1,40})\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b([A-Z0-9][\w./\-]{1,40})\b\s+v(\d+(?:\.\d+){0,3})\b",
            re.IGNORECASE,
        ),
    ],
}

# Anti-patterns to suppress obvious false positives per relation type.
_RELATION_ANTI_PATTERNS = {
    "works_at": [
        re.compile(r"\b(uyuyor|dinleniyor|oturuyor)\b", re.IGNORECASE),
    ],
    "uses": [
        re.compile(r"\b(kullanmıyor|kullanmiyor|does\s+not\s+use|don't\s+use)\b", re.IGNORECASE),
    ],
    "depends_on": [
        re.compile(r"\b(bağlı\s+değil|bagli\s+degil|independent\s+of)\b", re.IGNORECASE),
    ],
    "scheduled_at": [
        re.compile(r"\b(schedule\s+yok|zamanlama\s+yok|not\s+scheduled)\b", re.IGNORECASE),
    ],
    "version_of": [
        re.compile(r"\b(v(?:ery)?\s+good)\b", re.IGNORECASE),
    ],
}


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------

class EntityExtractor:
    """Extract named entities from text using regex + heuristics."""

    def extract(
        self,
        text: str,
        source: str = "",
        timestamp: str = "",
    ) -> ExtractedEntities:
        """Extract all entity types from *text*."""
        result = ExtractedEntities()
        text_lower = text.lower()

        # Known entities
        for person in KNOWN_PEOPLE:
            if person in text_lower:
                result.people.append(
                    Entity(person.title(), "person", 1.0, source, timestamp)
                )

        for place in KNOWN_PLACES:
            if place in text_lower:
                result.places.append(
                    Entity(place.title(), "place", 1.0, source, timestamp)
                )

        for org in KNOWN_ORGS:
            if org in text_lower:
                result.organizations.append(
                    Entity(org.title(), "org", 1.0, source, timestamp)
                )

        for tech in KNOWN_TECH:
            if re.search(r"\b" + re.escape(tech) + r"\b", text_lower):
                result.tech_terms.append(
                    Entity(tech, "tech", 0.9, source, timestamp)
                )

        # Name pattern (Turkish names)
        for match in _NAME_RE.finditer(text):
            name = match.group(1)
            if name.lower() not in {e.text.lower() for e in result.people}:
                result.people.append(
                    Entity(name, "person", 0.7, source, timestamp)
                )

        # Context-based person extraction
        for match in _PERSON_CONTEXT_RE.finditer(text):
            name = match.group(1).strip()
            if (
                len(name) > 2
                and name.lower() not in {e.text.lower() for e in result.people}
                and name[0].isupper()
            ):
                result.people.append(
                    Entity(name, "person", 0.6, source, timestamp)
                )

        # Context-based place extraction
        for match in _PLACE_CONTEXT_RE.finditer(text):
            place = match.group(1).strip()
            if (
                len(place) > 2
                and place.lower() not in {e.text.lower() for e in result.places}
                and place[0].isupper()
            ):
                result.places.append(
                    Entity(place, "place", 0.6, source, timestamp)
                )

        # Phone numbers
        for match in _PHONE_RE.finditer(text):
            # Store as concept since it's contact info
            result.concepts.append(
                Entity(match.group(), "phone", 0.9, source, timestamp)
            )

        # Emails
        for match in _EMAIL_RE.finditer(text):
            result.concepts.append(
                Entity(match.group(), "email", 0.9, source, timestamp)
            )

        # Date patterns
        for pattern in _DATE_PATTERNS:
            for match in pattern.finditer(text):
                if match.group().lower() not in {e.text.lower() for e in result.dates}:
                    result.dates.append(
                        Entity(match.group(), "date", 0.8, source, timestamp)
                    )

        # Products
        for pattern in _PRODUCT_PATTERNS:
            for match in pattern.finditer(text):
                result.products.append(
                    Entity(match.group(), "product", 0.9, source, timestamp)
                )

        # Deduplicate
        result.people = _dedupe(result.people)
        result.places = _dedupe(result.places)
        result.organizations = _dedupe(result.organizations)
        result.tech_terms = _dedupe(result.tech_terms)
        result.products = _dedupe(result.products)
        result.dates = _dedupe(result.dates)
        result.concepts = _dedupe(result.concepts)

        return result

    def extract_typed_relations(
        self, text: str, entities: ExtractedEntities
    ) -> List[Dict[str, Any]]:
        """Extract structured relations from text using typed patterns."""
        relations = []
        text_norm = text.replace("\n", " ")

        # Create lookup map for entities to normalize names
        # Map lowercased original text -> canonical entity object
        entity_map = {}
        for e in entities.all_entities():
            entity_map[e.text.lower()] = e

        for rel_type, patterns in _RELATION_PATTERNS.items():
            for pattern in patterns:
                for match in pattern.finditer(text_norm):
                    # Groups: 1=Subject, 2=Object (usually)
                    # Some patterns like status/has_state might just be subject+state
                    if len(match.groups()) < 2:
                        continue

                    subject_raw = match.group(1).strip()
                    object_raw = match.group(2).strip()

                    # Try to map subject to an extracted entity
                    subject_ent = entity_map.get(subject_raw.lower())
                    subject_name = subject_ent.text if subject_ent else resolve_alias(subject_raw)

                    # For object, it might be an entity or just a string value
                    # If it maps to an entity, use canonical name
                    object_ent = entity_map.get(object_raw.lower())
                    object_val = object_ent.text if object_ent else object_raw

                    relations.append({
                        "relation_type": rel_type,
                        "subject": subject_name,
                        "object": object_val,
                        "confidence": 0.8,  # High confidence for regex match
                        "text_span": match.group(0)
                    })

        return relations



def _dedupe(entities: List[Entity]) -> List[Entity]:
    """Remove duplicates (case-insensitive)."""
    seen: set[str] = set()
    unique: list[Entity] = []
    for e in entities:
        key = e.text.lower()
        if key not in seen:
            seen.add(key)
            unique.append(e)
    return unique


# ---------------------------------------------------------------------------
# Knowledge graph operations
# ---------------------------------------------------------------------------

class KnowledgeGraph:
    """High-level API for entity storage + relationship management."""

    def __init__(self, storage: MemoryStorage) -> None:
        self.storage = storage
        self.extractor = EntityExtractor()

    def process_text(
        self,
        text: str,
        source: str = "",
        timestamp: str = "",
    ) -> ExtractedEntities:
        """Extract entities and persist to storage + create co-occurrence links."""
        entities = self.extractor.extract(text, source, timestamp)
        all_ents = entities.all_entities()

        # Store each entity (resolve aliases to canonical names)
        entity_ids: list[str] = []
        for ent in all_ents:
            canonical_name = resolve_alias(ent.text)
            eid = self.storage.store_entity(
                name=canonical_name,
                entity_type=ent.label,
            )
            entity_ids.append(eid)

        # Co-occurrence relationships — use typed relation when both are tech [S14, 2026-02-17]
        if len(entity_ids) > 1:
            for i in range(len(entity_ids)):
                for j in range(i + 1, len(entity_ids)):
                    # Determine relation type based on entity types
                    ent_i = all_ents[i] if i < len(all_ents) else None
                    ent_j = all_ents[j] if j < len(all_ents) else None
                    rel_type = "mentioned_with"
                    confidence = 0.5
                    if ent_i and ent_j:
                        both_tech = ent_i.label == "tech" and ent_j.label == "tech"
                        if both_tech:
                            rel_type = "uses"
                            confidence = 0.6
                    self.storage.link_entities(
                        entity_ids[i],
                        entity_ids[j],
                        relation_type=rel_type,
                        confidence=confidence,
                        context=text[:200],
                    )

        # Typed relations + Conflict Detection
        typed_rels = self.extractor.extract_typed_relations(text, entities)
        detector = ConflictDetector(self.storage)

        for rel in typed_rels:
            # Map subject name to entity ID
            subj_name = rel["subject"]
            subj_eid = None

            # Find subject entity ID
            # 1. Check extracted entities first
            for e_obj, e_id in zip(all_ents, entity_ids):
                if resolve_alias(e_obj.text).lower() == subj_name.lower():
                    subj_eid = e_id
                    break

            # 2. Or search storage if not in current extraction (less likely but possible via regex)
            if not subj_eid:
                matches = self.storage.search_entities(subj_name, limit=1)
                if matches:
                    subj_eid = matches[0]["id"]

            if not subj_eid:
                # Implicit entity creation if not found?
                # For now, skip if subject entity not resolved
                continue

            # Store typed relationship link
            # Only if object is also an entity (which regex tries to determine)
            # But regex returns object string. If it matches an entity, link it.
            obj_val = rel["object"]
            obj_eid = None

            # Check if object value maps to an extracted entity
            for e_obj, e_id in zip(all_ents, entity_ids):
                if resolve_alias(e_obj.text).lower() == obj_val.lower():
                    obj_eid = e_id
                    break

            if obj_eid:
                self.storage.link_entities(
                    subj_eid,
                    obj_eid,
                    relation_type=rel["relation_type"],
                    confidence=rel["confidence"],
                    context=rel["text_span"]
                )

            # Detect conflicts and store temporal fact
            detector.check_and_store(
                entity_id=subj_eid,
                relation_type=rel["relation_type"],
                object_value=obj_val,
                confidence=rel["confidence"],
                valid_from=None,  # defaults to now
                source_memory_id=None # passed if available (todo: pass from caller?)
            )

        return entities

    def search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search entities by name, with alias resolution."""
        canonical = resolve_alias(query)
        results = self.storage.search_entities(canonical, limit)
        # If alias resolved to different name and no results, try original
        if not results and canonical != query:
            results = self.storage.search_entities(query, limit)
        return results
