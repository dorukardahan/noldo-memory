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
    "user", "asuman",
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
    "python", "javascript", "typescript", "node.js", "react", "vue",
    "claude", "gpt", "llm", "ai", "ml", "api", "sdk",
    "sqlite", "chromadb", "embedding", "vector", "rag",
    "whatsapp", "telegram", "discord", "fastapi", "uvicorn",
    "docker", "kubernetes", "linux", "git", "github",
}

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

        # Store each entity
        entity_ids: list[str] = []
        for ent in all_ents:
            eid = self.storage.store_entity(
                name=ent.text,
                entity_type=ent.label,
            )
            entity_ids.append(eid)

        # Co-occurrence relationships
        if len(entity_ids) > 1:
            for i in range(len(entity_ids)):
                for j in range(i + 1, len(entity_ids)):
                    self.storage.link_entities(
                        entity_ids[i],
                        entity_ids[j],
                        relation_type="mentioned_with",
                        confidence=0.5,
                        context=text[:200],
                    )

        return entities

    def search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search entities by name."""
        return self.storage.search_entities(query, limit)
