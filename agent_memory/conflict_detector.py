"""Conflict detection for temporal facts in the knowledge graph.

Handles exclusive vs non-exclusive relations, auto-resolution based on confidence,
and flagging of conflicts for human review.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional, Dict, List, Any

from .storage import MemoryStorage

logger = logging.getLogger(__name__)

EXCLUSIVE_RELATIONS = {"lives_in", "works_at", "located_in", "status", "has_state"}
NON_EXCLUSIVE_RELATIONS = {"prefers", "decided_to", "mentioned_with"}
AUTO_RESOLVE_CONFIDENCE_MARGIN = 0.20


@dataclass
class Conflict:
    entity_id: str
    relation_type: str
    existing_value: str
    new_value: str
    existing_confidence: float
    new_confidence: float
    resolution: str  # "auto_new_wins", "flagged", "no_conflict"


class ConflictDetector:
    def __init__(self, storage: MemoryStorage):
        self.storage = storage

    def check_and_store(
        self,
        entity_id: str,
        relation_type: str,
        object_value: str,
        confidence: float = 0.7,
        valid_from: Optional[float] = None,
        source_memory_id: Optional[str] = None,
    ) -> Conflict:
        """Check for conflicts, store temporal fact, return conflict info."""
        now = time.time()
        valid_from = valid_from or now

        # Default conflict result (no conflict)
        result = Conflict(
            entity_id=entity_id,
            relation_type=relation_type,
            existing_value="",
            new_value=object_value,
            existing_confidence=0.0,
            new_confidence=confidence,
            resolution="no_conflict",
        )

        # 1. Non-exclusive: just store
        if relation_type not in EXCLUSIVE_RELATIONS:
            self.storage.store_typed_fact(
                entity_id=entity_id,
                relation_type=relation_type,
                object_value=object_value,
                confidence=confidence,
                valid_from=valid_from,
                source_memory_id=source_memory_id,
            )
            return result

        # 2. Exclusive: Check active facts
        # We need a method in storage to get active facts for (entity, relation)
        active_facts = self.storage.get_active_facts(entity_id, relation_type)
        conflict_found = False

        for fact in active_facts:
            existing_val = fact.get("object_value", "")
            existing_conf = fact.get("confidence", 0.5)
            fact_id = fact["id"]

            # Check if value is different (case-insensitive)
            if existing_val.lower() != object_value.lower():
                conflict_found = True
                
                # Conflict logic
                if confidence > (existing_conf + AUTO_RESOLVE_CONFIDENCE_MARGIN):
                    # Auto-resolve: Deactivate old
                    self.storage.deactivate_fact(fact_id, reason="superseded_auto")
                    resolution = "auto_new_wins"
                else:
                    # Flag for review
                    resolution = "flagged"

                result = Conflict(
                    entity_id=entity_id,
                    relation_type=relation_type,
                    existing_value=existing_val,
                    new_value=object_value,
                    existing_confidence=existing_conf,
                    new_confidence=confidence,
                    resolution=resolution,
                )
                
                # If auto-resolved, we can stop checking (assuming one active fact usually)
                if resolution == "auto_new_wins":
                    break
            else:
                # Same value - reinforcement logic could go here
                pass

        # 3. Store the new fact
        # Note: If flagged, both will be active until resolved manually
        self.storage.store_typed_fact(
            entity_id=entity_id,
            relation_type=relation_type,
            object_value=object_value,
            confidence=confidence,
            valid_from=valid_from,
            source_memory_id=source_memory_id,
        )

        return result
