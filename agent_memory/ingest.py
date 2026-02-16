"""Session ingestion module.

Parses OpenClaw JSONL session files, chunks user→assistant pairs,
deduplicates via MD5, and batch-embeds via OpenRouter.

Adapted from Mahmory's ``realtime_ingest.py`` / ``ingest_sessions.py``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from .config import Config, load_config
from .embeddings import OpenRouterEmbeddings
from .entities import KnowledgeGraph
from .rules import RuleDetector
from .storage import MemoryStorage
from .triggers import score_importance, should_trigger

logger = logging.getLogger(__name__)
rule_detector = RuleDetector()

# Messages to skip
_SKIP_CONTENT: set[str] = {
    "HEARTBEAT_OK",
    "NO_REPLY",
}

_SKIP_PREFIXES: tuple[str, ...] = (
    "HEARTBEAT_OK",
    "NO_REPLY",
)


def _md5(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:16]


def _extract_text(content: Any) -> str:
    """Extract plain text from an OpenClaw message content field.

    Content can be a string or a list of {type, text} dicts.
    """
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
        return "\n".join(parts).strip()
    return ""


def _is_tool_call(entry: Dict[str, Any]) -> bool:
    """Return True if the JSONL entry is a tool call / tool result."""
    msg = entry.get("message", {})
    role = msg.get("role", "")
    if role in ("tool", "function"):
        return True
    # Check for tool_calls in content
    content = msg.get("content", [])
    if isinstance(content, list):
        for item in content:
            if isinstance(item, dict) and item.get("type") in ("tool_use", "tool_result"):
                return True
    return False


def _should_skip(text: str, role: str) -> bool:
    """Determine if a message should be skipped for ingestion."""
    if not text or len(text.strip()) < 3:
        return True
    stripped = text.strip()
    if stripped in _SKIP_CONTENT:
        return True
    for prefix in _SKIP_PREFIXES:
        if stripped.startswith(prefix):
            return True
    if role == "system":
        return True
    return False


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    """A conversation chunk (typically a Q&A pair)."""
    text: str
    timestamp: str
    session_id: str
    role: str  # 'qa_pair' | 'user' | 'assistant'
    md5: str = ""

    def __post_init__(self) -> None:
        self.md5 = _md5(self.text)




def _chunk_session(
    entries: List[Dict[str, Any]],
    session_id: str,
    gap_hours: float = 4.0,
) -> List[Chunk]:
    """Group session entries into Q&A pair chunks.

    * Pairs consecutive user→assistant messages.
    * Splits on time gaps > *gap_hours*.
    * Filters out tool calls, heartbeats, etc.
    """
    chunks: list[Chunk] = []
    pending_user: Optional[str] = None
    pending_ts: Optional[str] = None
    last_ts: Optional[float] = None

    for entry in entries:
        if entry.get("type") != "message":
            continue
        if _is_tool_call(entry):
            continue

        msg = entry.get("message", {})
        role = msg.get("role", "")
        content = msg.get("content", "")
        text = _extract_text(content)
        timestamp = entry.get("timestamp", "")

        if _should_skip(text, role):
            continue

        # Time-gap splitting
        try:
            ts_epoch = datetime.fromisoformat(
                timestamp.replace("Z", "+00:00")
            ).timestamp()
        except Exception:
            ts_epoch = time.time()

        if last_ts and (ts_epoch - last_ts) > gap_hours * 3600:
            # Flush pending user message
            if pending_user:
                chunks.append(Chunk(
                    text=pending_user[:2000],
                    timestamp=pending_ts or timestamp,
                    session_id=session_id,
                    role="user",
                ))
                pending_user = None
                pending_ts = None

        last_ts = ts_epoch

        if role == "user":
            # If we had a pending user without assistant reply, flush it
            if pending_user:
                chunks.append(Chunk(
                    text=pending_user[:2000],
                    timestamp=pending_ts or timestamp,
                    session_id=session_id,
                    role="user",
                ))
            pending_user = text[:2000]
            pending_ts = timestamp

        elif role == "assistant":
            if pending_user:
                # Form Q&A pair
                qa_text = f"User: {pending_user}\nAssistant: {text[:1500]}"
                chunks.append(Chunk(
                    text=qa_text[:2000],
                    timestamp=pending_ts or timestamp,
                    session_id=session_id,
                    role="qa_pair",
                ))
                pending_user = None
                pending_ts = None
            else:
                # Standalone assistant message
                if len(text) > 20:
                    chunks.append(Chunk(
                        text=text[:2000],
                        timestamp=timestamp,
                        session_id=session_id,
                        role="assistant",
                    ))

    # Flush any remaining pending user message
    if pending_user:
        chunks.append(Chunk(
            text=pending_user[:2000],
            timestamp=pending_ts or "",
            session_id=session_id,
            role="user",
        ))

    return chunks


# ---------------------------------------------------------------------------
# Session parser
# ---------------------------------------------------------------------------

def parse_session_file(path: Path, gap_hours: float = 4.0) -> List[Chunk]:
    """Parse a single JSONL session file into chunks."""
    entries: list[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    return _chunk_session(entries, session_id=path.stem, gap_hours=gap_hours)


def discover_sessions(sessions_dir: Optional[str] = None) -> List[Path]:
    """Return all .jsonl session files sorted by modification time."""
    cfg = load_config()
    sdir = Path(sessions_dir or cfg.sessions_dir)
    if not sdir.is_dir():
        logger.warning("Sessions directory not found: %s", sdir)
        return []
    return sorted(sdir.glob("*.jsonl"), key=lambda p: p.stat().st_mtime)


# ---------------------------------------------------------------------------
# Batch ingest
# ---------------------------------------------------------------------------

async def ingest_sessions(
    storage: MemoryStorage,
    embedder: OpenRouterEmbeddings,
    sessions_dir: Optional[str] = None,
    gap_hours: float = 4.0,
    batch_size: int = 50,
    progress_cb: Optional[Callable[[int, int], None]] = None,
    knowledge_graph: Optional[KnowledgeGraph] = None,
) -> Dict[str, Any]:
    """Ingest all session files into storage.

    * Parses JSONL sessions
    * Chunks into Q&A pairs
    * Deduplicates via MD5
    * Batch-embeds via OpenRouter
    * Stores in sqlite-vec + FTS5
    * Optionally extracts entities into knowledge graph

    Returns stats dict.
    """
    sessions = discover_sessions(sessions_dir)
    if not sessions:
        return {"sessions": 0, "chunks": 0, "stored": 0, "skipped_dup": 0}

    all_chunks: list[Chunk] = []
    for session_path in sessions:
        chunks = parse_session_file(session_path, gap_hours=gap_hours)
        all_chunks.extend(chunks)

    logger.info("Parsed %d chunks from %d sessions", len(all_chunks), len(sessions))

    # Deduplicate
    seen_md5: set[str] = set()
    unique_chunks: list[Chunk] = []
    for chunk in all_chunks:
        if chunk.md5 not in seen_md5:
            seen_md5.add(chunk.md5)
            unique_chunks.append(chunk)
    skipped_dup = len(all_chunks) - len(unique_chunks)

    # Also check against existing memories
    existing_chunks = []
    for chunk in unique_chunks:
        existing = storage.get_memory(chunk.md5)
        if existing is None:
            existing_chunks.append(chunk)
    skipped_existing = len(unique_chunks) - len(existing_chunks)
    unique_chunks = existing_chunks

    logger.info(
        "After dedup: %d chunks (%d dups, %d already stored)",
        len(unique_chunks), skipped_dup, skipped_existing,
    )

    stored = 0
    total = len(unique_chunks)

    # Process in batches
    for batch_start in range(0, total, batch_size):
        batch = unique_chunks[batch_start : batch_start + batch_size]
        texts = [c.text for c in batch]

        # Batch embed
        try:
            vectors = await embedder.embed_batch(texts)
        except Exception as exc:
            logger.error("Embedding batch failed: %s — storing without vectors", exc)
            vectors = [None] * len(batch)

        # Store
        items: list[Dict[str, Any]] = []
        for chunk, vector in zip(batch, vectors):
            importance = score_importance(chunk.text, {"role": chunk.role})
            category = chunk.role
            if rule_detector.detect(chunk.text) or rule_detector.check_safeword(chunk.text):
                category = "rule"
                importance = 1.0
                logger.info(f"Rule detected: {chunk.text[:50]}...")

            items.append({
                "id": chunk.md5,
                "text": chunk.text,
                "vector": vector,
                "category": category,
                "importance": importance,
                "source_session": chunk.session_id,
            })

        ids = storage.store_memories_batch(items)
        stored += len(ids)

        # Knowledge graph
        if knowledge_graph:
            for chunk in batch:
                try:
                    knowledge_graph.process_text(
                        chunk.text,
                        source=chunk.session_id,
                        timestamp=chunk.timestamp,
                    )
                except Exception as exc:
                    logger.debug("KG extraction failed: %s", exc)

        if progress_cb:
            progress_cb(min(batch_start + batch_size, total), total)

        logger.info(
            "Progress: %d/%d stored", min(batch_start + batch_size, total), total
        )

    return {
        "sessions": len(sessions),
        "chunks": len(all_chunks),
        "stored": stored,
        "skipped_dup": skipped_dup + skipped_existing,
    }
