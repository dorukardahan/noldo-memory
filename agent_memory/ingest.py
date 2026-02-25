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

# System noise patterns — gateway connects, test msgs, cron boilerplate [S9, 2026-02-17]
_NOISE_PATTERNS_RE = [
    re.compile(r"whatsapp gateway (?:connected|disconnected)", re.IGNORECASE),
    re.compile(r"slack (?:socket mode )?(?:connected|disconnected)", re.IGNORECASE),
    re.compile(r"^GatewayRestart:", re.IGNORECASE),
    re.compile(r"^\[queued messages while agent was busy\]", re.IGNORECASE),
    re.compile(r"^say\s+(?:ok|hello|hi|test|something)\s*$", re.IGNORECASE),
    re.compile(r"^Conversation info \(untrusted metadata\)", re.IGNORECASE),
    re.compile(r"^Replied message \(untrusted", re.IGNORECASE),
    re.compile(r"^\[cron:", re.IGNORECASE),
    re.compile(r"^User: \[cron:", re.IGNORECASE),
    re.compile(r"^User: Conversation info", re.IGNORECASE),
    re.compile(r"^NO_REPLY$"),
    re.compile(r"^User: Replied message \(untrusted", re.IGNORECASE),
    re.compile(r"^User: Pre-compaction memory flush", re.IGNORECASE),
]


def _md5(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:16]


_ENTITY_NAME_RE = re.compile(
    r"\b[A-ZÇĞİÖŞÜ][a-zçğıöşü]{2,}\s+[A-ZÇĞİÖŞÜ][a-zçğıöşü]{2,}\b",
    re.UNICODE,
)

_FACTUAL_PATTERNS_RE = re.compile(
    r"\b(?:is|are|was|were|born|lives?|works?|located|founded|means|"
    r"adım|adim|ismim|my\s+name\s+is|name\s+is|yaşar|yaşıyor|"
    r"calisir|çalışır|çalışıyor|calisiyor|dedi|söyledi|soyledi|var(?:dır)?)\b",
    re.IGNORECASE | re.UNICODE,
)

_PREFERENCE_KEYWORDS_RE = re.compile(
    r"\b(?:sevdiğim|sevdigim|prefer|like|hate|always|never)\b",
    re.IGNORECASE | re.UNICODE,
)


def classify_memory_type(text: str) -> str:
    """Classify memory text into fact/preference/rule/conversation."""
    content = (text or "").strip()
    if not content:
        return "conversation"

    if _ENTITY_NAME_RE.search(content) or _FACTUAL_PATTERNS_RE.search(content):
        return "fact"
    if _PREFERENCE_KEYWORDS_RE.search(content):
        return "preference"
    if rule_detector.detect(content) or rule_detector.check_safeword(content):
        return "rule"
    return "conversation"


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


# Patterns for operationally important tool outputs [S13, 2026-02-17]
_IMPORTANT_TOOL_PATTERNS = [
    re.compile(r"systemctl\s+(restart|stop|start|enable|disable)", re.IGNORECASE),
    re.compile(r"docker\s+(compose|build|push|up|down|restart|stop)", re.IGNORECASE),
    re.compile(r"git\s+(push|merge|tag|commit)", re.IGNORECASE),
    re.compile(r"\b(deploy|migrate|backup|restore)\b", re.IGNORECASE),
    re.compile(r"\b(error|fail|crash|killed|SIGKILL|OOM|denied)\b", re.IGNORECASE),
    re.compile(r"apt(?:-get)?\s+(install|upgrade|remove)", re.IGNORECASE),
    re.compile(r"npm\s+(publish|install|update)", re.IGNORECASE),
    re.compile(r"pip3?\s+install", re.IGNORECASE),
    re.compile(r"ufw\s+(allow|deny|enable|disable)", re.IGNORECASE),
    re.compile(r"certbot", re.IGNORECASE),
]


def _is_tool_call(entry: Dict[str, Any]) -> bool:
    """Return True if the JSONL entry is a tool call that should be SKIPPED.

    Selectively allows important tool results through (deploy, git, config,
    service, error outputs). [S13, 2026-02-17]
    """
    msg = entry.get("message", {})
    role = msg.get("role", "")
    if role in ("tool", "function"):
        # Check if tool result contains operationally important content
        content = _extract_text(msg.get("content", ""))
        if content and any(p.search(content) for p in _IMPORTANT_TOOL_PATTERNS):
            return False  # Don't skip — capture this important tool output
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
    # Pattern-based noise filtering [S9, 2026-02-17]
    for pattern in _NOISE_PATTERNS_RE:
        if pattern.search(stripped):
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
                    text=pending_user[:4000],
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
                    text=pending_user[:4000],
                    timestamp=pending_ts or timestamp,
                    session_id=session_id,
                    role="user",
                ))
            pending_user = text[:4000]
            pending_ts = timestamp

        elif role == "assistant":
            if pending_user:
                # Form Q&A pair
                qa_text = f"User: {pending_user}\nAssistant: {text[:3000]}"
                chunks.append(Chunk(
                    text=qa_text[:4000],
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
                        text=text[:4000],
                        timestamp=timestamp,
                        session_id=session_id,
                        role="assistant",
                    ))

    # Flush any remaining pending user message
    if pending_user:
        chunks.append(Chunk(
            text=pending_user[:4000],
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
            memory_type = classify_memory_type(chunk.text)
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
                "memory_type": memory_type,
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
