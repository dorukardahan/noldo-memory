"""Trim asuman-memory export to top 100 high-value + 30 recent + top 50 entities."""
import os
import sqlite3
import time
from datetime import datetime
from pathlib import Path

def _resolve_data_dir() -> Path:
    """Resolve data dir: env var > ~/.agent-memory > legacy fallback."""
    env_dir = os.environ.get("AGENT_MEMORY_DATA_DIR")
    if env_dir:
        return Path(env_dir).expanduser()
    new_dir = Path.home() / ".agent-memory"
    legacy_dir = Path.home() / ".agent-memory-legacy"
    if legacy_dir.is_dir() and not new_dir.is_dir():
        return legacy_dir
    return new_dir


_data_dir = _resolve_data_dir()
DB_PATH = Path(os.environ.get("AGENT_MEMORY_DB", _data_dir / "memory.sqlite"))
_workspace = Path(os.environ.get("OPENCLAW_WORKSPACE", Path.home() / ".openclaw" / "workspace"))
OUTPUT = _workspace / "memory" / "memory-export.md"

conn = sqlite3.connect(str(DB_PATH))
conn.row_factory = sqlite3.Row
now = time.time()
week_ago = now - 7 * 86400

high_value = conn.execute("""
    SELECT id, text, category, importance, strength, created_at
    FROM memories WHERE deleted_at IS NULL
      AND (importance > 0.70 OR COALESCE(strength, 1.0) > 2.0)
    ORDER BY importance DESC, strength DESC LIMIT 100
""").fetchall()

recent_user = conn.execute("""
    SELECT id, text, category, importance, created_at
    FROM memories WHERE deleted_at IS NULL AND category = 'user' AND created_at > ?
    ORDER BY created_at DESC LIMIT 30
""", (week_ago,)).fetchall()

top_entities = conn.execute("""
    SELECT name, type, mention_count, aliases
    FROM entities WHERE mention_count > 5
    ORDER BY mention_count DESC LIMIT 50
""").fetchall()

lines = []
lines.append("# Agent Memory Export")
lines.append(f"*Auto-generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*\n")

lines.append("## High-Value Memories\n")
seen = set()
for row in high_value:
    mid = row["id"]
    if mid in seen:
        continue
    seen.add(mid)
    text = (row["text"] or "").strip()[:400]
    if not text:
        continue
    ts = datetime.fromtimestamp(row["created_at"]).strftime("%m/%d")
    lines.append(f"- [{ts}] (imp:{row['importance']:.1f}) {text}\n")

lines.append("\n## Recent User Messages (7d)\n")
for row in recent_user:
    mid = row["id"]
    if mid in seen:
        continue
    seen.add(mid)
    text = (row["text"] or "").strip()[:250]
    if not text:
        continue
    ts = datetime.fromtimestamp(row["created_at"]).strftime("%m/%d %H:%M")
    lines.append(f"- [{ts}] {text}\n")

lines.append("\n## Key Entities\n")
lines.append("| Entity | Type | Mentions |")
lines.append("|--------|------|----------|")
for row in top_entities:
    lines.append(f"| {row['name']} | {row['type'] or '?'} | {row['mention_count']} |")

OUTPUT.write_text("\n".join(lines), encoding="utf-8")
conn.close()
print(f"Trimmed: {len(high_value)} high-value, {len(recent_user)} recent, {len(top_entities)} entities")
print(f"Lines: {len(lines)}")
