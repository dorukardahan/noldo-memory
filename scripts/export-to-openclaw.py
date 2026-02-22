#!/usr/bin/env python3
"""Export whatsapp-memory highlights to OpenClaw workspace for indexing.

Runs daily via cron. Exports high-importance memories from the last N days
as a single markdown file that OpenClaw auto-indexes.

Usage:
    python3 ./scripts/export-to-openclaw.py [--days 7]
"""

import argparse
import json
import logging
import os
import sqlite3
import time
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DB_PATH = os.environ.get("AGENT_MEMORY_DB", os.path.expanduser("~/.asuman/memory.sqlite"))
OUTPUT_DIR = Path("/root/.openclaw/workspace/memory")
OUTPUT_FILE = OUTPUT_DIR / "whatsapp-highlights.md"


def export(days: int = 7, min_importance: float = 0.6, min_strength: float = 0.5) -> None:
    """Export recent high-value memories to markdown."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    cutoff = time.time() - (days * 86400)

    rows = conn.execute(
        """
        SELECT id, text, category, importance, strength, created_at, last_accessed_at
          FROM memories
         WHERE deleted_at IS NULL
           AND created_at > ?
           AND importance >= ?
           AND COALESCE(strength, 1.0) >= ?
         ORDER BY importance DESC, created_at DESC
         LIMIT 200
        """,
        (cutoff, min_importance, min_strength),
    ).fetchall()

    if not rows:
        logger.info("No memories matching criteria (days=%d imp>=%.1f str>=%.1f)", days, min_importance, min_strength)
        return

    entities = conn.execute(
        """
        SELECT name, type, mention_count, aliases
          FROM entities
         ORDER BY mention_count DESC
         LIMIT 50
        """,
    ).fetchall()

    conn.close()

    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        "# WhatsApp Memory Highlights",
        "",
        "Auto-exported from whatsapp-memory ({} memories, {} top entities).".format(len(rows), len(entities)),
        "Last updated: {}. Covers last {} days.".format(now_str, days),
        "",
        "---",
        "",
    ]

    by_cat = {}
    for r in rows:
        cat = r["category"] or "other"
        by_cat.setdefault(cat, []).append(r)

    for cat, memories in sorted(by_cat.items()):
        lines.append("## {} ({})".format(cat.title(), len(memories)))
        lines.append("")
        for m in memories[:50]:
            ts = datetime.fromtimestamp(m["created_at"]).strftime("%m/%d %H:%M")
            imp = m["importance"]
            text = m["text"].replace("\n", " ").strip()
            if len(text) > 300:
                text = text[:297] + "..."
            strength = m["strength"] or 1.0
            lines.append("- **[{}]** (imp:{:.2f} str:{:.1f}) {}".format(ts, imp, strength, text))
        lines.append("")

    if entities:
        lines.append("## Key Entities")
        lines.append("")
        for e in entities:
            aliases = json.loads(e["aliases"] or "[]")
            alias_str = " (aka: {})".format(", ".join(aliases)) if aliases else ""
            lines.append("- **{}** [{}] — {} mentions{}".format(e["name"], e["type"], e["mention_count"], alias_str))
        lines.append("")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Exported %d memories + %d entities -> %s", len(rows), len(entities), OUTPUT_FILE)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=7, help="Look back N days (default: 7)")
    parser.add_argument("--min-importance", type=float, default=0.6)
    parser.add_argument("--min-strength", type=float, default=0.5)
    args = parser.parse_args()
    export(days=args.days, min_importance=args.min_importance, min_strength=args.min_strength)
