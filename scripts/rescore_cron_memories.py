#!/usr/bin/env python3
"""Rescore cron/automated output memories to lower importance.

Automated messages (heartbeats, cron summaries, system notifications) often
get stored with the same importance as human conversations. This script
finds them via regex patterns and caps their importance at a configurable max.

Usage:
    python scripts/rescore_cron_memories.py                    # default patterns
    python scripts/rescore_cron_memories.py --max-importance 0.25
    python scripts/rescore_cron_memories.py --dry-run

Patterns can be customized by editing CRON_PATTERNS below or by setting
AGENT_MEMORY_CRON_PATTERNS as a comma-separated list of regex strings.
"""

from __future__ import annotations

import argparse
import os
import re
import sqlite3
from pathlib import Path

# Default patterns that match common automated/cron output
DEFAULT_PATTERNS = [
    r"^\[cron:",
    r"HEARTBEAT_OK",
    r"Return your summary as plain text",
    r"^\[System Message\].*cron job",
    r"^---.*SYNC START",
]


def load_patterns() -> list[re.Pattern]:
    """Load cron patterns from env or defaults."""
    env_patterns = os.environ.get("AGENT_MEMORY_CRON_PATTERNS")
    if env_patterns:
        raw = [p.strip() for p in env_patterns.split(",") if p.strip()]
    else:
        raw = DEFAULT_PATTERNS
    return [re.compile(p, re.IGNORECASE) for p in raw]


def main():
    parser = argparse.ArgumentParser(description="Rescore cron/automated memories")
    parser.add_argument("--max-importance", type=float, default=0.30,
                        help="Cap importance at this value (default: 0.30)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be updated without writing")
    parser.add_argument("--agent", default=None,
                        help="Agent DB to rescore (default: main)")
    args = parser.parse_args()

    db_path = os.environ.get("AGENT_MEMORY_DB",
                             str(Path.home() / ".agent-memory" / "memory.sqlite"))
    if args.agent and args.agent != "main":
        base = Path(db_path).parent
        db_path = str(base / f"memory-{args.agent}.sqlite")

    patterns = load_patterns()

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT id, text, importance FROM memories WHERE deleted_at IS NULL"
    ).fetchall()

    updated = 0
    for r in rows:
        text = r["text"] or ""
        imp = float(r["importance"] or 0.5)
        if any(p.search(text) for p in patterns) and imp > args.max_importance:
            if not args.dry_run:
                conn.execute(
                    "UPDATE memories SET importance = ? WHERE id = ?",
                    (args.max_importance, r["id"]),
                )
            updated += 1

    if not args.dry_run:
        conn.commit()

    action = "would update" if args.dry_run else "updated"
    print(f"processed={len(rows)} {action}={updated} max_importance={args.max_importance}")
    conn.close()


if __name__ == "__main__":
    main()
