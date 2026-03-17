#!/usr/bin/env python3
"""Reclassify existing memories using the updated classifier.

Usage:
    python scripts/reclassify_memories.py --dry-run     # preview changes
    python scripts/reclassify_memories.py --apply        # apply changes
    python scripts/reclassify_memories.py --apply --agent main  # single agent
"""

import argparse
import sqlite3
import sys
from collections import Counter
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent_memory.ingest import classify_memory_type  # noqa: E402
from agent_memory.config import load_config  # noqa: E402


def find_databases(config) -> list[tuple[str, Path]]:
    """Find all agent memory databases across all known data directories."""
    search_dirs = set()

    # Config-resolved path
    search_dirs.add(Path(config.db_path).parent)
    # Legacy path
    legacy = Path.home() / ".asuman"
    if legacy.exists():
        search_dirs.add(legacy)
    # New standard path
    new_std = Path.home() / ".agent-memory"
    if new_std.exists():
        search_dirs.add(new_std)

    dbs = {}  # agent_name -> path (dedup by agent)
    for data_dir in search_dirs:
        main_db = data_dir / "memory.sqlite"
        if main_db.exists() and "main" not in dbs:
            dbs["main"] = main_db

        for db_file in sorted(data_dir.glob("memory-*.sqlite")):
            agent_name = db_file.stem.replace("memory-", "")
            if agent_name not in dbs:
                dbs[agent_name] = db_file

    return sorted(dbs.items())


def reclassify_db(agent: str, db_path: Path, dry_run: bool = True) -> dict:
    """Reclassify memories in a single database."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    results = {
        "agent": agent,
        "total": 0,
        "changed": 0,
        "unchanged": 0,
        "transitions": Counter(),
    }

    # Check if memories table exists
    has_table = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='memories'"
    ).fetchone()
    if not has_table:
        conn.close()
        return results

    rows = conn.execute(
        "SELECT id, text, memory_type FROM memories WHERE deleted_at IS NULL"
    ).fetchall()

    results["total"] = len(rows)
    updates = []

    for row in rows:
        old_type = row["memory_type"] or "other"
        text = row["text"] or ""
        new_type = classify_memory_type(text)

        if new_type != old_type:
            results["changed"] += 1
            results["transitions"][(old_type, new_type)] += 1
            updates.append((new_type, row["id"]))
        else:
            results["unchanged"] += 1

    if not dry_run and updates:
        for new_type, mem_id in updates:
            conn.execute(
                "UPDATE memories SET memory_type = ? WHERE id = ?",
                (new_type, mem_id),
            )
        conn.commit()

    conn.close()
    return results


def main():
    parser = argparse.ArgumentParser(description="Reclassify memories with updated classifier")
    parser.add_argument("--dry-run", action="store_true", help="Preview changes without applying")
    parser.add_argument("--apply", action="store_true", help="Apply reclassification")
    parser.add_argument("--agent", type=str, default=None, help="Single agent to reclassify")
    args = parser.parse_args()

    if not args.dry_run and not args.apply:
        print("Error: specify --dry-run or --apply")
        sys.exit(1)

    dry_run = not args.apply
    config = load_config()
    databases = find_databases(config)

    if args.agent:
        databases = [(a, p) for a, p in databases if a == args.agent]
        if not databases:
            print(f"Error: agent '{args.agent}' not found")
            sys.exit(1)

    mode = "DRY RUN" if dry_run else "APPLYING"
    print(f"=== Memory Reclassification ({mode}) ===\n")

    total_changed = 0
    total_memories = 0
    all_transitions = Counter()

    for agent, db_path in databases:
        results = reclassify_db(agent, db_path, dry_run=dry_run)
        total_changed += results["changed"]
        total_memories += results["total"]
        all_transitions += results["transitions"]

        if results["changed"] > 0:
            print(f"[{agent}] {results['total']} memories, {results['changed']} would change")
            # Show top transitions for this agent
            for (old, new), count in results["transitions"].most_common(5):
                print(f"  {old} → {new}: {count}")
            print()

    print(f"=== Summary ===")
    print(f"Total memories scanned: {total_memories}")
    print(f"Total reclassified: {total_changed} ({total_changed * 100 // max(total_memories, 1)}%)")
    print()

    if all_transitions:
        print("Top transitions:")
        for (old, new), count in all_transitions.most_common(15):
            print(f"  {old:20s} → {new:20s}: {count}")

    if dry_run:
        print(f"\nThis was a dry run. Use --apply to commit changes.")


if __name__ == "__main__":
    main()
