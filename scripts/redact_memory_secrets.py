#!/usr/bin/env python3
"""Redact secret-like values from an existing NoldoMem SQLite database.

Dry-run is the default. The script never prints memory text or secret values.
When applied, it creates a SQLite backup, redacts matching values in text
columns, and invalidates vectors for rows whose searchable text changed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agent_memory.secret_redaction import redact_secret_like  # noqa: E402


def default_db_path() -> Path:
    env_path = os.environ.get("AGENT_MEMORY_DB")
    if env_path:
        return Path(env_path).expanduser()
    try:
        from agent_memory.config import load_config

        return Path(load_config().db_path).expanduser()
    except Exception:
        return Path.home() / ".agent-memory" / "memory.sqlite"


def table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}


def has_table(conn: sqlite3.Connection, table: str) -> bool:
    return bool(
        conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type IN ('table', 'virtual table') AND name = ?",
            (table,),
    ).fetchone()
    )


def try_load_sqlite_vec(conn: sqlite3.Connection) -> bool:
    try:
        import sqlite_vec

        conn.enable_load_extension(True)
        try:
            sqlite_vec.load(conn)
        finally:
            conn.enable_load_extension(False)
        return True
    except Exception:
        try:
            conn.enable_load_extension(False)
        except Exception:
            pass
        return False


def active_where(columns: set[str], include_deleted: bool) -> str:
    if include_deleted:
        return "1=1"
    if "deleted_at" in columns:
        return "deleted_at IS NULL"
    if "deleted" in columns:
        return "COALESCE(deleted, 0) = 0"
    return "1=1"


def hash_id(memory_id: Any) -> str:
    return hashlib.sha256(str(memory_id).encode("utf-8", "ignore")).hexdigest()[:16]


def backup_sqlite_db(db_path: Path, backup_dir: Path) -> Path:
    backup_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    backup_path = backup_dir / f"{db_path.name}.pre-secret-redaction-{timestamp}.bak"

    src = sqlite3.connect(str(db_path))
    try:
        dst = sqlite3.connect(str(backup_path))
        try:
            src.backup(dst)
        finally:
            dst.close()
    finally:
        src.close()

    return backup_path


def redact_db(
    db_path: Path,
    *,
    apply: bool = False,
    include_deleted: bool = False,
    backup_dir: Path | None = None,
    sample_limit: int = 20,
) -> dict[str, Any]:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        if not has_table(conn, "memories"):
            return {"db_path": str(db_path), "error": "missing memories table"}

        columns = table_columns(conn, "memories")
        text_columns = [col for col in ("text", "original_text") if col in columns]
        if not text_columns:
            return {"db_path": str(db_path), "error": "missing text columns"}

        where = active_where(columns, include_deleted)
        select_columns = ["id", *text_columns]
        if "vector_rowid" in columns:
            select_columns.append("vector_rowid")
        rows = conn.execute(
            f"SELECT {', '.join(select_columns)} FROM memories WHERE {where}"
        ).fetchall()

        pattern_counts: Counter[str] = Counter()
        sample_hashes: list[str] = []
        row_updates: list[dict[str, Any]] = []
        rows_matched = 0
        text_changed = 0
        original_text_changed = 0
        vectors_to_delete: list[int] = []

        for row in rows:
            updates: dict[str, str | None] = {}
            row_counts: Counter[str] = Counter()
            for column in text_columns:
                redacted, counts = redact_secret_like(row[column])
                if counts:
                    row_counts.update(counts)
                    if redacted != row[column]:
                        updates[column] = redacted

            if not row_counts:
                continue

            rows_matched += 1
            pattern_counts.update(row_counts)
            if len(sample_hashes) < sample_limit:
                sample_hashes.append(hash_id(row["id"]))
            if "text" in updates:
                text_changed += 1
                if "vector_rowid" in columns and row["vector_rowid"] is not None:
                    vectors_to_delete.append(int(row["vector_rowid"]))
            if "original_text" in updates:
                original_text_changed += 1

            if updates:
                row_updates.append({"id": row["id"], "updates": updates})

        report: dict[str, Any] = {
            "db_path": str(db_path),
            "mode": "apply" if apply else "dry-run",
            "scope": "all rows" if include_deleted else "active rows",
            "rows_scanned": len(rows),
            "rows_matched": rows_matched,
            "rows_changed": len(row_updates),
            "text_changed": text_changed,
            "original_text_changed": original_text_changed,
            "fts_refreshed": 0,
            "vectors_invalidated": len(vectors_to_delete),
            "vectors_deleted": 0,
            "vector_delete_skipped": 0,
            "pattern_counts": dict(pattern_counts),
            "sample_id_hashes": sample_hashes,
            "backup_path": None,
        }

        if not apply or not row_updates:
            return report

        backup_path = backup_sqlite_db(db_path, backup_dir or (db_path.parent / "backups"))
        report["backup_path"] = str(backup_path)

        has_vectors = has_table(conn, "memory_vectors")
        if has_vectors and vectors_to_delete and not try_load_sqlite_vec(conn):
            has_vectors = False
            report["vector_delete_skipped"] = len(vectors_to_delete)
        has_fts = has_table(conn, "memory_fts")
        now = time.time()
        with conn:
            for item in row_updates:
                set_parts = []
                params: list[Any] = []
                for column, value in item["updates"].items():
                    set_parts.append(f"{column} = ?")
                    params.append(value)
                if "updated_at" in columns:
                    set_parts.append("updated_at = ?")
                    params.append(now)
                if "text" in item["updates"] and "vector_rowid" in columns:
                    set_parts.append("vector_rowid = NULL")
                params.append(item["id"])
                conn.execute(f"UPDATE memories SET {', '.join(set_parts)} WHERE id = ?", params)
                if "text" in item["updates"] and has_fts:
                    conn.execute("DELETE FROM memory_fts WHERE id = ?", (item["id"],))
                    conn.execute(
                        "INSERT INTO memory_fts(id, text) VALUES (?, ?)",
                        (item["id"], item["updates"]["text"]),
                    )
                    report["fts_refreshed"] += 1

            if has_vectors:
                for rowid in vectors_to_delete:
                    cur = conn.execute("DELETE FROM memory_vectors WHERE rowid = ?", (rowid,))
                    report["vectors_deleted"] += cur.rowcount if cur.rowcount and cur.rowcount > 0 else 0

        return report
    finally:
        conn.close()


def print_text_report(report: dict[str, Any]) -> None:
    if "error" in report:
        print(f"db_path={report['db_path']}")
        print(f"error={report['error']}")
        return

    print(f"db_path={report['db_path']}")
    print(f"mode={report['mode']}")
    print(f"scope={report['scope']}")
    for key in (
        "rows_scanned",
        "rows_matched",
        "rows_changed",
        "text_changed",
        "original_text_changed",
        "fts_refreshed",
        "vectors_invalidated",
        "vectors_deleted",
        "vector_delete_skipped",
    ):
        print(f"{key}={report[key]}")
    print("pattern_counts:")
    for key, value in report["pattern_counts"].items():
        print(f"  {key}: {value}")
    if report["sample_id_hashes"]:
        print("sample_id_hashes:")
        for value in report["sample_id_hashes"]:
            print(f"  {value}")
    if report["backup_path"]:
        print(f"backup_path={report['backup_path']}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Redact secret-like values from NoldoMem DB")
    parser.add_argument("--db", type=Path, default=default_db_path(), help="SQLite DB path")
    parser.add_argument("--apply", action="store_true", help="Apply changes. Default is dry-run")
    parser.add_argument("--include-deleted", action="store_true", help="Include soft-deleted rows")
    parser.add_argument("--backup-dir", type=Path, default=None, help="Backup directory for apply mode")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of text")
    parser.add_argument("--sample-limit", type=int, default=20, help="Number of hashed row IDs to include")
    args = parser.parse_args()

    report = redact_db(
        args.db.expanduser(),
        apply=args.apply,
        include_deleted=args.include_deleted,
        backup_dir=args.backup_dir.expanduser() if args.backup_dir else None,
        sample_limit=args.sample_limit,
    )
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print_text_report(report)
    return 1 if "error" in report else 0


if __name__ == "__main__":
    raise SystemExit(main())
