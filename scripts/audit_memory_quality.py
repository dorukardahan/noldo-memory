#!/usr/bin/env python3
"""Read-only NoldoMem SQLite quality audit.

The script intentionally does not print memory text. It reports aggregate
counts for vector coverage, invalid types, duplicate text groups, long/short
records, namespace distribution, and secret-like patterns.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sqlite3
import sys
from collections import Counter
from pathlib import Path
from typing import Any

VALID_MEMORY_TYPES = {"fact", "preference", "rule", "conversation", "lesson", "other"}

SECRET_PATTERNS = {
    "api_key_assignment": re.compile(
        r"\b(?:api[_-]?key|token|secret|password|passwd|pwd)\s*[:=]\s*[^\s,;]+",
        re.IGNORECASE,
    ),
    "bearer_token": re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=-]{12,}", re.IGNORECASE),
    "openai_like_key": re.compile(r"\bsk-[A-Za-z0-9_-]{12,}"),
    "slack_token": re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{12,}|\bxapp-[A-Za-z0-9-]{12,}"),
    "private_key_block": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
}


def default_db_path() -> Path:
    env_path = os.environ.get("AGENT_MEMORY_DB")
    if env_path:
        return Path(env_path).expanduser()
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from agent_memory.config import load_config

        return Path(load_config().db_path).expanduser()
    except Exception:
        return Path.home() / ".agent-memory" / "memory.sqlite"


def table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}


def active_where(columns: set[str], include_deleted: bool) -> str:
    if include_deleted:
        return "1=1"
    if "deleted_at" in columns:
        return "deleted_at IS NULL"
    if "deleted" in columns:
        return "COALESCE(deleted, 0) = 0"
    return "1=1"


def scalar(conn: sqlite3.Connection, sql: str, params: tuple[Any, ...] = ()) -> int:
    row = conn.execute(sql, params).fetchone()
    return int(row[0] or 0)


def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "ignore")).hexdigest()[:16]


def audit_db(db_path: Path, include_deleted: bool = False, top_limit: int = 20) -> dict[str, Any]:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        has_memories = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='memories'"
        ).fetchone()
        if not has_memories:
            return {"db_path": str(db_path), "error": "missing memories table"}

        columns = table_columns(conn, "memories")
        where = active_where(columns, include_deleted)

        report: dict[str, Any] = {
            "db_path": str(db_path),
            "scope": "all rows" if include_deleted else "active rows",
            "counts": {},
            "memory_types": {},
            "namespaces": {},
            "secret_like": {"rows": 0, "matches": {}, "sample_hashes": []},
        }

        counts = report["counts"]
        counts["total"] = scalar(conn, "SELECT COUNT(*) FROM memories")
        counts["scoped"] = scalar(conn, f"SELECT COUNT(*) FROM memories WHERE {where}")
        counts["blank_text"] = scalar(
            conn,
            f"SELECT COUNT(*) FROM memories WHERE {where} AND (text IS NULL OR TRIM(text) = '')",
        )

        if "vector_rowid" in columns:
            counts["vectorless"] = scalar(
                conn,
                f"SELECT COUNT(*) FROM memories WHERE {where} AND vector_rowid IS NULL",
            )

        if "memory_type" in columns:
            counts["invalid_memory_type"] = scalar(
                conn,
                f"""
                SELECT COUNT(*) FROM memories
                WHERE {where}
                  AND COALESCE(memory_type, 'other') NOT IN ({','.join('?' for _ in VALID_MEMORY_TYPES)})
                """,
                tuple(sorted(VALID_MEMORY_TYPES)),
            )
            report["memory_types"] = dict(
                conn.execute(
                    f"""
                    SELECT COALESCE(memory_type, 'other') AS memory_type, COUNT(*) AS count
                    FROM memories
                    WHERE {where}
                    GROUP BY COALESCE(memory_type, 'other')
                    ORDER BY count DESC
                    LIMIT ?
                    """,
                    (top_limit,),
                ).fetchall()
            )

        if "namespace" in columns:
            counts["null_namespace"] = scalar(
                conn,
                f"SELECT COUNT(*) FROM memories WHERE {where} AND (namespace IS NULL OR TRIM(namespace) = '')",
            )
            report["namespaces"] = dict(
                conn.execute(
                    f"""
                    SELECT COALESCE(namespace, '') AS namespace, COUNT(*) AS count
                    FROM memories
                    WHERE {where}
                    GROUP BY COALESCE(namespace, '')
                    ORDER BY count DESC
                    LIMIT ?
                    """,
                    (top_limit,),
                ).fetchall()
            )

        counts["duplicate_text_groups"] = scalar(
            conn,
            f"""
            SELECT COUNT(*) FROM (
              SELECT text, COUNT(*) AS c
              FROM memories
              WHERE {where} AND text IS NOT NULL
              GROUP BY text
              HAVING c > 1
            )
            """,
        )
        counts["long_over_8000_chars"] = scalar(
            conn,
            f"SELECT COUNT(*) FROM memories WHERE {where} AND LENGTH(text) > 8000",
        )
        counts["short_under_20_chars"] = scalar(
            conn,
            f"SELECT COUNT(*) FROM memories WHERE {where} AND LENGTH(TRIM(text)) < 20",
        )

        text_columns = [col for col in ("text", "original_text") if col in columns]
        match_counts: Counter[str] = Counter()
        secret_rows = 0
        sample_hashes: list[str] = []
        if text_columns:
            rows = conn.execute(
                f"SELECT id, {', '.join(text_columns)} FROM memories WHERE {where}"
            )
            for row in rows:
                haystack = "\n".join(str(row[col] or "") for col in text_columns)
                matched = False
                for name, pattern in SECRET_PATTERNS.items():
                    matches = pattern.findall(haystack)
                    if matches:
                        match_counts[name] += len(matches)
                        matched = True
                if matched:
                    secret_rows += 1
                    if len(sample_hashes) < top_limit:
                        sample_hashes.append(hash_text(str(row["id"])))

        report["secret_like"] = {
            "rows": secret_rows,
            "matches": dict(match_counts),
            "sample_id_hashes": sample_hashes,
        }
        return report
    finally:
        conn.close()


def print_text_report(report: dict[str, Any]) -> None:
    if "error" in report:
        print(f"db_path={report['db_path']}")
        print(f"error={report['error']}")
        return

    print(f"db_path={report['db_path']}")
    print(f"scope={report['scope']}")
    print("counts:")
    for key, value in report["counts"].items():
        print(f"  {key}: {value}")
    print("memory_types:")
    for key, value in report["memory_types"].items():
        print(f"  {key}: {value}")
    print("namespaces:")
    for key, value in report["namespaces"].items():
        print(f"  {key or '<blank>'}: {value}")
    print("secret_like:")
    print(f"  rows: {report['secret_like']['rows']}")
    for key, value in report["secret_like"]["matches"].items():
        print(f"  {key}: {value}")
    if report["secret_like"]["sample_id_hashes"]:
        print("  sample_id_hashes:")
        for value in report["secret_like"]["sample_id_hashes"]:
            print(f"    {value}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Read-only NoldoMem DB quality audit")
    parser.add_argument("--db", type=Path, default=default_db_path(), help="SQLite DB path")
    parser.add_argument("--include-deleted", action="store_true", help="Include soft-deleted rows")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of text")
    parser.add_argument("--top-limit", type=int, default=20, help="Top namespace/type rows and sample hashes")
    parser.add_argument(
        "--fail-on-secret-like",
        action="store_true",
        help="Exit 2 when secret-like patterns are found",
    )
    args = parser.parse_args()

    report = audit_db(args.db.expanduser(), include_deleted=args.include_deleted, top_limit=args.top_limit)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print_text_report(report)

    if args.fail_on_secret_like and report.get("secret_like", {}).get("rows", 0):
        return 2
    return 1 if "error" in report else 0


if __name__ == "__main__":
    raise SystemExit(main())
