import json
import sqlite3
import subprocess
import sys
from pathlib import Path


DUMMY_VALUE = "abc" + "1234567890"


def make_db(path: Path) -> None:
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE memories (
            id TEXT PRIMARY KEY,
            text TEXT NOT NULL,
            memory_type TEXT DEFAULT 'other',
            namespace TEXT DEFAULT 'default',
            vector_rowid INTEGER,
            deleted_at REAL,
            original_text TEXT
        );
        """
    )
    rows = [
        ("1", "normal memory", "fact", "default", 1, None, None),
        ("2", "dup", "other", "default", 2, None, None),
        ("3", "dup", "other", "default", 3, None, None),
        ("4", "tiny", "bad_type", "", None, None, None),
        ("5", f"api_key={DUMMY_VALUE}", "other", "ops", 5, None, None),
        ("6", f"deleted secret={DUMMY_VALUE}", "other", "ops", None, 123.0, None),
    ]
    conn.executemany(
        """
        INSERT INTO memories
        (id, text, memory_type, namespace, vector_rowid, deleted_at, original_text)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    conn.commit()
    conn.close()


def run_audit(db_path: Path, *args: str) -> subprocess.CompletedProcess:
    script = Path(__file__).resolve().parents[1] / "scripts" / "audit_memory_quality.py"
    return subprocess.run(
        [sys.executable, str(script), "--db", str(db_path), *args],
        text=True,
        capture_output=True,
        check=False,
    )


def test_audit_memory_quality_reports_aggregate_counts_without_text(tmp_path: Path) -> None:
    db_path = tmp_path / "memory.sqlite"
    make_db(db_path)

    result = run_audit(db_path, "--json")

    assert result.returncode == 0
    data = json.loads(result.stdout)
    assert data["counts"]["total"] == 6
    assert data["counts"]["scoped"] == 5
    assert data["counts"]["vectorless"] == 1
    assert data["counts"]["invalid_memory_type"] == 1
    assert data["counts"]["duplicate_text_groups"] == 1
    assert data["counts"]["null_namespace"] == 1
    assert data["secret_like"]["rows"] == 1
    assert data["secret_like"]["matches"]["api_key_assignment"] == 1
    assert DUMMY_VALUE not in result.stdout
    assert "api_key=" not in result.stdout


def test_audit_memory_quality_can_fail_on_secret_like(tmp_path: Path) -> None:
    db_path = tmp_path / "memory.sqlite"
    make_db(db_path)

    result = run_audit(db_path, "--fail-on-secret-like")

    assert result.returncode == 2
    assert "secret_like" in result.stdout
