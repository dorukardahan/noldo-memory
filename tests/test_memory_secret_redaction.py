import json
import sqlite3
import subprocess
import sys
from pathlib import Path


def make_db(path: Path) -> None:
    openai_like = "sk-" + "a" * 16
    bearer_value = "b" * 15
    slack_token = "xoxb-" + "1" * 12
    deleted_value = "abc" + "1234567890"
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
            updated_at REAL,
            original_text TEXT
        );
        CREATE TABLE memory_vectors (embedding BLOB);
        CREATE TABLE memory_fts (id TEXT, text TEXT);
        INSERT INTO memory_vectors(rowid, embedding) VALUES (42, X'00');
        INSERT INTO memory_fts(id, text) VALUES
            ('1', 'stale secret text'),
            ('2', 'already safe');
        """
    )
    rows = [
        (
            "1",
            f"OPENROUTER_API_KEY={openai_like} and Bearer {bearer_value}",
            "other",
            "default",
            42,
            None,
            1.0,
            f"token: {slack_token}",
        ),
        ("2", "already api_key=<redacted:api_key>", "other", "default", None, None, 1.0, None),
        ("3", f"deleted secret={deleted_value}", "other", "default", None, 123.0, 1.0, None),
    ]
    conn.executemany(
        """
        INSERT INTO memories
        (id, text, memory_type, namespace, vector_rowid, deleted_at, updated_at, original_text)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    conn.commit()
    conn.close()


def run_script(script_name: str, db_path: Path, *args: str) -> subprocess.CompletedProcess:
    script = Path(__file__).resolve().parents[1] / "scripts" / script_name
    return subprocess.run(
        [sys.executable, str(script), "--db", str(db_path), *args],
        text=True,
        capture_output=True,
        check=False,
    )


def test_redact_memory_secrets_dry_run_does_not_change_db_or_print_secrets(tmp_path: Path) -> None:
    db_path = tmp_path / "memory.sqlite"
    make_db(db_path)

    result = run_script("redact_memory_secrets.py", db_path, "--json")

    assert result.returncode == 0
    data = json.loads(result.stdout)
    assert data["mode"] == "dry-run"
    assert data["rows_matched"] == 1
    assert data["rows_changed"] == 1
    assert data["vectors_invalidated"] == 1
    assert ("sk-" + "a" * 16) not in result.stdout
    assert "xoxb-" not in result.stdout

    conn = sqlite3.connect(db_path)
    text, vector_rowid = conn.execute("SELECT text, vector_rowid FROM memories WHERE id = '1'").fetchone()
    conn.close()
    assert ("sk-" + "a" * 16) in text
    assert vector_rowid == 42


def test_redact_memory_secrets_apply_redacts_and_invalidates_vectors(tmp_path: Path) -> None:
    db_path = tmp_path / "memory.sqlite"
    backup_dir = tmp_path / "backups"
    make_db(db_path)

    result = run_script("redact_memory_secrets.py", db_path, "--apply", "--backup-dir", str(backup_dir), "--json")

    assert result.returncode == 0
    data = json.loads(result.stdout)
    assert data["mode"] == "apply"
    assert data["rows_matched"] == 1
    assert data["text_changed"] == 1
    assert data["original_text_changed"] == 1
    assert data["fts_refreshed"] == 1
    assert data["vectors_deleted"] == 1
    assert data["backup_path"]
    assert Path(data["backup_path"]).exists()
    assert ("sk-" + "a" * 16) not in result.stdout
    assert "xoxb-" not in result.stdout

    conn = sqlite3.connect(db_path)
    row = conn.execute("SELECT text, original_text, vector_rowid FROM memories WHERE id = '1'").fetchone()
    vector_count = conn.execute("SELECT COUNT(*) FROM memory_vectors WHERE rowid = 42").fetchone()[0]
    fts_text = conn.execute("SELECT text FROM memory_fts WHERE id = '1'").fetchone()[0]
    conn.close()

    assert row[2] is None
    assert vector_count == 0
    assert ("sk-" + "a" * 16) not in row[0]
    assert f"Bearer {'b' * 15}" not in row[0]
    assert "xoxb-" not in row[1]
    assert fts_text == row[0]
    assert "<redacted:openai_like_key>" in row[0]
    assert "Bearer <redacted:bearer_token>" in row[0]
    assert "<redacted:slack_token>" in row[1]

    audit = run_script("audit_memory_quality.py", db_path, "--json")
    audit_data = json.loads(audit.stdout)
    assert audit_data["secret_like"]["rows"] == 0
