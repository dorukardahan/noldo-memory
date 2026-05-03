#!/usr/bin/env python3
"""Re-embed all memories with current embedding configuration.

Reads model, dimensions, and API URL from environment / .env file.
Safe for profile changes (e.g. switching from 0.6B to 4B).

Steps:
1. Read all memories from SQLite
2. Drop old memory_vectors table
3. Create new table with configured dimensions
4. Re-embed each memory via the configured embedding endpoint
5. Update vector_rowid references

Usage:
  # Uses settings from .env / environment:
  python3 scripts/reindex_embeddings.py

  # Override for a specific agent DB:
  AGENT_MEMORY_DB=~/.agent-memory/memory-codex.sqlite python3 scripts/reindex_embeddings.py

  # Dry run (show what would happen):
  python3 scripts/reindex_embeddings.py --dry-run
"""

import argparse
import os
import sqlite3
import struct
import sys
import time

import requests

# ─── Configuration (from environment, matching agent_memory/config.py) ───
def _resolve_db_path() -> str:
    """Resolve DB path: env var > ~/.agent-memory > legacy fallback."""
    if os.environ.get("AGENT_MEMORY_DB"):
        return os.path.expanduser(os.environ["AGENT_MEMORY_DB"])
    new_dir = os.path.expanduser("~/.agent-memory")
    legacy_dir = os.path.expanduser("~/.agent-memory-legacy")
    if os.path.isdir(legacy_dir) and not os.path.isdir(new_dir):
        return os.path.join(legacy_dir, "memory.sqlite")
    return os.path.join(new_dir, "memory.sqlite")


DB_PATH = _resolve_db_path()
EMBED_BASE_URL = os.environ.get("OPENROUTER_BASE_URL", "http://localhost:8090/v1").rstrip("/")
EMBED_URL = f"{EMBED_BASE_URL}/embeddings"
EMBED_MODEL = os.environ.get("AGENT_MEMORY_MODEL", "qwen/qwen3-embedding-0.6b")
EMBED_DIM = int(os.environ.get("AGENT_MEMORY_DIMENSIONS", "1024"))
MAX_CHARS = int(os.environ.get("AGENT_MEMORY_MAX_EMBED_CHARS", "3500"))
API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
BATCH_SIZE = 2  # conservative for local llama-server
SLEEP_BETWEEN = 0.5  # seconds between batches


def embed_batch(texts: list[str]) -> list[list[float]]:
    """Call embedding endpoint for a batch of texts."""
    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"

    resp = requests.post(
        EMBED_URL,
        json={"input": texts, "model": EMBED_MODEL},
        headers=headers,
        timeout=180,
    )
    resp.raise_for_status()
    data = resp.json()
    embeddings = sorted(data["data"], key=lambda d: d["index"])
    return [item["embedding"][:EMBED_DIM] for item in embeddings]


def float_list_to_blob(vec: list[float]) -> bytes:
    """Convert list of floats to sqlite-vec compatible blob."""
    return struct.pack(f"{len(vec)}f", *vec)


def main():
    parser = argparse.ArgumentParser(description="Re-embed all memories with current config")
    parser.add_argument("--dry-run", action="store_true", help="Show plan without executing")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Texts per API call")
    parser.add_argument("--sleep", type=float, default=SLEEP_BETWEEN, help="Seconds between batches")
    args = parser.parse_args()

    print("Configuration:")
    print(f"  DB:         {DB_PATH}")
    print(f"  Embed URL:  {EMBED_URL}")
    print(f"  Model:      {EMBED_MODEL}")
    print(f"  Dimensions: {EMBED_DIM}")
    print(f"  Max chars:  {MAX_CHARS}")
    print(f"  Batch size: {args.batch_size}")
    print()

    if not os.path.exists(DB_PATH):
        print(f"ERROR: Database not found: {DB_PATH}")
        sys.exit(1)

    conn = sqlite3.connect(DB_PATH)
    conn.enable_load_extension(True)
    import sqlite_vec
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.execute("PRAGMA journal_mode=WAL")

    # 1. Count memories
    rows = conn.execute("""
        SELECT m.id, m.text, m.vector_rowid
        FROM memories m
        WHERE m.text IS NOT NULL AND m.text != '' AND m.deleted_at IS NULL
        ORDER BY m.id
    """).fetchall()
    print(f"Found {len(rows)} active memories to re-embed")

    if not rows:
        print("No memories found. Exiting.")
        return

    if args.dry_run:
        print(f"\nDry run: would drop memory_vectors, recreate with float[{EMBED_DIM}], re-embed {len(rows)} memories")
        conn.close()
        return

    # 2. Test embedding endpoint
    print("Testing embedding endpoint...")
    try:
        test_vec = embed_batch(["test"])
        actual_dim = len(test_vec[0])
        print(f"  Endpoint OK, returns {actual_dim} dimensions")
        if actual_dim < EMBED_DIM:
            print(f"  WARNING: model returns {actual_dim} dims but config expects {EMBED_DIM}")
            print("  Vectors will be truncated/padded. Consider updating AGENT_MEMORY_DIMENSIONS.")
    except Exception as e:
        print(f"ERROR: Cannot reach embedding endpoint: {e}")
        sys.exit(1)

    # 3. Drop old vector table and recreate
    print(f"Recreating memory_vectors with float[{EMBED_DIM}]...")
    conn.execute("DROP TABLE IF EXISTS memory_vectors")
    conn.execute(f"""
        CREATE VIRTUAL TABLE memory_vectors USING vec0(
            embedding float[{EMBED_DIM}]
        )
    """)
    conn.commit()

    # 4. Re-embed in batches
    total = len(rows)
    done = 0
    errors = 0
    start_time = time.time()

    for i in range(0, total, args.batch_size):
        batch = rows[i:i + args.batch_size]
        texts = [row[1][:MAX_CHARS] for row in batch]

        try:
            vectors = embed_batch(texts)
        except Exception as e:
            print(f"  ERROR batch {i}-{i+len(batch)}: {e}")
            errors += len(batch)
            continue

        for (mem_id, _content, _old_rowid), vec in zip(batch, vectors):
            blob = float_list_to_blob(vec)
            cursor = conn.execute(
                "INSERT INTO memory_vectors(embedding) VALUES (?)",
                (blob,),
            )
            new_rowid = cursor.lastrowid
            conn.execute(
                "UPDATE memories SET vector_rowid = ? WHERE id = ?",
                (new_rowid, mem_id),
            )

        conn.commit()
        done += len(batch)
        elapsed = time.time() - start_time
        rate = done / elapsed if elapsed > 0 else 0
        eta = (total - done) / rate if rate > 0 else 0
        print(f"  [{done}/{total}] {rate:.1f} mem/s, ETA: {eta:.0f}s")

        if i + args.batch_size < total:
            time.sleep(args.sleep)

    elapsed = time.time() - start_time
    print(f"\nDone! Re-embedded {done}/{total} memories in {elapsed:.1f}s ({errors} errors)")
    if done > 0:
        print(f"Average: {elapsed/done*1000:.1f}ms per memory")

    # 5. Verify
    vec_count = conn.execute(
        "SELECT COUNT(*) FROM memories WHERE vector_rowid IS NOT NULL AND deleted_at IS NULL"
    ).fetchone()[0]
    vectorless = conn.execute(
        "SELECT COUNT(*) FROM memories WHERE vector_rowid IS NULL AND deleted_at IS NULL"
    ).fetchone()[0]
    print(f"Vectors: {vec_count}, Vectorless: {vectorless}")

    conn.close()
    print("Reindex complete!")


if __name__ == "__main__":
    main()
