#!/usr/bin/env python3
"""Re-embed all memories using the new local llama-server.

Steps:
1. Read all memories from SQLite
2. Drop old memory_vectors table (4096-dim)
3. Create new memory_vectors table (1024-dim)
4. Re-embed each memory via llama-server /v1/embeddings
5. Insert new vectors

Usage: python3 scripts/reindex_embeddings.py
"""

import os
import sqlite3
import struct
import time

import requests

DB_PATH = os.environ.get("AGENT_MEMORY_DB", os.path.expanduser("~/.asuman/memory.sqlite"))
EMBED_URL = "http://localhost:8090/v1/embeddings"
NEW_DIM = 1024
BATCH_SIZE = 20  # texts per API call


def embed_batch(texts: list[str]) -> list[list[float]]:
    """Call llama-server to embed a batch of texts."""
    resp = requests.post(
        EMBED_URL,
        json={"input": texts, "model": "qwen3"},
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()
    embeddings = sorted(data["data"], key=lambda d: d["index"])
    return [item["embedding"] for item in embeddings]


def float_list_to_blob(vec: list[float]) -> bytes:
    """Convert list of floats to sqlite-vec compatible blob."""
    return struct.pack(f"{len(vec)}f", *vec)


def main():
    conn = sqlite3.connect(DB_PATH)
    conn.enable_load_extension(True)
    import sqlite_vec
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.execute("PRAGMA journal_mode=WAL")

    # 1. Read all memories with their vector rowids
    print("Reading memories...")
    rows = conn.execute("""
        SELECT m.id, m.text, m.vector_rowid
        FROM memories m
        WHERE m.text IS NOT NULL AND m.text != ''
        ORDER BY m.id
    """).fetchall()
    print(f"Found {len(rows)} memories to re-embed")

    if not rows:
        print("No memories found. Exiting.")
        return

    # 2. Drop old vector table and recreate with new dimensions
    print(f"Recreating memory_vectors table with float[{NEW_DIM}]...")
    conn.execute("DROP TABLE IF EXISTS memory_vectors")
    conn.execute(f"""
        CREATE VIRTUAL TABLE memory_vectors USING vec0(
            embedding float[{NEW_DIM}]
        )
    """)
    conn.commit()

    # 3. Re-embed in batches
    total = len(rows)
    done = 0
    start_time = time.time()

    for i in range(0, total, BATCH_SIZE):
        batch = rows[i:i + BATCH_SIZE]
        texts = [row[1][:2000] for row in batch]  # truncate long texts

        try:
            vectors = embed_batch(texts)
        except Exception as e:
            print(f"ERROR embedding batch {i}-{i+len(batch)}: {e}")
            continue

        for (mem_id, content, old_rowid), vec in zip(batch, vectors):
            blob = float_list_to_blob(vec)
            cursor = conn.execute(
                "INSERT INTO memory_vectors(embedding) VALUES (?)",
                (blob,),
            )
            new_rowid = cursor.lastrowid
            # Update the memory to point to new vector rowid
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

    elapsed = time.time() - start_time
    print(f"\nDone! Re-embedded {done}/{total} memories in {elapsed:.1f}s")
    print(f"Average: {elapsed/done*1000:.1f}ms per memory")

    # 4. Verify
    count = conn.execute("SELECT COUNT(*) FROM memory_vectors").fetchone()[0]
    print(f"Vectors in table: {count}")

    # Check dimensions
    sample = conn.execute(
        "SELECT embedding FROM memory_vectors LIMIT 1"
    ).fetchone()
    if sample:
        dim = len(struct.unpack(f"{NEW_DIM}f", sample[0]))
        print(f"Vector dimensions: {dim}")

    conn.close()
    print("Reindex complete!")


if __name__ == "__main__":
    main()
