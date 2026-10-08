"""Durable admission ledger and fenced index-job state (no public content)."""
from __future__ import annotations

import hashlib
import json
import time
import uuid
from contextlib import contextmanager

from .storage import ForgottenSourceError


class AdmissionError(Exception):
    def __init__(self, code, status=503):
        self.code, self.status = code, status
        super().__init__(code)


def canonical_request_id(value):
    if not isinstance(value, str) or len(value) != 36:
        raise ValueError('invalid_request_id')
    try:
        parsed = uuid.UUID(value)
    except ValueError:
        raise ValueError('invalid_request_id') from None
    if parsed.version != 4 or str(parsed) != value:
        raise ValueError('invalid_request_id')
    return value


def migrate(conn):
    """Add only two tables; no old-row backfill or stage relabeling."""
    conn.execute('''CREATE TABLE IF NOT EXISTS memory_operations (
        namespace TEXT NOT NULL, operation TEXT NOT NULL, request_id TEXT NOT NULL,
        fingerprint TEXT NOT NULL, state TEXT NOT NULL, durable INTEGER NOT NULL,
        receipt_json TEXT NOT NULL, memory_ids_json TEXT NOT NULL, error_code TEXT,
        created_at REAL NOT NULL, updated_at REAL NOT NULL,
        PRIMARY KEY(namespace, operation, request_id))''')
    conn.execute('''CREATE TABLE IF NOT EXISTS memory_index_jobs (
        memory_id TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
        stage TEXT NOT NULL CHECK(stage IN ('embed','graph')),
        state TEXT NOT NULL, attempts INTEGER NOT NULL, lease_until REAL,
        next_attempt_at REAL NOT NULL, error_code TEXT,
        created_at REAL NOT NULL, updated_at REAL NOT NULL,
        PRIMARY KEY(memory_id, stage))''')
    conn.execute('CREATE INDEX IF NOT EXISTS idx_index_jobs_ready ON memory_index_jobs(state,next_attempt_at,lease_until)')


@contextmanager
def sql_budget(storage, deadline):
    """Bound lock wait and VM work; restore the connection's existing policy."""
    conn = storage._get_conn()
    previous = conn.execute('PRAGMA busy_timeout').fetchone()[0]
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise AdmissionError('deadline_exceeded', 504)
    conn.execute(f'PRAGMA busy_timeout={max(1, min(50, int(remaining * 1000)))}')
    valid = getattr(storage, '_request_valid', lambda: True)
    conn.set_progress_handler(lambda: int(time.monotonic() >= deadline or not valid()), 500)
    try:
        yield
        if time.monotonic() >= deadline:
            raise AdmissionError('deadline_exceeded', 504)
    finally:
        conn.set_progress_handler(None, 0)
        conn.execute(f'PRAGMA busy_timeout={previous}')


def fingerprint(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _jobs(conn, ids):
    if not ids:
        return []
    return conn.execute('SELECT stage,state,updated_at FROM memory_index_jobs WHERE memory_id IN ('
                        + ','.join('?' for _ in ids) + ')', ids).fetchall()


def public_status(storage, namespace, operation, request_id):
    conn = storage._get_conn()
    row = conn.execute('SELECT * FROM memory_operations WHERE namespace=? AND operation=? AND request_id=?',
                       (namespace, operation, request_id)).fetchone()
    if row is None:
        return None
    receipt = json.loads(row['receipt_json'])
    ids = json.loads(row['memory_ids_json'])
    stages = {}
    jobs = _jobs(conn, ids)
    for stage in ('embed', 'graph'):
        states = [job['state'] for job in jobs if job['stage'] == stage]
        stages[stage] = next((state for state in ('blocked','failed','running','pending') if state in states),
                             'completed' if states else 'not_required')
    indexing = next((state for state in ('blocked','failed','running','pending') if state in stages.values()),
                    'completed' if 'completed' in stages.values() else 'not_required')
    state = ('blocked' if row['state'] == 'blocked' or indexing == 'blocked' else
             'failed' if row['state'] == 'failed' or indexing == 'failed' else
             'accepted' if indexing in {'pending','running'} else 'completed')
    counts = {key: int(receipt.get(key, 0)) for key in ('stored','merged','blocked','total')}
    latest_update = max([row['updated_at'], *(job['updated_at'] for job in jobs)])
    return {'request_id': request_id, 'operation': operation, 'state': state,
            'durable': bool(row['durable']), 'indexing_state': indexing, 'stage_states': stages,
            'error_code': ('source_blocked' if state == 'blocked' else
                           row['error_code'] or ('indexing_failed' if state == 'failed' else None)),
            'counts': counts, 'timing': {'elapsed_seconds': max(0.0, latest_update - row['created_at'])}}


def accept(storage, *, namespace, operation, request_id, payload, rows, total, embed_required,
           graph_required, queue_cap, deadline, revision=None):
    """Atomically commit vectorless rows, jobs and replayable metadata receipt."""
    from .metrics import record_stage_metric

    fp = fingerprint(payload)
    key = (namespace, operation, request_id)
    started = time.monotonic()
    with sql_budget(storage, deadline), storage.transaction() as conn:
        previous = conn.execute('SELECT fingerprint,receipt_json FROM memory_operations WHERE namespace=? AND operation=? AND request_id=?', key).fetchone()
        if previous:
            if previous['fingerprint'] != fp:
                raise AdmissionError('request_conflict', 409)
            receipt = json.loads(previous['receipt_json'])
            return {**receipt, **public_status(storage, *key)}
        # Replay/conflict/blocked state belongs to the stable ledger identity.
        # Mutable predecessor checks apply only to a genuinely new admission.
        if revision is not None:
            predecessor = storage.get_memory(revision['supersedes'])
            if predecessor is None or predecessor['namespace'] != namespace:
                raise AdmissionError('previous_memory_not_found', 404)
            if predecessor.get('valid_to') is not None:
                raise AdmissionError('previous_memory_superseded', 409)
        stored = merged = blocked = 0
        ids = []
        revision_receipt = {}
        required = ['embed'] if embed_required else []
        if graph_required:
            required.append('graph')
        for item in rows:
            if time.monotonic() >= deadline:
                raise AdmissionError('deadline_exceeded', 504)
            try:
                if revision is not None:
                    try:
                        result = storage.revise_memory(
                            revision['supersedes'], text=item['text'], vector=None,
                            valid_from=revision['valid_from'],
                            source_session=item.get('source_session'), evidence=item.get('evidence'),
                        )
                    except ForgottenSourceError:
                        raise
                    except ValueError:
                        raise AdmissionError('revision_conflict', 409) from None
                    revision_receipt = {field: result[field] for field in ('action', 'supersedes')}
                else:
                    result = storage.merge_or_store(
                        vector=None,
                        _dedup_observer=lambda duration: record_stage_metric(
                            operation=operation, stage='dedup', duration_seconds=duration
                        ),
                        **item,
                    )
            except ForgottenSourceError:
                blocked += 1
                continue
            mid = result['id']
            ids.append(mid)
            stored += int(result['action'] == 'inserted')
            merged += int(result['action'] == 'merged')
            # Another identity sharing the same exact provenance reuses jobs.
            # Legacy rows are never rewritten or automatically queued here.
            if result['action'] == 'inserted':
                now = time.time()
                for stage in required:
                    conn.execute('INSERT OR IGNORE INTO memory_index_jobs VALUES (?,?,?,?,?,?,?,?,?)',
                                 (mid, stage, 'pending', 0, None, now, None, now, now))
        pending = conn.execute("SELECT count(*) FROM memory_index_jobs WHERE state IN ('pending','running')").fetchone()[0]
        if pending > queue_cap:
            raise AdmissionError('queue_full', 429)
        now = time.time()
        counts = {'stored': stored, 'merged': merged, 'blocked': blocked, 'total': total}
        receipt = dict(counts)
        if operation == 'store' and ids:
            receipt.update(id=ids[0], stored=bool(stored), merged=bool(merged), similarity=1.0 if merged else None)
            # Preserve the existing store response; status remains an explicit,
            # identifier-free projection and does not expose lineage metadata.
            receipt.update(revision_receipt)
        conn.execute('INSERT INTO memory_operations VALUES (?,?,?,?,?,?,?,?,?,?,?)',
                     (*key, fp, 'blocked' if blocked else 'accepted', 1,
                      json.dumps(receipt, separators=(',', ':')), json.dumps(list(dict.fromkeys(ids))),
                      'source_blocked' if blocked else None, now, now))
        storage.invalidate_search_cache()
        # Check inside the transaction so expiry rolls every new row back.
        if time.monotonic() >= deadline or not getattr(storage, '_request_valid', lambda: True)():
            raise AdmissionError('deadline_exceeded', 504)
        status = public_status(storage, *key)
        status['timing'] = {'persist_seconds': max(0.0, time.monotonic() - started)}
        conn.execute('UPDATE memory_operations SET state=?,updated_at=? WHERE namespace=? AND operation=? AND request_id=?',
                     (status['state'], time.time(), *key))
    return {**receipt, **status}


def block_linked_operations(conn, memory_id):
    """Called under the forgetting writer transaction before physical deletion."""
    conn.execute("UPDATE memory_index_jobs SET state='blocked',attempts=attempts+1,lease_until=NULL,error_code='source_blocked',updated_at=? WHERE memory_id=?", (time.time(), memory_id))
    conn.execute("""UPDATE memory_operations SET state='blocked',error_code='source_blocked',updated_at=?
                    WHERE EXISTS(SELECT 1 FROM json_each(memory_ids_json) WHERE value=?)""", (time.time(), memory_id))


def claim(storage, *, budget=30, max_attempts=3):
    now = time.time()
    with storage.transaction() as conn:
        conn.execute("""UPDATE memory_index_jobs SET state='failed',error_code='attempts_exhausted',lease_until=NULL,updated_at=?
                        WHERE attempts>=? AND (state='pending' OR (state='running' AND lease_until<=?))""", (now, max_attempts, now))
        row = conn.execute("""SELECT * FROM memory_index_jobs WHERE attempts<? AND
                   ((state='pending' AND next_attempt_at<=?) OR (state='running' AND lease_until<=?))
                   ORDER BY next_attempt_at,created_at LIMIT 1""", (max_attempts, now, now)).fetchone()
        if row is None:
            return None
        result = conn.execute("""UPDATE memory_index_jobs SET state='running',attempts=attempts+1,
                   lease_until=?,updated_at=? WHERE memory_id=? AND stage=? AND attempts=? AND state=?""",
                   (now + budget + 5, now, row['memory_id'], row['stage'], row['attempts'], row['state']))
        if result.rowcount != 1:
            return None
        return {**dict(row), 'attempts': row['attempts'] + 1, 'lease_until': now + budget + 5, 'state': 'running'}


def fenced_memory(storage, job):
    conn = storage._get_conn()
    row = conn.execute("SELECT * FROM memory_index_jobs WHERE memory_id=? AND stage=? AND attempts=? AND state='running' AND lease_until>?",
                       (job['memory_id'], job['stage'], job['attempts'], time.time())).fetchone()
    memory = storage.get_memory(job['memory_id'])
    if row is None or memory is None or memory.get('deleted_at') is not None or storage.source_is_forgotten(memory.get('source_session')):
        return None
    return memory


def finish(storage, job, *, vector=None, graph=None, error=None, max_attempts=3, deadline=None):
    with storage.transaction() as conn:
        memory = fenced_memory(storage, job)
        if memory is None or (deadline is not None and time.monotonic() >= deadline):
            return False
        if error is None:
            if vector is not None:
                storage.update_memory(job['memory_id'], vector=vector)
            if graph is not None:
                graph(memory)
            state, next_attempt = 'completed', time.time()
            storage.invalidate_search_cache()
        else:
            state = 'failed' if job['attempts'] >= max_attempts else 'pending'
            next_attempt = time.time() + min(2 ** job['attempts'], 8)
        conn.execute('''UPDATE memory_index_jobs SET state=?,lease_until=NULL,next_attempt_at=?,error_code=?,updated_at=?
                        WHERE memory_id=? AND stage=? AND attempts=?''',
                     (state, next_attempt, error, time.time(), job['memory_id'], job['stage'], job['attempts']))
        return True
