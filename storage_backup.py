
import sqlite3
import json
import hashlib
from contextlib import contextmanager

DB_PATH = "invoice_agent.db"


@contextmanager
def get_db():
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row

    try:
        conn.execute("PRAGMA busy_timeout = 30000")
        conn.execute("PRAGMA foreign_keys = ON")
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db():
    with get_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS tasks (
                task_id TEXT PRIMARY KEY,
                principal TEXT NOT NULL,
                context_id TEXT NOT NULL,
                batch_id TEXT NOT NULL,
                state TEXT NOT NULL,
                task_json TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(principal, task_id)
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                principal TEXT NOT NULL,
                message_id TEXT NOT NULL,
                message_hash TEXT NOT NULL,
                task_id TEXT NOT NULL,
                response_json TEXT NOT NULL,
                PRIMARY KEY(principal, message_id),
                FOREIGN KEY(principal, task_id)
                    REFERENCES tasks(principal, task_id)
            )
        """)


def canonical_message_hash(message):
    canonical = json.dumps(
        message,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def get_saved_message(principal, message_id):
    with get_db() as conn:
        row = conn.execute("""
            SELECT message_hash, task_id, response_json
            FROM messages
            WHERE principal = ? AND message_id = ?
        """, (principal, message_id)).fetchone()

        return dict(row) if row else None


def get_task(principal, task_id):
    with get_db() as conn:
        row = conn.execute("""
            SELECT task_json
            FROM tasks
            WHERE principal = ? AND task_id = ?
        """, (principal, task_id)).fetchone()

        return json.loads(row["task_json"]) if row else None


def list_tasks(principal):
    with get_db() as conn:
        rows = conn.execute("""
            SELECT task_json
            FROM tasks
            WHERE principal = ?
            ORDER BY created_at DESC, task_id
        """, (principal,)).fetchall()

        return [json.loads(row["task_json"]) for row in rows]


def create_task_if_new(
    principal,
    message_id,
    message_hash,
    task,
    response,
    batch_id,
):
    """
    Atomically create a Task and save its original response.

    Returns:
        ("created", response) for a new message
        ("replay", original_response) for an identical retry
        ("conflict", None) if the message ID was reused
        ("task_conflict", None) if the task ID already exists
    """

    task_json = json.dumps(
        task,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    response_json = json.dumps(
        response,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )

    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row

    try:
        conn.execute("PRAGMA busy_timeout = 30000")
        conn.execute("BEGIN IMMEDIATE")

        existing = conn.execute("""
            SELECT message_hash, response_json
            FROM messages
            WHERE principal = ? AND message_id = ?
        """, (principal, message_id)).fetchone()

        if existing:
            conn.rollback()

            if existing["message_hash"] != message_hash:
                return "conflict", None

            return "replay", json.loads(existing["response_json"])

        conn.execute("""
            INSERT INTO tasks (
                task_id, principal, context_id,
                batch_id, state, task_json
            )
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            task["id"],
            principal,
            task["contextId"],
            batch_id,
            task["status"]["state"],
            task_json,
        ))

        conn.execute("""
            INSERT INTO messages (
                principal, message_id, message_hash,
                task_id, response_json
            )
            VALUES (?, ?, ?, ?, ?)
        """, (
            principal,
            message_id,
            message_hash,
            task["id"],
            response_json,
        ))

        conn.commit()
        return "created", response

    except sqlite3.IntegrityError:
        conn.rollback()
        return "task_conflict", None

    finally:
        conn.close()


def update_task(principal, task):
    """
    Persist an updated Task only if the authenticated principal owns it.
    """
    task_json = json.dumps(
        task,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )

    with get_db() as conn:
        cursor = conn.execute("""
            UPDATE tasks
            SET context_id = ?,
                batch_id = ?,
                state = ?,
                task_json = ?
            WHERE principal = ? AND task_id = ?
        """, (
            task["contextId"],
            task.get("batchId", ""),
            task["status"]["state"],
            task_json,
            principal,
            task["id"],
        ))

        return cursor.rowcount == 1