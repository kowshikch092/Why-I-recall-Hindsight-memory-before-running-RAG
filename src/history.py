import re
import sqlite3

from src.config import CHAT_DB_PATH


def get_connection():
    return sqlite3.connect(str(CHAT_DB_PATH))


def _table_has_column(conn, table, column):
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return any(row[1] == column for row in rows)


def initialize_database():
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS threads (
                thread_id TEXT PRIMARY KEY,
                customer_id TEXT NOT NULL,
                title TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        messages_table_exists = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='messages'"
        ).fetchone() is not None

        # One-time migration from the old schema, where "customer_id" on
        # the messages table doubled as both the customer identity and the
        # conversation identity, and was NOT NULL. SQLite can't drop a
        # NOT NULL constraint with ALTER TABLE, so the table is rebuilt
        # onto the new (thread_id-only) schema rather than patched in
        # place. Each distinct old customer_id becomes a legacy thread
        # (same value as both thread_id and customer_id) so no existing
        # conversation is lost or renamed.
        if messages_table_exists and _table_has_column(conn, "messages", "customer_id"):
            conn.execute("ALTER TABLE messages RENAME TO messages_legacy")

            conn.execute("""
                CREATE TABLE messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    thread_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            conn.execute("""
                INSERT INTO messages (thread_id, role, content, created_at)
                SELECT customer_id, role, content, created_at
                FROM messages_legacy
                ORDER BY id ASC
            """)

            legacy_customers = conn.execute(
                """
                SELECT DISTINCT customer_id
                FROM messages_legacy
                WHERE customer_id IS NOT NULL
                """
            ).fetchall()

            for (legacy_id,) in legacy_customers:
                existing = conn.execute(
                    "SELECT 1 FROM threads WHERE thread_id = ?",
                    (legacy_id,)
                ).fetchone()

                if existing:
                    continue

                first_message = conn.execute(
                    """
                    SELECT content, created_at FROM messages_legacy
                    WHERE customer_id = ? AND role = 'user'
                    ORDER BY id ASC LIMIT 1
                    """,
                    (legacy_id,)
                ).fetchone()

                title = (
                    _fallback_title(first_message[0])
                    if first_message else "Conversation"
                )
                created_at = first_message[1] if first_message else None

                conn.execute(
                    """
                    INSERT INTO threads (thread_id, customer_id, title, created_at)
                    VALUES (?, ?, ?, COALESCE(?, CURRENT_TIMESTAMP))
                    """,
                    (legacy_id, legacy_id, title, created_at)
                )

            conn.execute("DROP TABLE messages_legacy")
        else:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    thread_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

        conn.commit()


def _fallback_title(question, max_words=6):
    words = question.strip().split()
    title = " ".join(words[:max_words])
    return title or "Conversation"


def generate_thread_id():
    # thread_id is the primary key of the threads table (global, not
    # scoped per customer), so the next number must be computed from
    # every existing thread, not just this customer's. Scanning only
    # this customer's threads would hand out "Thread-001" to every new
    # customer's first thread and collide with whichever customer
    # already holds that id.
    with get_connection() as conn:
        rows = conn.execute("SELECT thread_id FROM threads").fetchall()

    max_n = 0
    for (thread_id,) in rows:
        match = re.match(r"^Thread-(\d+)$", thread_id)
        if match:
            max_n = max(max_n, int(match.group(1)))

    return f"Thread-{max_n + 1:03d}"


def create_thread(customer_id, title=None):
    with get_connection() as conn:
        for _ in range(20):
            thread_id = generate_thread_id()
            try:
                conn.execute(
                    """
                    INSERT INTO threads (thread_id, customer_id, title)
                    VALUES (?, ?, ?)
                    """,
                    (thread_id, customer_id, title or "New Conversation")
                )
                conn.commit()
                return thread_id
            except sqlite3.IntegrityError:
                # thread_id was taken by a concurrent write; retry with
                # the next number instead of overwriting the existing one.
                continue

    raise RuntimeError(
        f"Could not allocate a unique thread id for customer {customer_id}"
    )


def update_thread_title(thread_id, title):
    with get_connection() as conn:
        conn.execute(
            "UPDATE threads SET title = ? WHERE thread_id = ?",
            (title, thread_id)
        )
        conn.commit()


def get_customer_threads(customer_id):
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT thread_id, title, created_at, updated_at
            FROM threads
            WHERE customer_id = ?
            ORDER BY updated_at DESC, created_at DESC
            """,
            (customer_id,)
        ).fetchall()

    return [
        {
            "thread_id": row[0],
            "title": row[1] or "New Conversation",
            "created_at": row[2],
            "updated_at": row[3]
        }
        for row in rows
    ]


def get_latest_thread_for_customer(customer_id):
    threads = get_customer_threads(customer_id)
    return threads[0]["thread_id"] if threads else None


def save_message(thread_id, role, content):
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO messages (thread_id, role, content)
            VALUES (?, ?, ?)
            """,
            (thread_id, role, content)
        )
        conn.execute(
            "UPDATE threads SET updated_at = CURRENT_TIMESTAMP WHERE thread_id = ?",
            (thread_id,)
        )
        conn.commit()


def get_chat_history(thread_id):
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT role, content, created_at
            FROM messages
            WHERE thread_id = ?
            ORDER BY id ASC
            """,
            (thread_id,)
        ).fetchall()

    return [
        {
            "role": row[0],
            "content": row[1],
            "created_at": row[2]
        }
        for row in rows
    ]


def get_all_customers():
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT DISTINCT customer_id
            FROM threads
            ORDER BY customer_id
            """
        ).fetchall()

    return [row[0] for row in rows]


def clear_chat_history(customer_id):
    with get_connection() as conn:
        conn.execute(
            """
            DELETE FROM messages
            WHERE thread_id IN (
                SELECT thread_id FROM threads WHERE customer_id = ?
            )
            """,
            (customer_id,)
        )
        conn.execute(
            "DELETE FROM threads WHERE customer_id = ?",
            (customer_id,)
        )
        conn.commit()
