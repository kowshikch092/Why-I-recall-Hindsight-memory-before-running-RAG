import sqlite3
from src.config import CHAT_DB_PATH


def get_connection():
    return sqlite3.connect(str(CHAT_DB_PATH))


def initialize_database():
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                customer_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()


def save_message(customer_id, role, content):
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO messages (customer_id, role, content)
            VALUES (?, ?, ?)
            """,
            (customer_id, role, content)
        )
        conn.commit()


def get_chat_history(customer_id):
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT role, content, created_at
            FROM messages
            WHERE customer_id = ?
            ORDER BY id ASC
            """,
            (customer_id,)
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
            FROM messages
            ORDER BY customer_id
            """
        ).fetchall()

    return [row[0] for row in rows]


def clear_chat_history(customer_id):
    with get_connection() as conn:
        conn.execute(
            "DELETE FROM messages WHERE customer_id = ?",
            (customer_id,)
        )
        conn.commit()