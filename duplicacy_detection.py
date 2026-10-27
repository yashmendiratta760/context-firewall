
import hashlib
import sqlite3
from pathlib import Path

DB_DIR = Path("chat_databases")
DB_DIR.mkdir(exist_ok=True)


def get_db_path(chat_id: str) -> Path:
    """Get a separate database for each chat."""
    safe_chat_id = hashlib.sha256(
        chat_id.encode("utf-8")
    ).hexdigest()

    return DB_DIR / f"{safe_chat_id}.db"


def init_db(chat_id: str):
    db_path = get_db_path(chat_id)

    with sqlite3.connect(db_path) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS uploaded_files (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                sha256 TEXT NOT NULL UNIQUE,
                uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)


def check_and_register(
    file_bytes: bytes,
    filename: str,
    chat_id: str
) -> dict:

    db_path = get_db_path(chat_id)
    init_db(chat_id)

    file_hash = hashlib.sha256(file_bytes).hexdigest()

    with sqlite3.connect(db_path) as conn:
        existing = conn.execute(
            """
            SELECT id, filename
            FROM uploaded_files
            WHERE sha256 = ?
            """,
            (file_hash,)
        ).fetchone()

        if existing:
            return {
                "is_duplicate": True,
                "file_hash": file_hash,
                "original_file_id": existing[0],
                "original_filename": existing[1],
                "message": "This file has already been uploaded in this chat."
            }

        cursor = conn.execute(
            """
            INSERT INTO uploaded_files (filename, sha256)
            VALUES (?, ?)
            """,
            (filename, file_hash)
        )

        return {
            "is_duplicate": False,
            "file_hash": file_hash,
            "file_id": cursor.lastrowid,
            "message": "New file registered successfully."
        }
