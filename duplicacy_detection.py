
import hashlib
import sqlite3
from pathlib import Path

DB_PATH = Path("context_firewall.db")


def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS uploaded_files (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                sha256 TEXT NOT NULL UNIQUE,
                uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)


def check_and_register(file_bytes: bytes, filename: str) -> dict:
    # Generate a fingerprint from the file's exact bytes
    file_hash = hashlib.sha256(file_bytes).hexdigest()

    with sqlite3.connect(DB_PATH) as conn:
        existing = conn.execute(
            "SELECT id, filename FROM uploaded_files WHERE sha256 = ?",
            (file_hash,)
        ).fetchone()

        if existing:
            return {
                "is_duplicate": True,
                "file_hash": file_hash,
                "original_file_id": existing[0],
                "original_filename": existing[1],
                "message": "This file has already been uploaded."
            }

        conn.execute(
            "INSERT INTO uploaded_files (filename, sha256) VALUES (?, ?)",
            (filename, file_hash)
        )

        file_id = conn.execute(
            "SELECT id FROM uploaded_files WHERE sha256 = ?",
            (file_hash,)
        ).fetchone()[0]

        return {
            "is_duplicate": False,
            "file_hash": file_hash,
            "file_id": file_id,
            "message": "New file registered successfully."
        }


init_db()
