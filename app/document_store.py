import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


class DocumentStore:
    def __init__(self, database_path: Path):
        self.database_path = database_path
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    document_id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    file_type TEXT NOT NULL,
                    chunk_count INTEGER NOT NULL,
                    chunks_json TEXT NOT NULL,
                    original_file BLOB,
                    created_at TEXT NOT NULL
                )
                """
            )
            document_columns = {
                row["name"]
                for row in connection.execute(
                    "PRAGMA table_info(documents)"
                ).fetchall()
            }
            if "original_file" not in document_columns:
                connection.execute(
                    "ALTER TABLE documents ADD COLUMN original_file BLOB"
                )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    document_id TEXT NOT NULL,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    sources_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (document_id)
                        REFERENCES documents (document_id)
                        ON DELETE CASCADE
                )
                """
            )

    def save(
        self,
        document_id: str,
        filename: str,
        file_type: str,
        chunks: list[dict],
        original_file: bytes | None = None,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO documents (
                    document_id, filename, file_type, chunk_count,
                    chunks_json, original_file, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    document_id,
                    filename,
                    file_type,
                    len(chunks),
                    json.dumps(chunks),
                    original_file,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )

    def get(self, document_id: str) -> dict | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM documents WHERE document_id = ?",
                (document_id,),
            ).fetchone()

        if row is None:
            return None

        return {
            "document_id": row["document_id"],
            "filename": row["filename"],
            "file_type": row["file_type"],
            "chunk_count": row["chunk_count"],
            "chunks": json.loads(row["chunks_json"]),
            "created_at": row["created_at"],
        }

    def get_original_file(self, document_id: str) -> dict | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT filename, file_type, original_file
                FROM documents
                WHERE document_id = ?
                """,
                (document_id,),
            ).fetchone()

        if row is None:
            return None

        return {
            "filename": row["filename"],
            "file_type": row["file_type"],
            "content": row["original_file"],
        }

    def list_documents(self) -> list[dict]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT document_id, filename, file_type, chunk_count, created_at
                FROM documents
                ORDER BY created_at DESC
                """
            ).fetchall()

        return [dict(row) for row in rows]

    def save_conversation(
        self,
        document_id: str,
        question: str,
        answer: str,
        sources: list[dict],
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO conversations (
                    document_id, question, answer, sources_json, created_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    document_id,
                    question,
                    answer,
                    json.dumps(sources),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )

    def list_conversations(self, document_id: str) -> list[dict]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, question, answer, sources_json, created_at
                FROM conversations
                WHERE document_id = ?
                ORDER BY id
                """,
                (document_id,),
            ).fetchall()

        return [
            {
                "id": row["id"],
                "question": row["question"],
                "answer": row["answer"],
                "sources": json.loads(row["sources_json"]),
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    def clear_conversations(self, document_id: str) -> bool:
        with self._connect() as connection:
            document = connection.execute(
                "SELECT 1 FROM documents WHERE document_id = ?",
                (document_id,),
            ).fetchone()
            if document is None:
                return False

            connection.execute(
                "DELETE FROM conversations WHERE document_id = ?",
                (document_id,),
            )

        return True

    def delete(self, document_id: str) -> bool:
        with self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM documents WHERE document_id = ?",
                (document_id,),
            )

        return cursor.rowcount > 0
