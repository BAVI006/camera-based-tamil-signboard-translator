"""Small SQLite-backed history store; photos are never retained."""

from datetime import datetime, timezone
from contextlib import closing
import json
import os
from pathlib import Path
import sqlite3
from typing import Any


DEFAULT_HISTORY_PATH = (
    Path(os.getenv("LOCALAPPDATA", str(Path.home())))
    / "SignSenseAI"
    / "history.sqlite3"
)


class HistoryService:
    def __init__(self, database_path: Path | None = None):
        self.database_path = database_path

    def _path(self) -> Path:
        configured = os.getenv("SIGN_HISTORY_DB")
        return Path(configured) if configured else (
            self.database_path or DEFAULT_HISTORY_PATH
        )

    def _connect(self) -> sqlite3.Connection:
        path = self._path()
        path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS sign_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                file_name TEXT NOT NULL,
                result_json TEXT NOT NULL
            )
            """
        )
        return connection

    def save(self, file_name: str, result: dict[str, Any]) -> dict[str, Any]:
        created_at = datetime.now(timezone.utc).isoformat()
        with closing(self._connect()) as connection:
            with connection:
                cursor = connection.execute(
                    "INSERT INTO sign_history (created_at, file_name, result_json) VALUES (?, ?, ?)",
                    (
                        created_at,
                        Path(file_name or "signboard").name[:255],
                        json.dumps(result, ensure_ascii=False),
                    ),
                )
                item_id = cursor.lastrowid
        return {
            "id": item_id,
            "created_at": created_at,
            "file_name": Path(file_name or "signboard").name[:255],
            "result": result,
        }

    def list_recent(self, limit: int = 50) -> list[dict[str, Any]]:
        if not 1 <= limit <= 100:
            raise ValueError("History limit must be between 1 and 100.")
        with closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT id, created_at, file_name, result_json "
                "FROM sign_history ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [
            {
                "id": row["id"],
                "created_at": row["created_at"],
                "file_name": row["file_name"],
                "result": json.loads(row["result_json"]),
            }
            for row in rows
        ]

    def clear(self) -> None:
        with closing(self._connect()) as connection:
            with connection:
                connection.execute("DELETE FROM sign_history")
