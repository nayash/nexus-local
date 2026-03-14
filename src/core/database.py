import sqlite3
import uuid
import datetime
from typing import List, Dict, Optional
import os
from src.core.config import Config

DB_PATH = Config.SQLITE_PATH

def init_db():
    """Initializes the SQLite database and creates tables if they don't exist."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Chats Table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS chats (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            focused_file TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Backward-compatible migration for existing DBs created before focused_file existed.
    cursor.execute("PRAGMA table_info(chats)")
    chat_columns = [row[1] for row in cursor.fetchall()]
    if "focused_file" not in chat_columns:
        cursor.execute("ALTER TABLE chats ADD COLUMN focused_file TEXT")
    if "workspace_id" not in chat_columns:
        cursor.execute("ALTER TABLE chats ADD COLUMN workspace_id TEXT")
    
    # Messages Table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            reasoning_content TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (chat_id) REFERENCES chats (id)
        )
    ''')

    # Backward-compatible migration for older DBs without reasoning_content.
    cursor.execute("PRAGMA table_info(messages)")
    message_columns = [row[1] for row in cursor.fetchall()]
    if "reasoning_content" not in message_columns:
        cursor.execute("ALTER TABLE messages ADD COLUMN reasoning_content TEXT")

    # Watched Paths Table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS watched_paths (
            id TEXT PRIMARY KEY,
            path TEXT NOT NULL,
            table_name TEXT NOT NULL,
            status TEXT DEFAULT 'active',
            last_scan REAL,
            strategy TEXT DEFAULT 'organize_and_ingest',
            workspace_id TEXT,
            watch_mode TEXT DEFAULT 'organize_and_ingest',
            recursive INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    cursor.execute("PRAGMA table_info(watched_paths)")
    watched_columns = [row[1] for row in cursor.fetchall()]
    if "workspace_id" not in watched_columns:
        cursor.execute("ALTER TABLE watched_paths ADD COLUMN workspace_id TEXT")
    if "watch_mode" not in watched_columns:
        cursor.execute("ALTER TABLE watched_paths ADD COLUMN watch_mode TEXT DEFAULT 'organize_and_ingest'")
    if "recursive" not in watched_columns:
        cursor.execute("ALTER TABLE watched_paths ADD COLUMN recursive INTEGER DEFAULT 0")

    cursor.execute(
        '''
        CREATE TABLE IF NOT EXISTS workspaces (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            root_path TEXT NOT NULL UNIQUE,
            status TEXT DEFAULT 'active',
            ingest_status TEXT DEFAULT 'pending',
            last_ingested_at TIMESTAMP,
            last_watched_at TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    '''
    )
    
    conn.commit()
    conn.close()

class ChatRepository:
    def __init__(self):
        self.db_path = DB_PATH
        init_db()

    def _get_conn(self):
        return sqlite3.connect(self.db_path)

    def create_chat(
        self,
        title: str = "New Chat",
        focused_file: Optional[str] = None,
        workspace_id: Optional[str] = None,
    ) -> str:
        """Creates a new chat session and returns its ID."""
        chat_id = str(uuid.uuid4())
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO chats (id, title, focused_file, workspace_id, created_at) VALUES (?, ?, ?, ?, ?)",
            (chat_id, title, focused_file, workspace_id, datetime.datetime.now())
        )
        conn.commit()
        conn.close()
        return chat_id

    def update_chat_title(self, chat_id: str, title: str):
        """Updates the title of a chat."""
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE chats SET title = ? WHERE id = ?",
            (title, chat_id)
        )
        conn.commit()
        conn.close()

    def update_chat_focused_file(self, chat_id: str, focused_file: Optional[str]):
        """Updates the focused file associated with a chat."""
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE chats SET focused_file = ? WHERE id = ?",
            (focused_file, chat_id)
        )
        conn.commit()
        conn.close()

    def update_chat_workspace(self, chat_id: str, workspace_id: Optional[str]):
        """Updates the workspace associated with a chat."""
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE chats SET workspace_id = ? WHERE id = ?",
            (workspace_id, chat_id)
        )
        conn.commit()
        conn.close()

    def get_chat(self, chat_id: str) -> Optional[Dict]:
        """Retrieves core chat metadata for a specific chat."""
        conn = self._get_conn()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, title, focused_file, workspace_id, created_at FROM chats WHERE id = ?",
            (chat_id,)
        )
        row = cursor.fetchone()
        conn.close()
        return dict(row) if row else None

    def add_message(self, chat_id: str, role: str, content: str, reasoning_content: Optional[str] = None):
        """Adds a message to a chat."""
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO messages (chat_id, role, content, reasoning_content, created_at) VALUES (?, ?, ?, ?, ?)",
            (chat_id, role, content, reasoning_content, datetime.datetime.now())
        )
        conn.commit()
        conn.close()

    def get_chat_history(self, chat_id: str, limit: Optional[int] = None) -> List[Dict]:
        """Retrieves messages for a specific chat, optionally limited to most recent ones."""
        conn = self._get_conn()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        query = "SELECT role, content FROM messages WHERE chat_id = ? ORDER BY created_at ASC"
        params = [chat_id]
        
        if limit:
            # We want the MOST RECENT messages, but in ASCENDING order for LangChain.
            # So we select recent ones in DESC, then wrap it to sort back to ASC.
            query = f"""
                SELECT role, content FROM (
                    SELECT role, content, created_at 
                    FROM messages 
                    WHERE chat_id = ? 
                    ORDER BY created_at DESC 
                    LIMIT ?
                ) ORDER BY created_at ASC
            """
            params.append(limit)
            
        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()
        return [{"role": row["role"], "content": row["content"]} for row in rows]

    def get_last_assistant_reasoning(self, chat_id: str) -> Optional[str]:
        """Returns the latest non-empty assistant reasoning summary for a chat."""
        conn = self._get_conn()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT reasoning_content
            FROM messages
            WHERE chat_id = ?
              AND role = 'assistant'
              AND reasoning_content IS NOT NULL
              AND TRIM(reasoning_content) != ''
            ORDER BY created_at DESC, id DESC
            LIMIT 1
            """,
            (chat_id,),
        )
        row = cursor.fetchone()
        conn.close()
        if not row:
            return None
        return row["reasoning_content"]

    def get_recent_chats(self, limit: int = 50) -> List[Dict]:
        """Retrieves a list of recent chats."""
        conn = self._get_conn()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, title, focused_file, workspace_id, created_at FROM chats ORDER BY created_at DESC LIMIT ?",
            (limit,)
        )
        rows = cursor.fetchall()
        conn.close()
        return [
            {
                "id": row["id"],
                "title": row["title"],
                "focused_file": row["focused_file"],
                "workspace_id": row["workspace_id"],
                "created_at": row["created_at"],
            }
            for row in rows
        ]


class WorkspaceRepository:
    def __init__(self):
        self.db_path = DB_PATH
        init_db()

    def _get_conn(self):
        return sqlite3.connect(self.db_path)

    def create_workspace(self, name: str, root_path: str, ingest_status: str = "pending") -> str:
        workspace_id = str(uuid.uuid4())
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO workspaces (
                id, name, root_path, status, ingest_status, created_at
            ) VALUES (?, ?, ?, 'active', ?, ?)
            """,
            (workspace_id, name, root_path, ingest_status, datetime.datetime.now()),
        )
        conn.commit()
        conn.close()
        return workspace_id

    def get_workspace(self, workspace_id: str) -> Optional[Dict]:
        conn = self._get_conn()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM workspaces WHERE id = ?", (workspace_id,))
        row = cursor.fetchone()
        conn.close()
        return dict(row) if row else None

    def get_workspace_by_path(self, root_path: str) -> Optional[Dict]:
        conn = self._get_conn()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM workspaces WHERE root_path = ?", (root_path,))
        row = cursor.fetchone()
        conn.close()
        return dict(row) if row else None

    def list_workspaces(self) -> List[Dict]:
        conn = self._get_conn()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM workspaces ORDER BY created_at DESC")
        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]

    def update_workspace_status(
        self,
        workspace_id: str,
        *,
        ingest_status: Optional[str] = None,
        status: Optional[str] = None,
        last_ingested_at: Optional[datetime.datetime] = None,
        last_watched_at: Optional[datetime.datetime] = None,
    ):
        fields = []
        params = []
        if ingest_status is not None:
            fields.append("ingest_status = ?")
            params.append(ingest_status)
        if status is not None:
            fields.append("status = ?")
            params.append(status)
        if last_ingested_at is not None:
            fields.append("last_ingested_at = ?")
            params.append(last_ingested_at)
        if last_watched_at is not None:
            fields.append("last_watched_at = ?")
            params.append(last_watched_at)
        if not fields:
            return

        params.append(workspace_id)
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute(f"UPDATE workspaces SET {', '.join(fields)} WHERE id = ?", params)
        conn.commit()
        conn.close()

    def delete_chat(self, chat_id: str):
        """Deletes a chat and its messages."""
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM messages WHERE chat_id = ?", (chat_id,))
        cursor.execute("DELETE FROM chats WHERE id = ?", (chat_id,))
        conn.commit()
        conn.close()

    def clear_all_chats(self):
        """Deletes all chats and messages from the database."""
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM messages")
        cursor.execute("DELETE FROM chats")
        conn.commit()
        conn.close()


class WatchedPathsRepository:
    def __init__(self):
        self.db_path = DB_PATH
        init_db()

    def _get_conn(self):
        return sqlite3.connect(self.db_path)

    def add_watched_path(
        self,
        path: str,
        table_name: str,
        strategy: str = 'organize_and_ingest',
        workspace_id: Optional[str] = None,
        watch_mode: str = "organize_and_ingest",
        recursive: bool = False,
    ) -> str:
        """Adds a new watched path."""
        path_id = str(uuid.uuid4())
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO watched_paths (
                id, path, table_name, strategy, workspace_id, watch_mode, recursive, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (path_id, path, table_name, strategy, workspace_id, watch_mode, int(bool(recursive)), datetime.datetime.now())
        )
        conn.commit()
        conn.close()
        return path_id

    def get_watched_paths(self) -> List[Dict]:
        """Retrieves all watched paths."""
        conn = self._get_conn()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM watched_paths")
        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]

    def remove_watched_path(self, path_id: str):
        """Removes a watched path."""
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM watched_paths WHERE id = ?", (path_id,))
        conn.commit()
        conn.close()

    def get_watched_path_by_path(self, path: str) -> Optional[Dict]:
        conn = self._get_conn()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM watched_paths WHERE path = ? LIMIT 1", (path,))
        row = cursor.fetchone()
        conn.close()
        return dict(row) if row else None

    def update_watched_path_workspace(self, path_id: str, workspace_id: Optional[str]):
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE watched_paths SET workspace_id = ? WHERE id = ?",
            (workspace_id, path_id),
        )
        conn.commit()
        conn.close()

    def update_last_scan(self, path_id: str):
        """Updates the last scan timestamp."""
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE watched_paths SET last_scan = ? WHERE id = ?",
            (datetime.datetime.now().timestamp(), path_id)
        )
        conn.commit()
        conn.close()
