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
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    conn.commit()
    conn.close()

class ChatRepository:
    def __init__(self):
        self.db_path = DB_PATH
        init_db()

    def _get_conn(self):
        return sqlite3.connect(self.db_path)

    def create_chat(self, title: str = "New Chat", focused_file: Optional[str] = None) -> str:
        """Creates a new chat session and returns its ID."""
        chat_id = str(uuid.uuid4())
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO chats (id, title, focused_file, created_at) VALUES (?, ?, ?, ?)",
            (chat_id, title, focused_file, datetime.datetime.now())
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

    def get_chat(self, chat_id: str) -> Optional[Dict]:
        """Retrieves core chat metadata for a specific chat."""
        conn = self._get_conn()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, title, focused_file, created_at FROM chats WHERE id = ?",
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
            "SELECT id, title, created_at FROM chats ORDER BY created_at DESC LIMIT ?",
            (limit,)
        )
        rows = cursor.fetchall()
        conn.close()
        return [{"id": row["id"], "title": row["title"], "created_at": row["created_at"]} for row in rows]

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

    def add_watched_path(self, path: str, table_name: str, strategy: str = 'organize_and_ingest') -> str:
        """Adds a new watched path."""
        path_id = str(uuid.uuid4())
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO watched_paths (id, path, table_name, strategy, created_at) VALUES (?, ?, ?, ?, ?)",
            (path_id, path, table_name, strategy, datetime.datetime.now())
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
