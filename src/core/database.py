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
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Messages Table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (chat_id) REFERENCES chats (id)
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

    def create_chat(self, title: str = "New Chat") -> str:
        """Creates a new chat session and returns its ID."""
        chat_id = str(uuid.uuid4())
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO chats (id, title, created_at) VALUES (?, ?, ?)",
            (chat_id, title, datetime.datetime.now())
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

    def add_message(self, chat_id: str, role: str, content: str):
        """Adds a message to a chat."""
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO messages (chat_id, role, content, created_at) VALUES (?, ?, ?, ?)",
            (chat_id, role, content, datetime.datetime.now())
        )
        conn.commit()
        conn.close()

    def get_chat_history(self, chat_id: str) -> List[Dict]:
        """Retrieves all messages for a specific chat."""
        conn = self._get_conn()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(
            "SELECT role, content FROM messages WHERE chat_id = ? ORDER BY created_at ASC",
            (chat_id,)
        )
        rows = cursor.fetchall()
        conn.close()
        return [{"role": row["role"], "content": row["content"]} for row in rows]

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
