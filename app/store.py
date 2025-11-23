import sqlite3
import json
import os
import tempfile
from typing import Dict, Any, Optional

# Use temp directory so it works on Read-Only file systems like Vercel/Render
DB_PATH = os.path.join(tempfile.gettempdir(), "interview_sessions.db")

def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute('''CREATE TABLE IF NOT EXISTS sessions 
                        (session_id TEXT PRIMARY KEY, state_json TEXT)''')

init_db()

def get_interview_state(session_id: str) -> Optional[Dict[str, Any]]:
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.execute("SELECT state_json FROM sessions WHERE session_id = ?", (session_id,))
            row = cursor.fetchone()
            return json.loads(row[0]) if row else None
    except Exception as e:
        print(f"DB Read Error: {e}")
        return None

def save_interview_state(session_id: str, state: Dict[str, Any]):
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("INSERT OR REPLACE INTO sessions (session_id, state_json) VALUES (?, ?)", 
                         (session_id, json.dumps(state)))
    except Exception as e:
        print(f"DB Write Error: {e}")

def clear_interview_state(session_id: str):
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
    except Exception as e:
        print(f"DB Delete Error: {e}")