import sqlite3
import json
from typing import Dict, Any, Optional

DB_FILE = "interview_state.db"

def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS sessions 
                 (session_id TEXT PRIMARY KEY, state_json TEXT)''')
    conn.commit()
    conn.close()

init_db()

def get_interview_state(session_id: str) -> Optional[Dict[str, Any]]:
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT state_json FROM sessions WHERE session_id = ?", (session_id,))
    row = c.fetchone()
    conn.close()
    if row:
        return json.loads(row[0])
    return None

def save_interview_state(session_id: str, state: Dict[str, Any]):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    state_json = json.dumps(state)
    c.execute("INSERT OR REPLACE INTO sessions (session_id, state_json) VALUES (?, ?)", 
              (session_id, state_json))
    conn.commit()
    conn.close()

def clear_interview_state(session_id: str):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
    conn.commit()
    conn.close()