import sqlite3
import json
import os
import tempfile
from typing import Dict, Any, Optional

# STRATEGIC TECHNICAL DECISION:
# Serverless environments (Vercel/AWS) have read-only file systems.
# We use the system's temporary directory (always writable) to store the SQLite DB.
# This ensures state persists across "warm" reloads without needing an external DB.
DB_FILENAME = "interview_state_v3.db"
DB_PATH = os.path.join(tempfile.gettempdir(), DB_FILENAME)

def init_db():
    """Initialize the DB if it doesn't exist."""
    try:
        # Use context manager to ensure connection closes automatically
        with sqlite3.connect(DB_PATH) as conn:
            c = conn.cursor()
            c.execute('''CREATE TABLE IF NOT EXISTS sessions 
                         (session_id TEXT PRIMARY KEY, state_json TEXT)''')
            conn.commit()
        print(f"✅ Database initialized at: {DB_PATH}")
    except Exception as e:
        print(f"❌ DB Init Error: {e}")

# Initialize immediately on module load
init_db()

def get_interview_state(session_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve state from disk with thread safety."""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            c = conn.cursor()
            c.execute("SELECT state_json FROM sessions WHERE session_id = ?", (session_id,))
            row = c.fetchone()
            
        if row:
            return json.loads(row[0])
        return None
    except Exception as e:
        print(f"⚠️ Read Error for {session_id}: {e}")
        return None

def save_interview_state(session_id: str, state: Dict[str, Any]):
    """Persist state to disk immediately."""
    try:
        state_json = json.dumps(state)
        with sqlite3.connect(DB_PATH) as conn:
            c = conn.cursor()
            c.execute("INSERT OR REPLACE INTO sessions (session_id, state_json) VALUES (?, ?)", 
                      (session_id, state_json))
            conn.commit()
    except Exception as e:
        print(f"⚠️ Write Error for {session_id}: {e}")

def clear_interview_state(session_id: str):
    """Clean up after interview finishes."""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            c = conn.cursor()
            c.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
            conn.commit()
    except Exception as e:
        print(f"⚠️ Delete Error for {session_id}: {e}")