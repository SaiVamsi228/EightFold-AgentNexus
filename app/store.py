from typing import Dict, Any, Optional

# In-memory storage for demo purposes
# In production, use Redis or a database
interview_sessions: Dict[str, Any] = {}

def get_interview_state(session_id: str) -> Optional[Dict[str, Any]]:
    return interview_sessions.get(session_id)

def save_interview_state(session_id: str, state: Dict[str, Any]):
    interview_sessions[session_id] = state

def clear_interview_state(session_id: str):
    if session_id in interview_sessions:
        del interview_sessions[session_id]