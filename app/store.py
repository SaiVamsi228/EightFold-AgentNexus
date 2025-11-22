# app/store.py
from typing import Dict, Any

# This Dictionary acts as our database.
# Key: call_id (string)
# Value: InterviewState (dict)
ACTIVE_INTERVIEWS: Dict[str, Any] = {}

def get_interview_state(call_id: str) -> Dict:
    """Retrieves state for a specific call, or None if new."""
    return ACTIVE_INTERVIEWS.get(call_id)

def save_interview_state(call_id: str, state: Dict):
    """Updates the state for a specific call."""
    ACTIVE_INTERVIEWS[call_id] = state

def clear_interview_state(call_id: str):
    """Removes state when interview ends."""
    if call_id in ACTIVE_INTERVIEWS:
        del ACTIVE_INTERVIEWS[call_id]