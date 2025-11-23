from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from app.graph import app_graph
from app.store import get_interview_state, save_interview_state, clear_interview_state
import traceback
import json

app = FastAPI()

@app.post("/chat")
async def chat_endpoint(request: Request):
    try:
        payload = await request.json()
        call_id = payload.get("call", {}).get("id", "default_session")
        
        # --- 1. ROBUST MESSAGE EXTRACTION ---
        user_message = ""
        if isinstance(payload.get("message"), dict): 
             user_message = payload.get("message", {}).get("content", "")
        elif isinstance(payload.get("message"), str):
             try:
                 parsed = json.loads(payload["message"])
                 user_message = parsed.get("message", payload["message"])
             except:
                 user_message = payload["message"]
        
        # Fallback for Vapi/Other formats
        if not user_message and "messages" in payload:
            user_message = payload["messages"][-1]["content"]
            
        if not user_message: user_message = "Start"

        # --- 2. STATE RETRIEVAL ---
        current_state = get_interview_state(call_id)

        # --- 3. ROBUST INITIALIZATION (THE FIX) ---
        if not current_state:
            print(f"🔵 STARTING NEW SESSION: {call_id}")
            
            # Detect Role from the FIRST message (Configuration Phase)
            role = "Unknown"
            msg_lower = user_message.lower()
            if "sales" in msg_lower or "sdr" in msg_lower: role = "SDR"
            elif "retail" in msg_lower: role = "Retail Associate"
            elif "software" in msg_lower or "engineer" in msg_lower: role = "Software Engineer"
            
            print(f"🔒 Role Detected: {role}")

            # INITIALIZE STATE
            # CRITICAL: 'messages' is empty! We do not pollute history with the setup text.
            current_state = {
                "messages": [], 
                "role": role, 
                "question_count": 0,
                "persona_detected": "Normal",
                "latest_evaluation": "Good",
                "is_finished": False,
                "feedback": None,
                "used_questions": [],
                "active_question": "",
                "retry_count": 0,
                "current_topic_depth": 0
            }
        
        else:
            # --- NORMAL TURN ---
            print(f"🟢 RESUMING: {call_id} | Role: {current_state['role']}")
            current_state["messages"].append({"role": "user", "content": user_message})

        # --- 4. EXECUTE GRAPH ---
        result = app_graph.invoke(current_state)
        
        # Safe extraction of bot response
        if result["messages"] and result["messages"][-1]["role"] == "assistant":
            bot_response = result["messages"][-1]["content"]
        else:
            bot_response = "I'm ready. Let's begin."

        # --- 5. PERSISTENCE ---
        save_interview_state(call_id, result) 

        # Append "Session Ended" only for the frontend response, don't save it to DB
        final_response = bot_response
        if result.get("is_finished"):
            final_response += " (Session Ended)"

        return JSONResponse({
            "results": [{"toolCallId": payload.get("toolCallId", "unknown"), "result": final_response}]
        })

    except Exception as e:
        print(f"🔴 ERROR: {traceback.format_exc()}")
        return JSONResponse({
            "results": [{"toolCallId": "error", "result": "I encountered an error. Let's try again."}]
        })

@app.get("/get-latest-feedback")
def get_latest_feedback(call_id: str = "default_session"):
    state = get_interview_state(call_id)
    if not state: return {"status": "No interview found"}
    if not state.get("is_finished"): return {"status": "Interview in progress"}
    
    return {
        "status": "Completed",
        "feedback": state.get("feedback", "No feedback generated."),
        "transcript": state["messages"]
    }