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
        call_id = payload.get("call", {}).get("id", "demo_session_final")
        
        # Message Extraction
        user_message = ""
        if isinstance(payload.get("message"), dict): 
             user_message = payload.get("message", {}).get("content", "")
        elif isinstance(payload.get("message"), str):
             try:
                 parsed = json.loads(payload["message"])
                 user_message = parsed.get("message", payload["message"])
             except:
                 user_message = payload["message"]
        
        if not user_message and "messages" in payload:
            user_message = payload["messages"][-1]["content"]
            
        if not user_message: 
            user_message = "Start Interview"

        # Retrieve State
        current_state = get_interview_state(call_id)

        if not current_state:
            print(f"🔵 STARTING NEW SESSION: {call_id}")
            
            # ATTEMPT INITIAL ROLE DETECTION
            role = "Unknown"
            msg_lower = user_message.lower()
            if "sales" in msg_lower or "sdr" in msg_lower: role = "SDR"
            elif "retail" in msg_lower: role = "Retail Associate"
            elif "engineer" in msg_lower or "developer" in msg_lower: role = "Software Engineer"
            
            current_state = {
                "messages": [{"role": "user", "content": user_message}],
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
            print(f"🟢 RESUMING: {call_id} | Role: {current_state['role']}")
            current_state["messages"].append({"role": "user", "content": user_message})

        # Execute Graph
        result = app_graph.invoke(current_state)
        bot_response = result["messages"][-1]["content"]

        # Handle Finish
        if result.get("is_finished"):
            # Don't clear immediately if you want to read feedback later, 
            # but usually we want to keep it for the feedback endpoint.
            # We'll rely on the frontend calling /get-latest-feedback
            save_interview_state(call_id, result) 
            bot_response += " (Session Ended)"
        else:
            save_interview_state(call_id, result)

        return JSONResponse({
            "results": [{"toolCallId": payload.get("toolCallId", "unknown"), "result": bot_response}]
        })

    except Exception as e:
        print(f"🔴 ERROR: {traceback.format_exc()}")
        return JSONResponse({
            "results": [{"toolCallId": "error", "result": "I had a glitch. Let's continue."}]
        })

@app.get("/get-latest-feedback")
def get_latest_feedback(call_id: str = "demo_session_final"):
    state = get_interview_state(call_id)
    
    if not state:
        return {"status": "No interview found"}
        
    if not state.get("is_finished"):
        return {"status": "Interview in progress"}
        
    return {
        "status": "Completed",
        "feedback": state.get("feedback", "No feedback generated."),
        "transcript": state["messages"]
    }