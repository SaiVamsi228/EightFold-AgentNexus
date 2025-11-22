from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from app.graph import app_graph
from app.store import get_interview_state, save_interview_state, clear_interview_state
import traceback
import json

app = FastAPI()

@app.get("/")
def home():
    return {"status": "Eightfold AI Agent Active [Robust Mode]"}

@app.post("/chat")
async def chat_endpoint(request: Request):
    try:
        payload = await request.json()
        
        # 1. GET CALL ID
        # Use 'id' from call object, or fallback to a test session if testing via Postman/Curl
        call_id = payload.get("call", {}).get("id", "demo_session_v1")
        
        # 2. EXTRACT MESSAGE
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

        # 3. RETRIEVE STATE
        current_state = get_interview_state(call_id)

        # 4. INITIALIZE IF NEW
        if not current_state:
            print(f"🔵 New Session: {call_id}")
            role = "Software Engineer"
            if "sales" in user_message.lower(): role = "SDR"
            elif "retail" in user_message.lower(): role = "Retail Associate"
            
            current_state = {
                "messages": [{"role": "user", "content": user_message}],
                "role": role,
                "question_count": 0,
                "persona_detected": "Normal",
                "latest_evaluation": "Good",
                "is_finished": False,
                "feedback": None,
                "used_questions": [],
                "active_question": "" 
            }
        else:
            print(f"🟢 Resuming Session {call_id} | Q: {current_state['question_count']}")
            current_state["messages"].append({"role": "user", "content": user_message})

        # 5. EXECUTE GRAPH
        result = app_graph.invoke(current_state)
        bot_response = result["messages"][-1]["content"]

        # 6. SAVE STATE
        if result.get("is_finished"):
            clear_interview_state(call_id)
            bot_response += " (Session Ended)"
        else:
            save_interview_state(call_id, result)

        return JSONResponse({
            "results": [{"toolCallId": payload.get("toolCallId", "unknown"), "result": bot_response}]
        })

    except Exception as e:
        print(f"🔴 ERROR: {traceback.format_exc()}")
        return JSONResponse({
            "results": [{"toolCallId": "error", "result": "I encountered a technical glitch. Please say that again."}]
        })