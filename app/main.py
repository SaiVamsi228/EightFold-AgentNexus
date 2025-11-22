from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from app.graph import app_graph
from app.store import get_interview_state, save_interview_state, clear_interview_state
import traceback
import json

app = FastAPI()

@app.get("/")
def home():
    return {"status": "Stateful Agent Active"}

@app.post("/chat")
async def chat_endpoint(request: Request):
    try:
        payload = await request.json()
        
        # 1. GET CALL ID (The Session Key)
        # Vapi sends 'call' object. We use 'id' as our unique key.
        call_id = payload.get("call", {}).get("id")
        
        # Fallback for testing tools that don't send call ID
        if not call_id:
            call_id = "test_session_default"

        # 2. GET USER MESSAGE
        user_message = ""
        # Check standard Vapi tool call
        if isinstance(payload.get("message"), dict): # Nested
             user_message = payload.get("message", {}).get("content", "")
             # Or inside toolCall args if Vapi configured that way
             if not user_message:
                 user_message = payload.get("message", {}).get("toolCall", {}).get("function", {}).get("arguments", {}).get("message", "")
        elif isinstance(payload.get("message"), str): # String
             try:
                 parsed = json.loads(payload["message"])
                 user_message = parsed.get("message", payload["message"])
             except:
                 user_message = payload["message"]
        
        # Fallback to Vapi transcript history
        if not user_message and "messages" in payload:
            user_message = payload["messages"][-1]["content"]
            
        if not user_message: 
            user_message = "Start Interview"

        # 3. RETRIEVE STATE FROM MEMORY (Stateful!)
        current_state = get_interview_state(call_id)

        # 4. INITIALIZE IF NEW CALL
        if not current_state:
            print(f"New Call Detected: {call_id}")
            
            # Detect Role
            role = "Software Engineer"
            if "sales" in user_message.lower(): role = "SDR"
            elif "retail" in user_message.lower(): role = "Retail Associate"
            
            current_state = {
                "messages": [{"role": "user", "content": user_message}],
                "role": role,
                "question_count": 0, # Start at 0
                "persona_detected": "Normal",
                "latest_evaluation": "Good",
                "is_finished": False,
                "feedback": None
            }
        else:
            print(f"Resuming Call {call_id} - Question Count: {current_state['question_count']}")
            # Append new user message to existing history
            current_state["messages"].append({"role": "user", "content": user_message})

        # 5. RUN GRAPH
        result = app_graph.invoke(current_state)
        
        # 6. EXTRACT RESPONSE
        bot_response = result["messages"][-1]["content"]

        # 7. SAVE STATE BACK TO MEMORY
        # We save the 'result' because it contains the updated question_count and messages
        if result.get("is_finished"):
            clear_interview_state(call_id) # Cleanup memory
            bot_response += " I've generated a PDF report."
        else:
            save_interview_state(call_id, result)

        return JSONResponse({
            "results": [{"toolCallId": payload.get("toolCallId", "unknown"), "result": bot_response}]
        })

    except Exception as e:
        print(f"CRITICAL ERROR: {traceback.format_exc()}")
        return JSONResponse({
            "results": [{"toolCallId": "error", "result": "I had a glitch. Let's continue."}]
        })