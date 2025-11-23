from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from app.graph import app_graph, generate_feedback
from app.store import get_interview_state, save_interview_state
import traceback
import json

app = FastAPI()

# --- 1. CORS CONFIGURATION (FIXES YOUR ERROR) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins (Azure, Vercel, Localhost)
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (POST, GET, etc.)
    allow_headers=["*"],  # Allows all headers
)

@app.get("/")
def home():
    return {"status": "Eightfold AI Agent Active"}

@app.post("/chat")
async def chat_endpoint(request: Request):
    try:
        payload = await request.json()
        call_id = payload.get("call", {}).get("id", "default_session")
        
        # --- ROBUST MESSAGE EXTRACTION ---
        user_message = ""
        # Handle Vapi's complex tool call structure
        if isinstance(payload.get("message"), dict):
             # Try standard content first
             user_message = payload.get("message", {}).get("content", "")
             # If empty, check tool call arguments (Vapi specific)
             if not user_message:
                 user_message = payload.get("message", {}).get("toolCall", {}).get("function", {}).get("arguments", {}).get("message", "")
        elif isinstance(payload.get("message"), str):
             try:
                 parsed = json.loads(payload["message"])
                 user_message = parsed.get("message", payload["message"])
             except:
                 user_message = payload["message"]
        
        # Fallback
        if not user_message and "messages" in payload:
            user_message = payload["messages"][-1]["content"]
            
        if not user_message: user_message = "Start"

        # --- STATE RETRIEVAL ---
        current_state = get_interview_state(call_id)

        # --- INITIALIZATION LOGIC ---
        if not current_state:
            print(f"🔵 STARTING NEW SESSION: {call_id}")
            
            # 1. Detect Role from Configuration Input
            role = "Unknown"
            msg_lower = user_message.lower()
            if "sales" in msg_lower or "sdr" in msg_lower: role = "SDR"
            elif "retail" in msg_lower: role = "Retail Associate"
            elif "software" in msg_lower or "engineer" in msg_lower: role = "Software Engineer"
            
            print(f"🔒 Role Detected: {role}")

            # 2. Initialize Empty State (Prevents 'Dense Question' Bug)
            current_state = {
                "messages": [], # Empty start so bot speaks first
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

        # --- EXECUTE GRAPH ---
        result = app_graph.invoke(current_state)
        
        # Extract Bot Response
        if result["messages"] and result["messages"][-1]["role"] == "assistant":
            bot_response = result["messages"][-1]["content"]
        else:
            bot_response = "I'm ready. Let's begin."

        # --- SAVE STATE (Do NOT clear it yet, needed for Report) ---
        save_interview_state(call_id, result) 

        # Only append "Session Ended" for Vapi to hear, don't save to DB
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

# --- REPORT GENERATION ENDPOINT ---
@app.post("/generate-report")
async def manual_report_generation(request: Request):
    """Called when user clicks 'End Interview' on Frontend"""
    try:
        data = await request.json()
        call_id = data.get("call_id")
        
        state = get_interview_state(call_id)
        if not state:
            return JSONResponse({"status": "error", "message": "No session found"}, status_code=404)
            
        # If feedback already exists, return it
        if state.get("feedback"):
             return JSONResponse({"status": "success", "feedback": state["feedback"]})

        # Otherwise, force generation
        print(f"📝 Generating Report for {call_id}...")
        final_state = generate_feedback(state)
        save_interview_state(call_id, final_state)
        
        return JSONResponse({
            "status": "success", 
            "feedback": final_state["feedback"]
        })
    except Exception as e:
        print(f"Error generating report: {e}")
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)