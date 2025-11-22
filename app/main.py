from fastapi import FastAPI, Request
from app.graph import app_graph
import os

app = FastAPI()

@app.get("/")
def read_root():
    return {"status": "Interview Agent Active"}

@app.post("/chat")
async def chat_endpoint(request: Request):
    data = await request.json()
    
    # Vapi sends the message history in 'message' or 'messages' depending on config
    # We look for the user's latest input
    try:
        # This handles Vapi's standard "assistant request" payload
        user_message = data.get("message", {}).get("content", "")
        if not user_message and "messages" in data:
             # Fallback if Vapi sends full history list
            user_message = data["messages"][-1]["content"]
    except:
        user_message = "Start Interview"

    # Default Role detection (Simple keyword match for MVP)
    # In a real app, you might ask the user to select this first
    role = "Software Engineer"
    if "sales" in user_message.lower() or "sdr" in user_message.lower():
        role = "SDR"
    elif "retail" in user_message.lower():
        role = "Retail Associate"

    # Initialize State
    # Note: For a stateless HTTP request, we rebuild state roughly from the transcript
    # In a full production app, you'd use a database (Redis/Postgres) to persist state ID.
    # Here we simplify for the hackathon: We assume the client sends history or we treat it turn-by-turn.
    
    initial_state = {
        "messages": [{"role": "user", "content": user_message}],
        "role": role,
        "question_count": 0, # In MVP, this resets per turn if no DB. 
                             # *FIX*: For Vapi, we usually rely on Vapi keeping context. 
                             # For this demo, we will fake the count logic or rely on Vapi's transcript.
        "current_question_type": "behavioral",
        "persona_detected": "Normal",
        "latest_evaluation": "Good",
        "is_finished": False,
        "feedback": None
    }

    # Run the Graph
    result = app_graph.invoke(initial_state)
    
    bot_response = result["messages"][-1].content
    
    # Return format Vapi expects
    return {
        "results": [
            {
                "toolCallId": data.get("toolCallId", "unknown"),
                "result": bot_response
            }
        ],
        # If simple text response is needed:
        "text": bot_response 
    }