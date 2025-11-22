# app/main.py
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from app.graph import app_graph, InterviewState
import json
import traceback

app = FastAPI()

@app.get("/")
def home():
    return {"status": "ready"}

@app.post("/webhook")
async def webhook(request: Request):
    try:
        payload = await request.json()

        # EXTRACT USER MESSAGE (handles both tool params and conversation)
        user_message = ""
        conversation = payload.get("conversation", [])

        # From tool parameters (fixed to check if 'message' is dict)
        message = payload.get("message")
        tool_params = {}
        if isinstance(message, dict):
            tool_params = message.get("toolCall", {}).get("parameters", {})
        if tool_params:
            user_message = tool_params.get("message", "")

        # From conversation if not
        if not user_message and conversation:
            last_msg = conversation[-1]
            if last_msg.get("role") == "user":
                user_message = last_msg.get("content", "")

        if not user_message:
            user_message = "software engineer"  # Fallback

        # Build full messages list
        messages = []
        for msg in conversation:
            if msg.get("role") in ["user", "assistant"] and msg.get("content"):
                messages.append({"role": msg["role"], "content": msg["content"]})

        # Add the latest user message
        messages.append({"role": "user", "content": user_message})

        # Detect role
        all_user_text = " ".join([m["content"].lower() for m in messages if m["role"] == "user"])
        role = "SDR" if any(x in all_user_text for x in ["sdr", "sales"]) else "Retail Associate" if "retail" in all_user_text else "Software Engineer"

        # Count questions
        question_count = sum(1 for m in messages if m["role"] == "assistant" and ("?" in m["content"] or m["content"].startswith(("Tell", "Describe", "Explain", "How", "What"))))

        # State
        state: InterviewState = {
            "messages": messages,
            "role": role,
            "question_count": question_count,
            "persona_detected": "Normal",
            "latest_evaluation": "Good",
            "is_finished": False,
            "feedback": {}
        }

        # Run
        result = app_graph.invoke(state)
        reply = result["messages"][-1]["content"]

        # Vapi format
        tool_id = payload.get("message", {}).get("toolCall", {}).get("id", "1") if isinstance(payload.get("message"), dict) else "1"
        return JSONResponse({
            "results": [{"toolCallId": tool_id, "result": reply}]
        })

    except Exception as e:
        print("ERROR:", traceback.format_exc())
        # Better fallback - starts the interview
        return JSONResponse({
            "results": [{"toolCallId": "error", "result": "Got it, Software Engineer role. Tell me about a time you debugged a critical production issue."}]
        })