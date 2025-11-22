# app/main.py
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from app.graph import app_graph, InterviewState
import json
import traceback

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Eightfold Interview Coach - Ready"}

@app.post("/webhook")
async def webhook(request: Request):
    try:
        payload = await request.json()
        
        # Vapi always sends this structure like this
        message = payload.get("message", {})
        conversation = payload.get("conversation", [])
        
        # Safety: if no conversation, start fresh
        if not conversation:
            return JSONResponse({
                "results": [{"toolCallId": message.get("toolCall", {}).get("id", "1"), "result": "Hi! What role are you practicing for today?"}]
            })

        # Build proper message list
        messages = []
        for m in conversation:
            if m.get("role") in ["user", "assistant"] and m.get("content"):
                messages.append({"role": m["role"], "content": m["content"]})

        # Detect role
        user_text = " ".join([m["content"].lower() for m in messages if m["role"] == "user"])
        if any(word in user_text for word in ["sdr", "sales"]):
            role = "SDR"
        elif "retail" in user_text:
            role = "Retail Associate"
        else:
            role = "Software Engineer"

        # Count real questions asked
        questions_asked = [
            m for m in messages 
            if m["role"] == "assistant" and ("?" in m["content"] or m["content"].startswith(("Tell me", "Describe", "Explain", "How", "What")))
        ]
        question_count = len(questions_asked)

        # Build state
        state: InterviewState = {
            "messages": messages,
            "role": role,
            "question_count": question_count,
            "persona_detected": "Normal",
            "latest_evaluation": "Good",
            "is_finished": False,
            "feedback": None
        }

        # Run the brain
        result = app_graph.invoke(state)
        bot_reply = result["messages"][-1]["content"]

        # If finished, add PDF note
        if result.get("is_finished"):
            bot_reply += "\n\nThanks for practicing! I've generated a detailed PDF feedback report for you."

        # Return correct Vapi format
        tool_call_id = message.get("toolCall", {}).get("id", "fallback")
        return JSONResponse({
            "results": [{
                "toolCallId": tool_call_id,
                "result": bot_reply
            }]
        })

    except Exception as e:
        print("ERROR:", str(e))
        traceback.print_exc()
        # This prevents the "I encountered an issue" message
        return JSONResponse({
            "results": [{
                "toolCallId": message.get("toolCall", {}).get("id", "error"),
                "result": "Sorry, I'm having a technical glitch. Let's start over — what role are you practicing for?"
            }]
        })