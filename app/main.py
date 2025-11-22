# app/main.py
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from app.graph import app_graph, InterviewState
import json
import traceback

app = FastAPI()

@app.post("/webhook")
async def webhook(request: Request):
    try:
        payload = await request.json()
        conversation = payload.get("conversation", [])

        if not conversation:
            return JSONResponse({"results": [{"toolCallId": "1", "result": "Hi! What role are you practicing for?"}]})

        # Build messages
        messages = [{"role": m["role"], "content": m.get("content","")} 
                   for m in conversation if m.get("role") in ["user","assistant"] and m.get("content")]

        # Role detection
        user_text = " ".join([m["content"].lower() for m in messages if m["role"]=="user"])
        role = "SDR" if any(w in user_text for w in ["sdr","sales"]) else "Retail Associate" if "retail" in user_text else "Software Engineer"

        # Question count
        question_count = sum(1 for m in messages if m["role"]=="assistant" and ("?" in m["content"] or m["content"].startswith(("Tell","Describe","Explain","How","What"))))

        state: InterviewState = {
            "messages": messages,
            "role": role,
            "question_count": question_count,
            "persona_detected": "Normal",
            "latest_evaluation": "Good",
            "is_finished": False,
            "feedback": {}
        }

        result = app_graph.invoke(state)
        reply = result["messages"][-1]["content"]

        tool_id = payload.get("message", {}).get("toolCall", {}).get("id", "1")

        return JSONResponse({
            "results": [{"toolCallId": tool_id, "result": reply}]
        })

    except Exception as e:
        traceback.print_exc()
        return JSONResponse({
            "results": [{"toolCallId": "error", "result": "Sorry, technical glitch. Let's continue — what role are you practicing for?"}]
        })