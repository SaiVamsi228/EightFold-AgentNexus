from fastapi import FastAPI, Request
from app.graph import app_graph
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import uuid
import os

app = FastAPI()

@app.get("/")
def read_root():
    return {"status": "Interview Agent Active"}

# app/main.py  (only this function changes)

@app.post("/chat")
async def chat_endpoint(request: Request):
    data = await request.json()

    # Vapi always sends the full conversation history
    conversation = data.get("conversation", [])
    
    # Find the latest user message
    user_messages = [m for m in conversation if m["role"] == "user"]
    if not user_messages:
        user_input = "Start interview"
    else:
        user_input = user_messages[-1]["content"]

    # Extract role from the very first user message
    all_user_text = " ".join([m["content"] for m in user_messages]).lower()
    if any(x in all_user_text for x in ["sdr", "sales"]):
        role = "SDR"
    elif "retail" in all_user_text:
        role = "Retail Associate"
    else:
        role = "Software Engineer"

    # Reconstruct full message history in LangGraph format
    messages = []
    for msg in conversation:
        if msg["role"] == "user":
            messages.append({"role": "user", "content": msg["content"]})
        elif msg["role"] == "assistant" and msg["content"]:
            messages.append({"role": "assistant", "content": msg["content"]})

    # Count how many real questions the assistant has already asked
    assistant_questions = [m for m in messages if m["role"] == "assistant" and ("?" in m["content"] or m["content"].startswith(("Tell me", "Describe", "Explain", "Walk me")))]
    question_count = len(assistant_questions)

    # Build proper persistent state
    initial_state: InterviewState = {
        "messages": messages + [{"role": "user", "content": user_input}],  # add latest
        "role": role,
        "question_count": question_count,
        "current_question_type": "behavioral",  # will be overridden by logic inside nodes
        "persona_detected": "Normal",
        "latest_evaluation": "Good",
        "is_finished": False,
        "feedback": None
    }

    # Run the graph with full history
    result = app_graph.invoke(initial_state)

    # Extract the assistant's reply
    bot_reply = result["messages"][-1]["content"]

    # If interview finished → generate PDF (optional)
    if result.get("is_finished") and result.get("feedback"):
        try:
            create_pdf(result["feedback"])
            bot_reply += "\n\nI’ve also created a detailed PDF feedback report for you!"
        except Exception as e:
            print("PDF error:", e)

    # Vapi expects this exact format
    return {
        "results": [
            {
                "toolCallId": data.get("message", {}).get("toolCall", {}).get("id", "no-id"),
                "result": bot_reply
            }
        ]
    }

def create_pdf(feedback_data):
    filename = f"feedback_{uuid.uuid4().hex[:8]}.pdf"
    c = canvas.Canvas(filename, pagesize=letter)
    width, height = letter

    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, height - 50, "Interview Feedback Report")

    c.setFont("Helvetica", 12)
    y = height - 80

    c.drawString(50, y, f"Overall Score: {feedback_data.get('score', 'N/A')}/10")
    y -= 30

    c.drawString(50, y, "Strengths:")
    y -= 20
    for s in feedback_data.get("strengths", []):
        c.drawString(70, y, f"- {s}")
        y -= 15

    y -= 20
    c.drawString(50, y, "Areas for Improvement:")
    y -= 20
    for i in feedback_data.get("improvements", []):
        c.drawString(70, y, f"- {i}")
        y -= 15

    c.save()
    return filename