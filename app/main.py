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

@app.post("/chat")
async def chat_endpoint(request: Request):
    data = await request.json()

    # 1. EXTRACT USER MESSAGE (Robust Fix)
    user_message = ""
    if isinstance(data.get("message"), str):
        user_message = data["message"]
    elif isinstance(data.get("message"), dict):
        user_message = data["message"].get("content", "")
    elif "messages" in data:
        user_message = data["messages"][-1]["content"]

    if not user_message:
        user_message = "Start Interview"

    # 2. DETECT ROLE
    role = "Software Engineer"
    if "sales" in user_message.lower() or "sdr" in user_message.lower():
        role = "SDR"
    elif "retail" in user_message.lower():
        role = "Retail Associate"

    # 3. SETUP STATE
    # Note: question_count resets to 0 here because we are stateless.
    # To fix "Confused" detection, we removed the length check in graph.py
    initial_state = {
        "messages": [{"role": "user", "content": user_message}],
        "role": role,
        "question_count": 0,
        "current_question_type": "behavioral",
        "persona_detected": "Normal",
        "latest_evaluation": "Good",
        "is_finished": False,
        "feedback": None
    }

    # 4. RUN INTELLIGENCE
    result = app_graph.invoke(initial_state)
    bot_response = result["messages"][-1].content

    # 5. GENERATE PDF IF DONE
    if result.get("is_finished") and result.get("feedback"):
        feedback_data = result["feedback"]
        # In production, upload this to S3. Here we just create it locally.
        try:
            create_pdf(feedback_data)
            bot_response += " I have also generated a PDF feedback report for you."
        except Exception as e:
            print(f"PDF Error: {e}")

    return {
        "results": [
            {
                "toolCallId": data.get("toolCallId", "unknown"),
                "result": bot_response
            }
        ],
        "text": bot_response
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