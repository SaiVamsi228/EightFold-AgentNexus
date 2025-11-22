from fastapi import FastAPI, Request
from app.graph import app_graph, InterviewState
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import uuid
import json

app = FastAPI()

@app.get("/")
def home():
    return {"status": "Eightfold Interview Coach Ready"}

@app.post("/webhook")
async def webhook(request: Request):
    data = await request.json()
    conversation = data.get("conversation", [])

    # Reconstruct full history
    messages = []
    for msg in conversation:
        if msg["role"] in ["user", "assistant"]:
            messages.append({"role": msg["role"], "content": msg.get("content", "")})

    # Detect role from entire history
    user_text = " ".join([m["content"] for m in messages if m["role"] == "user"]).lower()
    if any(x in user_text for x in ["sdr", "sales"]):
        role = "SDR"
    elif "retail" in user_text:
        role = "Retail Associate"
    else:
        role = "Software Engineer"

    # Count real questions asked
    questions_asked = [m for m in messages if m["role"] == "assistant" and ("?" in m["content"] or m["content"].startswith(("Tell me", "Describe", "Explain", "Walk me")))]
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

    # Run agent
    result = app_graph.invoke(state)
    assistant_reply = result["messages"][-1]["content"]

    # Generate PDF on finish
    if result.get("is_finished"):
        try:
            create_pdf(result["feedback"])
            assistant_reply += "\n\nI've also generated a detailed PDF feedback report for you!"
        except Exception as e:
            print("PDF failed:", e)

    # Vapi tool response format
    tool_call_id = data.get("message", {}).get("toolCall", {}).get("id", "no-id")
    return {
        "results": [{
            "toolCallId": tool_call_id,
            "result": assistant_reply
        }]
    }

def create_pdf(feedback: dict):
    filename = f"feedback_{uuid.uuid4().hex[:8]}.pdf"
    c = canvas.Canvas(filename, pagesize=letter)
    width, height = letter
    c.setFont("Helvetica-Bold", 20)
    c.drawString(50, height - 100, "Eightfold Interview Feedback")
    c.setFont("Helvetica", 14)
    y = height - 150
    c.drawString(50, y, f"Score: {feedback.get('score', 8)}/10")
    y -= 40
    c.drawString(50, y, "Strengths:")
    y -= 25
    for s in feedback.get("strengths", []):
        c.drawString(70, y, f"• {s}")
        y -= 20
    y -= 20
    c.drawString(50, y, "Improvements:")
    y -= 25
    for i in feedback.get("improvements", []):
        c.drawString(70, y, f"• {i}")
        y -= 20
    c.save()
    print(f"PDF saved: {filename}")