from fastapi import FastAPI, Request
from app.graph import app_graph
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import uuid

app = FastAPI()

@app.get("/")
def read_root():
    return {"status": "Interview Agent Active"}


# ---------------------------
# /chat ENDPOINT
# ---------------------------
@app.post("/chat")
async def chat_endpoint(request: Request):
    data = await request.json()

    # Extract user message from Vapi request format
    try:
        user_message = data.get("message", {}).get("content", "")
        if not user_message and "messages" in data:
            user_message = data["messages"][-1]["content"]
    except:
        user_message = "Start Interview"

    # Simple Role Detection (MVP)
    role = "Software Engineer"
    if "sales" in user_message.lower() or "sdr" in user_message.lower():
        role = "SDR"
    elif "retail" in user_message.lower():
        role = "Retail Associate"

    # Initialize Graph State (stateless MVP)
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

    # Run LangGraph
    result = app_graph.invoke(initial_state)
    bot_response = result["messages"][-1].content

    # -------- PDF Logic --------
    if result.get("is_finished") and result.get("feedback"):
        feedback_data = result["feedback"]
        pdf_file = create_pdf(feedback_data)
        bot_response += (
            " Your interview is complete. I have also generated a detailed PDF report for you."
        )

    # Vapi tool response format
    return {
        "results": [
            {
                "toolCallId": data.get("toolCallId", "unknown"),
                "result": bot_response
            }
        ],
        "text": bot_response
    }


# ---------------------------
# PDF GENERATION FUNCTION
# ---------------------------
def create_pdf(feedback_data):
    filename = f"feedback_{uuid.uuid4().hex[:8]}.pdf"
    c = canvas.Canvas(filename, pagesize=letter)
    width, height = letter

    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, height - 50, "Interview Feedback Report")

    c.setFont("Helvetica", 12)
    y = height - 80

    # Score
    c.drawString(50, y, f"Overall Score: {feedback_data.get('score', 'N/A')}/10")
    y -= 30

    # Strengths
    c.drawString(50, y, "Strengths:")
    y -= 20
    for s in feedback_data.get("strengths", []):
        c.drawString(70, y, f"- {s}")
        y -= 15

    # Improvements
    y -= 20
    c.drawString(50, y, "Areas for Improvement:")
    y -= 20
    for i in feedback_data.get("improvements", []):
        c.drawString(70, y, f"- {i}")
        y -= 15

    c.save()
    return filename
