from fastapi import FastAPI, Request
from app.graph import app_graph
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import uuid

app = FastAPI()


@app.get("/")
def read_root():
    return {"status": "Interview Agent Active"}


# ============================================================
# /chat ENDPOINT  (FINAL VERSION WITH YOUR FIX + PDF LOGIC)
# ============================================================
@app.post("/chat")
async def chat_endpoint(request: Request):
    data = await request.json()

    # ------------------------------
    # VAPI MESSAGE EXTRACTION FIX
    # ------------------------------
    user_message = ""

    # Case 1 → Vapi Tool Call (your current mode)
    # Payload: {"message": "User text"}
    if isinstance(data.get("message"), str):
        user_message = data["message"]

    # Case 2 → Vapi server event object
    # Payload: {"message": {"role": "user", "content": "..."}}
    elif isinstance(data.get("message"), dict):
        user_message = data["message"].get("content", "")

    # Case 3 → Fallback history
    elif "messages" in data:
        user_message = data["messages"][-1]["content"]

    # Final fallback
    if not user_message:
        user_message = "Start Interview"

    # -----------------------------------
    # ROLE DETECTION
    # -----------------------------------
    role = "Software Engineer"
    if "sales" in user_message.lower() or "sdr" in user_message.lower():
        role = "SDR"
    elif "retail" in user_message.lower():
        role = "Retail Associate"

    # -----------------------------------
    # GRAPH STATE
    # -----------------------------------
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

    # -----------------------------------
    # RUN GRAPH
    # -----------------------------------
    result = app_graph.invoke(initial_state)

    bot_response = result["messages"][-1].content

    # -----------------------------------
    # PDF GENERATION
    # -----------------------------------
    if result.get("is_finished") and result.get("feedback"):
        feedback_data = result["feedback"]
        create_pdf(feedback_data)
        bot_response += (
            " Your interview is complete. I have also generated a PDF report for you."
        )

    # -----------------------------------
    # RETURN TO VAPI TOOL
    # -----------------------------------
    return {
        "results": [
            {
                "toolCallId": data.get("toolCallId", "unknown"),
                "result": bot_response
            }
        ],
        "text": bot_response
    }


# ============================================================
# PDF CREATION FUNCTION
# ============================================================
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
