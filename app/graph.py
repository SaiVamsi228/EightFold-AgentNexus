import json
import random
import re
from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.2)

# --- STATE DEFINITION ---
class InterviewState(TypedDict):
    messages: List[Dict[str, str]]
    role: str
    question_count: int
    persona_detected: str
    latest_evaluation: str
    is_finished: bool
    feedback: Dict[str, Any]
    used_questions: List[str]
    active_question: str
    retry_count: int
    current_topic_depth: int

# --- QUESTION BANK ---
# (Keep your existing QUESTIONS_DB here - I am omitting it to save space)
QUESTIONS_DB = {
  "Software Engineer": {
    "technical": ["Explain process vs thread?", "Design a URL shortener?", "TCP vs UDP?", "Explain Dependency Injection?", "Debug memory leak?", "RESTful API concept?", "SQL vs NoSQL?", "Garbage collection?", "ACID properties?", "Polymorphism?"],
    "behavioral": ["Critical production issue?", "Disagreed with manager?", "Prioritize deadlines?", "Mistake handled?", "Learn tech quickly?"]
  },
  "SDR": {
    "technical": ["Research prospect?", "Gatekeeper strategy?", "Handle objection 'Happy with vendor'?", "Key metrics?", "Qualifying leads?"],
    "behavioral": ["Repeated rejection?", "Difficult sale?", "Missed quota?", "Stay motivated?"]
  },
  "Retail Associate": {
    "technical": ["Return without receipt?", "Coworker stealing?", "Out of stock?", "Messy display?"],
    "behavioral": ["Angry customer?", "Above and beyond?", "Pressure during holidays?"]
  }
}

def load_questions(role: str):
    return QUESTIONS_DB.get(role, QUESTIONS_DB["Software Engineer"])

# --- NODE 1: ANALYZE (FIXED) ---
def analyze_input(state: InterviewState) -> InterviewState:
    # !!! CRITICAL FIX: If no messages exist, skip analysis !!!
    if not state["messages"]:
        return {**state, "persona_detected": "Normal", "latest_evaluation": "Good"}

    user_input = state["messages"][-1]["content"]
    last_q = next((m["content"] for m in reversed(state["messages"]) if m["role"] == "assistant"), "Start of Interview")

    prompt = f"""
    You are an Interview Analyst.
    Context (Last Question): "{last_q}"
    Candidate Response: "{user_input}"
    
    Analyze:
    1. PERSONA: Confused, Efficient, Chatty, Edge, Normal.
    2. EVALUATION: Vague, Good, Off-topic.
    
    Return JSON ONLY: {{ "persona": "...", "evaluation": "..." }}
    """

    try:
        resp = llm.invoke([HumanMessage(content=prompt)])
        content = resp.content.strip().replace("```json", "").replace("```", "")
        match = re.search(r'\{.*\}', content, re.DOTALL)
        data = json.loads(match.group(0)) if match else {"persona": "Normal", "evaluation": "Good"}
    except:
        data = {"persona": "Normal", "evaluation": "Good"}

    if state["question_count"] == 0:
        data["evaluation"] = "Good"

    return {
        **state,
        "persona_detected": data.get("persona", "Normal"),
        "latest_evaluation": data.get("evaluation", "Good")
    }

# --- NODE 2: ASK NEW QUESTION ---
def ask_new_question(state: InterviewState) -> InterviewState:
    questions = load_questions(state["role"])
    q_type = "behavioral" if state["question_count"] % 2 == 0 else "technical"
    
    # Avoid repeats
    used = state.get("used_questions", [])
    available = [q for q in questions[q_type] if q not in used]
    if not available: available = questions[q_type]
    
    base_question = random.choice(available)

    if state["question_count"] == 0:
        final_q = f"Great. Let's start the {state['role']} interview. {base_question}"
    elif state["persona_detected"] == "Efficient":
        final_q = base_question
    else:
        final_q = f"Got it. Next question: {base_question}"

    return {
        **state,
        "messages": state["messages"] + [{"role": "assistant", "content": final_q}],
        "question_count": state["question_count"] + 1,
        "used_questions": used + [base_question],
        "active_question": base_question
    }

# --- NODE 3: HANDLE SPECIAL ---
def handle_special(state: InterviewState) -> InterviewState:
    last_q = state.get("active_question", "the topic")
    user_input = state["messages"][-1]["content"]
    persona = state["persona_detected"]

    if persona == "Confused":
        task = f"User is confused by '{last_q}'. Simplify it and ask again."
    elif state["latest_evaluation"] == "Vague":
        task = f"User answer '{user_input}' was too vague. Ask for a specific example."
    else:
        task = f"User is off-topic. Politely bring them back to: '{last_q}'."

    reply = llm.invoke([SystemMessage(content="You are an Interviewer."), HumanMessage(content=task)]).content.strip()
    return {**state, "messages": state["messages"] + [{"role": "assistant", "content": reply}]}

# --- NODE 4: FEEDBACK ---
def generate_feedback(state: InterviewState) -> InterviewState:
    transcript = "\n".join([f"{m['role']}: {m['content']}" for m in state['messages'] if m['role'] != 'system'])
    
    prompt = f"""
    Act as a Hiring Manager for {state['role']}. Review:
    {transcript}
    
    Return VALID JSON ONLY:
    {{
        "score": "X/10",
        "confidence": "High/Medium/Low",
        "strengths": ["Point 1", "Point 2"],
        "improvements": ["Point 1", "Point 2"],
        "summary": "Executive summary.",
        "recommendation": "Hire/No Hire"
    }}
    """
    try:
        resp = llm.invoke([HumanMessage(content=prompt)]).content.strip()
        clean_json = resp.replace("```json", "").replace("```", "").strip()
        feedback_data = json.loads(clean_json)
    except:
        feedback_data = {"score": "N/A", "summary": "Error analyzing.", "recommendation": "N/A", "confidence": "Low", "strengths": [], "improvements": []}

    return {
        **state,
        "messages": state["messages"] + [{"role": "assistant", "content": "Interview complete. Feedback generated."}],
        "feedback": feedback_data,
        "is_finished": True
    }

# --- ROUTING ---
def decide_next(state: InterviewState):
    if state["question_count"] == 0: return "ask_new_question"
    if state["question_count"] >= 5: return "generate_feedback"
    
    if state["latest_evaluation"] in ["Vague", "Off-topic"] or state["persona_detected"] in ["Confused", "Chatty", "Edge"]:
        return "handle_special"
        
    return "ask_new_question"

# --- COMPILE ---
workflow = StateGraph(InterviewState)
workflow.add_node("analyze", analyze_input)
workflow.add_node("ask_new_question", ask_new_question)
workflow.add_node("handle_special", handle_special)
workflow.add_node("generate_feedback", generate_feedback)

workflow.set_entry_point("analyze")
workflow.add_conditional_edges("analyze", decide_next, {
    "ask_new_question": "ask_new_question", 
    "handle_special": "handle_special", 
    "generate_feedback": "generate_feedback"
})
workflow.add_edge("ask_new_question", END)
workflow.add_edge("handle_special", END)
workflow.add_edge("generate_feedback", END)

app_graph = workflow.compile()