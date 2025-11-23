import json
import random
import re
from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()

# Gemini 2.0 Flash
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.2)

# --- STATE DEFINITION ---
class InterviewState(TypedDict):
    messages: List[Dict[str, str]]
    role: str
    question_count: int
    persona_detected: str
    latest_evaluation: str
    is_finished: bool
    feedback: Dict[str, Any]  # UPDATED: now storing JSON dict
    used_questions: List[str]  # NEW FIELD
    active_question: str       # NEW FIELD
    retry_count: int           # NEW FIELD
    current_topic_depth: int   # NEW FIELD

# --- QUESTION BANK ---
QUESTIONS_DB = {
    "Software Engineer": {
        "technical": [
            "Can you explain the difference between a process and a thread?",
            "How would you design a scalable URL shortener like Bitly?",
            "What is the difference between TCP and UDP protocols?",
            "In your own words, what is Dependency Injection and why is it useful?",
            "How do you typically approach debugging a memory leak in production?",
            "Explain the concept of a RESTful API.",
            "What is the difference between SQL and NoSQL databases?",
            "How does garbage collection work in your preferred programming language?",
            "What are the ACID properties in a database?",
            "Explain the concept of polymorphism in Object-Oriented Programming."
        ],
        "behavioral": [
            "Tell me about a time you had to debug a critical production issue.",
            "Describe a situation where you disagreed with a senior engineer or manager.",
            "How do you prioritize your tasks when you have multiple tight deadlines?",
            "Tell me about a mistake you made in a project and how you handled it.",
            "Describe a time you had to learn a new technology very quickly."
        ]
    }
}

def load_questions(role: str):
    return QUESTIONS_DB.get(role, QUESTIONS_DB["Software Engineer"])

# --- NODE 1: ANALYZE ---
def analyze_input(state: InterviewState) -> InterviewState:
    user_input = state["messages"][-1]["content"]

    last_q = next((m["content"] for m in reversed(state["messages"]) if m["role"] == "assistant"), "Start of Interview")

    prompt = f"""
    You are an Interview Analyst.

    Context (Last Question Asked): "{last_q}"
    Candidate Response: "{user_input}"

    Analyze the response using strict rules:

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
    base_question = random.choice(questions[q_type])

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
        "active_question": final_q
    }

# --- NODE 3: HANDLE SPECIAL ---
def handle_special(state: InterviewState) -> InterviewState:
    last_q = next((m["content"] for m in reversed(state["messages"]) if m["role"] == "assistant"), "the topic")

    if len(last_q) > 200:
        last_q = last_q[:200] + "..."

    user_input = state["messages"][-1]["content"]
    persona = state["persona_detected"]

    sys_prompt = "You are an Interviewer. Speak directly to the candidate."

    if persona == "Confused":
        task = f"""
        The candidate is stuck on "{last_q}".
        Rephrase simply and ask again.
        """
    elif persona == "Chatty":
        task = f"User is rambling. Redirect politely back to: {last_q}"
    elif persona == "Edge":
        task = f"User message inappropriate. Warn once and restate: {last_q}"
    elif state["latest_evaluation"] == "Vague":
        task = f"User answer too short. Ask for details about: {last_q}"
    else:
        task = f"Politely redirect back to: {last_q}"

    reply = llm.invoke([SystemMessage(content=sys_prompt), HumanMessage(content=task)]).content.strip()

    return {**state, "messages": state["messages"] + [{"role": "assistant", "content": reply}]}


# --- UPDATED NODE 4: STRICT JSON FEEDBACK ---
def generate_feedback(state: InterviewState) -> InterviewState:
    transcript = "\n".join([
        f"{m['role']}: {m['content']}"
        for m in state['messages']
        if m['role'] != 'system'
    ])

    prompt = f"""
    Act as a Hiring Manager for a {state['role']} role.
    Review the transcript:

    {transcript}

    Generate VALID JSON (no markdown, no backticks).

    REQUIRED STRUCTURE:
    {{
        "score": "X/10",
        "confidence": "High/Medium/Low",
        "strengths": ["...", "...", "..."],
        "improvements": ["...", "...", "..."],
        "summary": "2-3 sentence summary.",
        "recommendation": "Strong Hire / Hire / No Hire"
    }}
    """

    try:
        response = llm.invoke([HumanMessage(content=prompt)]).content.strip()
        clean_json = response.replace("```json", "").replace("```", "").strip()
        feedback_data = json.loads(clean_json)
    except Exception as e:
        print("JSON ERROR:", e)
        feedback_data = {
            "score": "N/A",
            "confidence": "Low",
            "strengths": ["Unable to analyze"],
            "improvements": ["Try again"],
            "summary": "Feedback generation failed.",
            "recommendation": "N/A"
        }

    return {
        **state,
        "messages": state["messages"] + [{"role": "assistant", "content": "Interview complete."}],
        "feedback": feedback_data,
        "is_finished": True
    }


# --- DECISION ---
def decide_next(state: InterviewState):
    if state["question_count"] == 0:
        return "ask_new_question"

    if state["question_count"] >= 5:
        return "generate_feedback"

    if state["persona_detected"] == "Efficient" and state["latest_evaluation"] != "Off-topic":
        return "ask_new_question"

    if state["latest_evaluation"] in ["Vague", "Off-topic"] or state["persona_detected"] in ["Confused", "Chatty", "Edge"]:
        return "handle_special"

    return "ask_new_question"


# --- WORKFLOW COMPILE ---
workflow = StateGraph(InterviewState)
workflow.add_node("analyze", analyze_input)
workflow.add_node("ask_new_question", ask_new_question)
workflow.add_node("handle_special", handle_special)
workflow.add_node("generate_feedback", generate_feedback)

workflow.set_entry_point("analyze")

workflow.add_conditional_edges(
    "analyze", decide_next,
    {
        "ask_new_question": "ask_new_question",
        "handle_special": "handle_special",
        "generate_feedback": "generate_feedback"
    }
)

workflow.add_edge("ask_new_question", END)
workflow.add_edge("handle_special", END)
workflow.add_edge("generate_feedback", END)

app_graph = workflow.compile()
