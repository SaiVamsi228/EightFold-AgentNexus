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

# --- STRICT STATE DEFINITION ---
class InterviewState(TypedDict):
    messages: List[Dict[str, str]]
    role: str
    question_count: int
    persona_detected: str
    latest_evaluation: str
    is_finished: bool
    feedback: Dict[str, Any]
    used_questions: List[str]
    active_question: str  # <--- NEW: Anchor for the current topic

# --- ROBUST QUESTION BANK (Same as before) ---
QUESTIONS_DB = {
  "Software Engineer": {
    "technical": [ "Explain the difference between a process and a thread.", "How would you design a scalable URL shortener?", "What is the difference between TCP and UDP?", "Explain Dependency Injection.", "How do you debug a memory leak?", "Explain RESTful APIs.", "SQL vs NoSQL?" ],
    "behavioral": [ "Tell me about a mistake you made and how you handled it.", "Describe a disagreement with a manager.", "How do you prioritize tasks?", "Tell me about learning a new tech quickly." ]
  },
  # ... (Include other roles here)
}

def load_questions(role: str):
    return QUESTIONS_DB.get(role, QUESTIONS_DB["Software Engineer"])

# --- NODE 1: ANALYZE (ANCHORED) ---
def analyze_input(state: InterviewState) -> InterviewState:
    user_input = state["messages"][-1]["content"]
    
    # ROBUSTNESS FIX: Compare against the Active Question, not just the last message
    # This prevents the "Dogs" or "Nervous" replies from overwriting the context.
    current_context = state.get("active_question")
    if not current_context:
        current_context = "Introduction and Role Selection"

    prompt = f"""
    You are an Interview Analyst.
    
    Primary Interview Question: "{current_context}"
    Candidate Response: "{user_input}"
    
    Analyze if the candidate is answering the PRIMARY Question, or drifting.
    
    1. DETECT PERSONA:
    - "Confused": Asking for help, "I don't understand".
    - "Chatty": Talking about dogs, weather, irrelevant life stories.
    - "Efficient": Short, direct.
    - "Normal": Answering the question.

    2. EVALUATE CONTENT:
    - "Vague": Needs more detail (e.g., "I fixed it" without explaining how).
    - "Good": Substantive answer to the PRIMARY Question.
    - "Off-topic": Completely unrelated.

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

# --- NODE 2: ASK NEW QUESTION (ROBUST) ---
def ask_new_question(state: InterviewState) -> InterviewState:
    questions = load_questions(state["role"])
    q_type = "behavioral" if state["question_count"] % 2 == 0 else "technical"
    
    used = state.get("used_questions", [])
    available = [q for q in questions[q_type] if q not in used]
    if not available: available = questions[q_type] 

    base_question = random.choice(available)
    
    # LOGIC FIX: Only show Intro if it's truly the start AND we haven't asked anything yet
    if state["question_count"] == 0 and not state.get("used_questions"):
        final_q = f"Great. Let's start the {state['role']} interview. {base_question}"
    elif state["latest_evaluation"] == "Good":
        transition_prompt = f"User gave a good answer to '{state.get('active_question')}'. Generate a 3-word positive acknowledgement."
        transition = llm.invoke([HumanMessage(content=transition_prompt)]).content.strip().replace('"','')
        final_q = f"{transition}. {base_question}"
    else:
        final_q = f"Okay. Moving on: {base_question}"
    
    return {
        **state, 
        "messages": state["messages"] + [{"role": "assistant", "content": final_q}], 
        "question_count": state["question_count"] + 1,
        "used_questions": used + [base_question],
        "active_question": base_question # <--- UPDATE ANCHOR
    }

# --- NODE 3: HANDLE SPECIAL (ANCHORED) ---
def handle_special(state: InterviewState) -> InterviewState:
    # ROBUSTNESS FIX: Always refer back to the Active Question
    target_topic = state.get("active_question", "the interview")
    user_input = state["messages"][-1]["content"]
    persona = state["persona_detected"]
    
    sys_prompt = "You are an Interviewer. Be professional but human."
    
    if persona == "Confused":
        task = f"Candidate is stuck on: '{target_topic}'. Simplify this question significantly and ask again."
    elif persona == "Chatty":
        task = f"Candidate is distracted by: '{user_input}'. Briefly acknowledge it (be empathetic), then FIRMLY steer back to: '{target_topic}'."
    elif state["latest_evaluation"] == "Vague":
        task = f"Candidate answer to '{target_topic}' was: '{user_input}'. This is too vague. Ask for a specific STAR method example."
    else:
        task = f"Steer candidate back to: '{target_topic}'."

    reply = llm.invoke([SystemMessage(content=sys_prompt), HumanMessage(content=task)]).content.strip()
    return {**state, "messages": state["messages"] + [{"role": "assistant", "content": reply}]}

# --- NODE 4: FEEDBACK ---
def generate_feedback(state: InterviewState) -> InterviewState:
    transcript = "\n".join([f"{m['role']}: {m['content']}" for m in state["messages"]])
    prompt = f"Interview Over. Transcript: {transcript}. Generate a markdown score report (Comm/Tech/Feedback)."
    closing = llm.invoke([HumanMessage(content=prompt)]).content
    return {**state, "messages": state["messages"] + [{"role": "assistant", "content": closing}], "is_finished": True}

# --- LOGIC ---
def decide_next(state: InterviewState):
    if state["question_count"] == 0: return "ask_new_question"
    if state["question_count"] >= 5: return "generate_feedback"
    
    # If user answered efficiently or normally, move on
    if state["latest_evaluation"] == "Good" or (state["persona_detected"] == "Efficient" and state["latest_evaluation"] != "Off-topic"):
        return "ask_new_question"

    return "handle_special"

workflow = StateGraph(InterviewState)
workflow.add_node("analyze", analyze_input)
workflow.add_node("ask_new_question", ask_new_question)
workflow.add_node("handle_special", handle_special)
workflow.add_node("generate_feedback", generate_feedback)
workflow.set_entry_point("analyze")
workflow.add_conditional_edges("analyze", decide_next, {"ask_new_question": "ask_new_question", "handle_special": "handle_special", "generate_feedback": "generate_feedback"})
workflow.add_edge("ask_new_question", END)
workflow.add_edge("handle_special", END)
workflow.add_edge("generate_feedback", END)
app_graph = workflow.compile()