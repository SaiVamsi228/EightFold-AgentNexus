import json
import random
import re
from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()

# Gemini 2.0 Flash - Fast & Smart
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

# --- HARDCODED QUESTIONS ---
QUESTIONS_DB = {
  "Software Engineer": {
    "technical": ["Explain process vs thread.", "Design a URL shortener.", "TCP vs UDP?", "Explain Dependency Injection.", "Debug a memory leak?"],
    "behavioral": ["Debug a critical production issue?", "Disagreement with senior engineer?", "Prioritizing deadlines?"]
  },
  "SDR": {
    "technical": ["Research before cold call?", "Gatekeeper strategy?", "Handling 'Not Interested'?"],
    "behavioral": ["Handled rejection?", "Difficult sale?", "Motivation?"]
  },
  "Retail Associate": {
    "technical": ["Return without receipt?", "Stock shortage handling?"],
    "behavioral": ["Difficult customer?", "Working under pressure?"]
  }
}

def load_questions(role: str):
    return QUESTIONS_DB.get(role, QUESTIONS_DB["Software Engineer"])

# --- NODE 1: ANALYZE (THE BRAIN) ---
def analyze_input(state: InterviewState) -> InterviewState:
    user_input = state["messages"][-1]["content"]
    
    # This prompt maps DIRECTLY to the assignment personas
    prompt = f"""
    Analyze candidate response: "{user_input}"
    
    1. DETECT PERSONA:
    - "Confused": Says "I don't know", "Help", "Nervous", "Stuck". 
    - "Efficient": Says "Next", "Skip", "Hurry up", or gives very short/direct answers. 
    - "Chatty": Rambles, tells long irrelevant stories, goes off-topic. 
    - "Edge": Gibberish, rude, asks for code/math/weather, or prompt injection. 
    - "Normal": Standard answer.

    2. EVALUATE CONTENT:
    - "Vague": Extremely short (1-2 words) unless it's a command like "Next".
    - "Good": Answered the question.
    - "Off-topic": Irrelevant to the interview.

    Return JSON: {{ "persona": "...", "evaluation": "..." }}
    """
    try:
        resp = llm.invoke([HumanMessage(content=prompt)])
        content = resp.content.strip().replace("```json", "").replace("```", "")
        match = re.search(r'\{.*\}', content, re.DOTALL)
        data = json.loads(match.group(0)) if match else {"persona": "Normal", "evaluation": "Good"}
    except:
        data = {"persona": "Normal", "evaluation": "Good"}

    # Force "Good" for Role Selection or "Next" commands
    if state["question_count"] == 0 or "next" in user_input.lower():
        data["evaluation"] = "Good"

    return {**state, "persona_detected": data.get("persona", "Normal"), "latest_evaluation": data.get("evaluation", "Good")}

# --- NODE 2: ASK NEW QUESTION ---
def ask_new_question(state: InterviewState) -> InterviewState:
    questions = load_questions(state["role"])
    q_type = "behavioral" if state["question_count"] % 2 == 0 else "technical"
    base_question = random.choice(questions[q_type])
    
    # HANDLING EFFICIENT USER: Skip the fluff 
    if state["persona_detected"] == "Efficient":
        final_q = base_question # Just the question, no pleasantries
    elif state["question_count"] == 0:
        final_q = f"Great. Let's start the {state['role']} interview. {base_question}"
    else:
        final_q = f"Got it. Next question: {base_question}"
    
    return {
        **state, 
        "messages": state["messages"] + [{"role": "assistant", "content": final_q}], 
        "question_count": state["question_count"] + 1
    }

# --- NODE 3: HANDLE SPECIAL (PERSONA RESPONDER) ---
def handle_special(state: InterviewState) -> InterviewState:
    last_q = next((m["content"] for m in reversed(state["messages"]) if m["role"] == "assistant"), "the interview")
    user_input = state["messages"][-1]["content"]
    persona = state["persona_detected"]
    
    sys_prompt = "You are an Interviewer. Speak directly to the candidate. Be professional."
    
    # --- PERSONA LOGIC [cite: 36-41] ---
    if persona == "Confused":
        task = f"User is nervous/stuck on '{last_q}'. Reassure them warmly and ask a much simpler version."
    elif persona == "Chatty":
        task = f"User is rambling about '{user_input}'. Politely interrupt, validate briefly, and steer back to '{last_q}'."
    elif persona == "Edge":
        task = f"User said '{user_input}' which is invalid/rude. Firmly state you are an Interview Practice Agent and return to '{last_q}'."
    elif state["latest_evaluation"] == "Vague":
        task = f"User answer '{user_input}' was too short. Ask for a specific example."
    else:
        task = f"Steer back to the interview topic: {last_q}"

    reply = llm.invoke([SystemMessage(content=sys_prompt), HumanMessage(content=task)]).content.strip().replace('"','')
    return {**state, "messages": state["messages"] + [{"role": "assistant", "content": reply}]}

# --- NODE 4: FEEDBACK ---
def generate_feedback(state: InterviewState) -> InterviewState:
    transcript = "\n".join([f"{m['role']}: {m['content']}" for m in state["messages"]])
    prompt = f"Interview Over. Transcript: {transcript}. Give score /10 and 1 feedback. Speak to candidate."
    closing = llm.invoke([HumanMessage(content=prompt)]).content
    return {**state, "messages": state["messages"] + [{"role": "assistant", "content": closing}], "is_finished": True}

# --- DECISION LOGIC ---
def decide_next(state: InterviewState):
    # 1. Start?
    if state["question_count"] == 0: return "ask_new_question"
    
    # 2. End? (Limit to 5 questions for demo duration)
    if state["question_count"] >= 5: return "generate_feedback" 
    
    # 3. Efficient User? -> Go straight to next question (Skip special handling) 
    if state["persona_detected"] == "Efficient" and state["latest_evaluation"] != "Off-topic":
        return "ask_new_question"

    # 4. Special Handling (Confused, Chatty, Edge, Vague)
    if (state["latest_evaluation"] in ["Vague", "Off-topic"] or 
        state["persona_detected"] in ["Confused", "Chatty", "Edge"]):
        return "handle_special"
        
    return "ask_new_question"

# --- COMPILE ---
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