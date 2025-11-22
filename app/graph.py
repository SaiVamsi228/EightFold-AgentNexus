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
    retry_count: int  # NEW: Track how many times we've tried the SAME question

# --- QUESTION BANK ---
QUESTIONS_DB = {
  "Software Engineer": {
    "technical": [
      "Can you explain the difference between a process and a thread?",
      "How would you design a scalable URL shortener?",
      "What is the difference between TCP and UDP?",
      "In your own words, what is Dependency Injection?",
      "How do you typically approach debugging a memory leak?",
      "Explain the concept of a RESTful API.",
      "SQL vs NoSQL databases?"
    ],
    "behavioral": [
      "Tell me about a mistake you made in a project and how you handled it.",
      "Describe a situation where you disagreed with a senior engineer.",
      "How do you prioritize your tasks when you have multiple deadlines?",
      "Tell me about a time you had to learn a new technology very quickly."
    ]
  },
  "SDR": {
    "technical": [
      "How do you research a prospect before making a cold call?",
      "What is your strategy for getting past a gatekeeper?",
      "How do you handle the objection: 'We are happy with our current vendor'?",
      "Walk me through your process for qualifying a lead."
    ],
    "behavioral": [
      "Tell me about a time you faced repeated rejection.",
      "Describe your most difficult sale and how you eventually closed it."
    ]
  },
  "Retail Associate": {
    "technical": [
      "How do you handle a customer returning an item without a receipt?",
      "What would you do if you saw a coworker stealing?",
      "Explain how you organize a messy display during a rush."
    ],
    "behavioral": [
      "Tell me about a time you dealt with a very difficult customer.",
      "Describe a time you went above and beyond for a customer."
    ]
  }
}

def load_questions(role: str):
    # Fallback to Software Engineer if role is invalid
    return QUESTIONS_DB.get(role, QUESTIONS_DB["Software Engineer"])

# --- NODE 1: ANALYZE (IMPROVED CLASSIFICATION) ---
def analyze_input(state: InterviewState) -> InterviewState:
    user_input = state["messages"][-1]["content"]
    
    # 1. CHECK FOR ROLE ATTEMPT (If user tries to switch, we must catch it)
    # Note: We won't switch, but we need to know they tried so we can block it.
    
    current_context = state.get("active_question")
    if not current_context: current_context = "Introduction"

    prompt = f"""
    You are an Interview Analyst.
    
    Current Role: {state['role']}
    Current Question: "{current_context}"
    User Input: "{user_input}"
    
    Analyze the input.
    
    1. PERSONA DETECTION:
    - "Distracted": User mentions external noise, dogs, internet lag, "can't hear", "wait a sec".
    - "Confused": User says "I don't know", "Explain please", "What does that mean?".
    - "Resisting": User tries to change the role (e.g., "I want retail instead"), or refuses to answer.
    - "Normal": Attempts to answer.

    2. EVALUATION:
    - "Vague": Too short/generic.
    - "Good": Relevant answer.
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

# --- NODE 2: ASK NEW QUESTION ---
def ask_new_question(state: InterviewState) -> InterviewState:
    questions = load_questions(state["role"])
    q_type = "behavioral" if state["question_count"] % 2 == 0 else "technical"
    
    used = state.get("used_questions", [])
    available = [q for q in questions[q_type] if q not in used]
    
    if not available: available = questions[q_type]

    base_question = random.choice(available)
    
    # Transition Logic
    if state["question_count"] == 0:
        final_q = f"Great. Let's start the {state['role']} interview. {base_question}"
    elif state["latest_evaluation"] == "Good":
        transition_prompt = f"User gave a good answer to '{state.get('active_question')}'. Generate a 3-word positive acknowledgement."
        transition = llm.invoke([HumanMessage(content=transition_prompt)]).content.strip().replace('"','')
        final_q = f"{transition}. {base_question}"
    else:
        # If we are forced to move on despite a bad answer (Loop breaking)
        final_q = f"Okay, let's move on to a different topic. {base_question}"
    
    return {
        **state, 
        "messages": state["messages"] + [{"role": "assistant", "content": final_q}], 
        "question_count": state["question_count"] + 1,
        "used_questions": used + [base_question],
        "active_question": base_question,
        "retry_count": 0 # Reset retry count for new question
    }

# --- NODE 3: HANDLE SPECIAL (FIXED LOGIC) ---
def handle_special(state: InterviewState) -> InterviewState:
    target_topic = state.get("active_question", "the interview")
    user_input = state["messages"][-1]["content"]
    persona = state["persona_detected"]
    role = state["role"]
    
    sys_prompt = "You are an Interviewer. Be professional."
    
    # LOGIC: How to handle specific issues
    if persona == "Resisting":
        # User tried to switch roles. BLOCK IT.
        task = f"User wants to switch roles or is refusing. Politely state: 'I am currently set up to conduct the {role} interview. Let's continue with that.' Then repeat: '{target_topic}'."
    
    elif persona == "Distracted":
        # FIX: Do NOT simplify. Just wait.
        task = f"User is distracted (e.g. dog barking, noise). Say something empathetic like 'No problem, I can wait.' Then gently repeat the EXACT SAME question: '{target_topic}'."
    
    elif persona == "Confused":
        # FIX: Simplify, but don't loop forever.
        task = f"User doesn't understand: '{target_topic}'. Explain the concept simply and ask them to try again."
        
    elif state["latest_evaluation"] == "Vague":
        task = f"User answer was too vague. Ask for a specific example related to: '{target_topic}'."
        
    else:
        task = f"User went off-topic. Politely steer them back to: '{target_topic}'."

    reply = llm.invoke([SystemMessage(content=sys_prompt), HumanMessage(content=task)]).content.strip().replace('"','')
    
    return {
        **state, 
        "messages": state["messages"] + [{"role": "assistant", "content": reply}],
        "retry_count": state.get("retry_count", 0) + 1
    }

# --- NODE 4: FEEDBACK ---
def generate_feedback(state: InterviewState) -> InterviewState:
    transcript = "\n".join([f"{m['role']}: {m['content']}" for m in state["messages"]])
    prompt = f"Interview Over. Transcript: {transcript}. Generate a short feedback summary score (1-10) for Communication and Technical skills."
    closing = llm.invoke([HumanMessage(content=prompt)]).content
    return {**state, "messages": state["messages"] + [{"role": "assistant", "content": closing}], "is_finished": True}

# --- ROUTING LOGIC ---
def decide_next(state: InterviewState):
    if state["question_count"] == 0: return "ask_new_question"
    if state["question_count"] >= 5: return "generate_feedback"
    
    # LOOP BREAKER: If we have retried the SAME question 2 times, give up and move on
    if state.get("retry_count", 0) >= 2:
        return "ask_new_question"

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