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
    feedback: Any        # UPDATED: now stores Markdown feedback
    used_questions: List[str]
    active_question: str
    retry_count: int     # Track retries

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
    return QUESTIONS_DB.get(role, QUESTIONS_DB["Software Engineer"])

# --- NODE 1: ANALYZE ---
def analyze_input(state: InterviewState) -> InterviewState:
    user_input = state["messages"][-1]["content"]
    current_context = state.get("active_question") or "Introduction"

    prompt = f"""
    You are an Interview Analyst.

    Current Role: {state['role']}
    Current Question: "{current_context}"
    User Input: "{user_input}"

    Classify:
    1. PERSONA: Distracted / Confused / Resisting / Normal
    2. EVALUATION: Good / Vague / Off-topic

    Return JSON ONLY: {{"persona":"...", "evaluation":"..."}}
    """

    try:
        resp = llm.invoke([HumanMessage(content=prompt)])
        cleaned = resp.content.strip().replace("```json","").replace("```","")
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        data = json.loads(match.group(0)) if match else {"persona":"Normal","evaluation":"Good"}
    except:
        data = {"persona":"Normal","evaluation":"Good"}

    if state["question_count"] == 0:
        data["evaluation"] = "Good"

    return {
        **state,
        "persona_detected": data["persona"],
        "latest_evaluation": data["evaluation"]
    }

# --- NODE 2: ASK NEW QUESTION ---
def ask_new_question(state: InterviewState) -> InterviewState:
    questions = load_questions(state["role"])
    q_type = "behavioral" if state["question_count"] % 2 == 0 else "technical"

    used = state["used_questions"]
    available = [q for q in questions[q_type] if q not in used]
    if not available:
        available = questions[q_type]

    base_question = random.choice(available)

    if state["question_count"] == 0:
        final = f"Great. Let's start the {state['role']} interview. {base_question}"
    elif state["latest_evaluation"] == "Good":
        transition_prompt = f"User answered '{state.get('active_question')}'. Give a 3-word positive acknowledgement."
        transition = llm.invoke([HumanMessage(content=transition_prompt)]).content.strip().replace('"','')
        final = f"{transition}. {base_question}"
    else:
        final = f"Okay, let's move on. {base_question}"

    return {
        **state,
        "messages": state["messages"] + [{"role":"assistant","content":final}],
        "question_count": state["question_count"] + 1,
        "used_questions": used + [base_question],
        "active_question": base_question,
        "retry_count": 0
    }

# --- NODE 3: HANDLE SPECIAL ---
def handle_special(state: InterviewState) -> InterviewState:
    target = state.get("active_question", "the interview")
    persona = state["persona_detected"]
    role = state["role"]

    if persona == "Resisting":
        task = f"I am currently set up to conduct the {role} interview. Let's continue. Here's the question again: '{target}'."

    elif persona == "Distracted":
        task = f"No problem, I can wait. Let's continue. '{target}'"

    elif persona == "Confused":
        task = f"Let me explain the question simply and then try again: '{target}'"

    elif state["latest_evaluation"] == "Vague":
        task = f"Your answer was vague. Please give a specific example for: '{target}'"

    else:
        task = f"Let's refocus. The question is: '{target}'"

    reply = llm.invoke([HumanMessage(content=task)]).content.strip()
    return {
        **state,
        "messages": state["messages"] + [{"role":"assistant","content":reply}],
        "retry_count": state["retry_count"] + 1
    }

# --- NEW UPDATED FEEDBACK NODE (from File 2) ---
def generate_feedback(state: InterviewState) -> InterviewState:
    transcript = "\n".join([f"{m['role']}: {m['content']}" for m in state['messages']])

    prompt = f"""
    Act as a Hiring Manager for a {state['role']} position.
    Review this transcript:

    {transcript}

    Generate a Markdown formatted interview feedback report containing:
    # Interview Feedback: {state['role']}
    ## 1. Executive Summary
    ## 2. Strengths
    ## 3. Areas for Improvement
    ## 4. Hiring Recommendation
    """

    report = llm.invoke([HumanMessage(content=prompt)]).content

    return {
        **state,
        "messages": state["messages"] + [{"role":"assistant","content":"Thank you. Here is your detailed feedback report."}],
        "feedback": report,
        "is_finished": True
    }

# --- ROUTER ---
def decide_next(state: InterviewState):
    if state["question_count"] == 0:
        return "ask_new_question"

    if state["question_count"] >= 5:
        return "generate_feedback"

    if state["retry_count"] >= 2:
        return "ask_new_question"

    if state["latest_evaluation"] == "Good":
        return "ask_new_question"

    return "handle_special"

# --- WORKFLOW ---
workflow = StateGraph(InterviewState)

workflow.add_node("analyze", analyze_input)
workflow.add_node("ask_new_question", ask_new_question)
workflow.add_node("handle_special", handle_special)
workflow.add_node("generate_feedback", generate_feedback)

workflow.set_entry_point("analyze")

workflow.add_conditional_edges(
    "analyze",
    decide_next,
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
