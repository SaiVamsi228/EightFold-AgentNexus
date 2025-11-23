import json
import re
from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()

# LLM instance
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.4)

# Maximum follow-up depth
MAX_DEPTH = 3

# --- STATE MODEL ---
class InterviewState(TypedDict):
    messages: List[Dict[str, str]]
    role: str
    question_count: int
    persona_detected: str
    latest_evaluation: str
    is_finished: bool
    feedback: str
    used_questions: List[str]
    active_question: str
    retry_count: int
    current_topic_depth: int

# ============================================================
# ✅ NODE 1 (UPDATED COMPLETELY): ANALYZE INPUT
# ============================================================
def analyze_input(state: InterviewState) -> InterviewState:
    # If no messages yet → skip analysis
    if not state["messages"]:
        return {**state, "persona_detected": "Normal", "latest_evaluation": "Good"}

    user_input = state["messages"][-1]["content"]

    # Immediate Exit
    if any(k in user_input.lower() for k in ["end interview", "stop", "quit", "finish"]):
        return {**state, "persona_detected": "End_Session", "is_finished": True}

    # SIMPLE ANALYSIS PROMPT
    current_context = state.get("active_question", "Introduction")

    prompt = f"""
    Role: {state['role']}
    Question asked: "{current_context}"
    User Answer: "{user_input}"
    
    Task: Analyze the user's answer.
    
    Classify Persona:
    - Confused
    - Distracted
    - Normal
    
    Classify Evaluation:
    - Good
    - Vague
    - Off-topic
    
    Return JSON only: {{ "persona": "...", "evaluation": "..." }}
    """

    try:
        resp = llm.invoke([HumanMessage(content=prompt)])
        cleaned = resp.content.replace("```json", "").replace("```", "")
        data = json.loads(re.search(r"\{.*\}", cleaned, re.DOTALL).group(0))
    except:
        data = {"persona": "Normal", "evaluation": "Good"}

    return {
        **state,
        "persona_detected": data.get("persona", "Normal"),
        "latest_evaluation": data.get("evaluation", "Good")
    }

# ============================================================
# NODE 2: ASK NEW QUESTION
# ============================================================
def ask_new_question(state: InterviewState) -> InterviewState:
    if state["role"] == "Unknown":
        return {
            **state,
            "messages": state["messages"] + [{
                "role": "assistant",
                "content": "Hi! What job role are you targeting today?"
            }],
        }

    q_type = "behavioral" if state["question_count"] % 2 == 0 else "technical"
    avoid_topics = ", ".join(state.get("used_questions", [])[-3:])

    prompt = f"""
    Generate a {q_type} interview question for a {state['role']}.
    Avoid: {avoid_topics}.
    Only output the question.
    """

    new_q = llm.invoke([HumanMessage(content=prompt)]).content.strip()

    prefix = ""
    if state["question_count"] == 0:
        prefix = f"Great. Let's start your {state['role']} interview. "
    elif state["latest_evaluation"] == "Good":
        prefix = "Good answer. Let's move on. "
    else:
        prefix = "Okay, let's switch gears. "

    return {
        **state,
        "messages": state["messages"] + [{
            "role": "assistant",
            "content": prefix + new_q
        }],
        "question_count": state["question_count"] + 1,
        "used_questions": state["used_questions"] + [new_q],
        "active_question": new_q,
        "retry_count": 0,
        "current_topic_depth": 1
    }

# ============================================================
# NODE 3: FOLLOW-UP QUESTION
# ============================================================
def ask_follow_up(state: InterviewState) -> InterviewState:
    last = state["messages"][-1]["content"]
    q = state.get("active_question")

    prompt = f"""
    Role: {state['role']}
    Main Question: "{q}"
    User Answer: "{last}"

    Generate a deeper follow-up question.
    """

    follow = llm.invoke([HumanMessage(content=prompt)]).content.strip()

    return {
        **state,
        "messages": state["messages"] + [{"role": "assistant", "content": follow}],
        "current_topic_depth": state["current_topic_depth"] + 1,
        "retry_count": 0
    }

# ============================================================
# ✅ NODE 4 (UPDATED COMPLETELY): HANDLE SPECIAL CASES
# ============================================================
def handle_special(state: InterviewState) -> InterviewState:
    target = state.get("active_question", "the interview")
    persona = state["persona_detected"]

    if persona == "Confused":
        instruction = f"The user is confused about '{target}'. Explain it simply. Do NOT ask a new question."
    elif persona == "Distracted":
        instruction = f"The user is distracted. Politely acknowledge it and repeat the question '{target}'."
    elif state["latest_evaluation"] == "Vague":
        instruction = f"The answer was vague. Ask them for a specific example related to '{target}'."
    else:
        instruction = f"The user is off-topic. Bring them back to the question '{target}'."

    prompt = f"""
    You are the Interviewer.
    Instruction: {instruction}
    Generate a concise professional reply.
    """

    reply = llm.invoke([HumanMessage(content=prompt)]).content.strip()

    return {
        **state,
        "messages": state["messages"] + [{"role": "assistant", "content": reply}],
        "retry_count": state["retry_count"] + 1
    }

# ============================================================
# NODE 5: GENERATE FEEDBACK
# ============================================================
def generate_feedback(state: InterviewState) -> InterviewState:
    transcript = "\n".join([f"{m['role']}: {m['content']}" for m in state["messages"]])

    prompt = f"""
    Act as a Hiring Manager for {state['role']}.
    Review this transcript:

    {transcript}

    Generate detailed Markdown feedback.
    """

    fb = llm.invoke([HumanMessage(content=prompt)]).content

    return {
        **state,
        "messages": state["messages"] + [
            {"role": "assistant", "content": "Thanks! I've prepared your feedback."}
        ],
        "feedback": fb,
        "is_finished": True
    }

# ============================================================
# ✅ UPDATED ROUTER (APPLIES YOUR LOGIC)
# ============================================================
def decide_next(state: InterviewState):
    if state["persona_detected"] == "End_Session":
        return "generate_feedback"

    # Force first question immediately
    if state["question_count"] == 0:
        return "ask_new_question"

    if state["question_count"] >= 5:
        return "generate_feedback"

    if state["persona_detected"] in ["Confused", "Distracted", "Off-topic"]:
        if state["retry_count"] >= 2:
            return "ask_new_question"
        return "handle_special"

    if state["latest_evaluation"] == "Good":
        if state["current_topic_depth"] < MAX_DEPTH:
            return "ask_follow_up"
        return "ask_new_question"

    return "handle_special"

# ============================================================
# GRAPH BUILD
# ============================================================
workflow = StateGraph(InterviewState)
workflow.add_node("analyze", analyze_input)
workflow.add_node("ask_new_question", ask_new_question)
workflow.add_node("ask_follow_up", ask_follow_up)
workflow.add_node("handle_special", handle_special)
workflow.add_node("generate_feedback", generate_feedback)

workflow.set_entry_point("analyze")

workflow.add_conditional_edges("analyze", decide_next, {
    "ask_new_question": "ask_new_question",
    "ask_follow_up": "ask_follow_up",
    "handle_special": "handle_special",
    "generate_feedback": "generate_feedback",
})

workflow.add_edge("ask_new_question", END)
workflow.add_edge("ask_follow_up", END)
workflow.add_edge("handle_special", END)
workflow.add_edge("generate_feedback", END)

app_graph = workflow.compile()
