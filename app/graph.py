# app/graph.py (Updated model to "gemini-pro" to fix 404 error)
import json
import random
from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
import os
from dotenv import load_dotenv

load_dotenv()
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-001", temperature=0.6)

class InterviewState(TypedDict):
    messages: List[Dict[str, str]]
    role: str
    question_count: int
    persona_detected: str
    latest_evaluation: str
    is_finished: bool
    feedback: Dict[str, Any]

# Load questions
def load_questions(role: str):
    with open("app/questions.json") as f:
        data = json.load(f)
    return data.get(role, data["Software Engineer"])

# NODE 1: ANALYZE (Already fixed with HumanMessage)
def analyze_input(state: InterviewState) -> InterviewState:
    history = "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in state["messages"][-10:]])
    user_input = state["messages"][-1]["content"]

    prompt = f"""
You are analyzing the candidate's latest response in a mock interview.

If this is the first message or the user is choosing the role (e.g., 'software engineer'), return {{ "persona": "Normal", "evaluation": "Good" }}.

Otherwise, based on recent conversation:
{history}

Latest user message: "{user_input}"

Classify persona and evaluation.
Persona: Confused (hesitant, nervous, 'um', 'I don't know'), Chatty (rambling, off-topic stories), Efficient (short/direct answers), Edge (invalid/rude/beyond scope), Normal.

Evaluation: Good (answers well, relevant), Vague (needs more detail, follow-up), Off-topic (not related to question).

Return ONLY valid JSON:
{{ "persona": "string", "evaluation": "string" }}
"""
    try:
        resp = llm.invoke([SystemMessage(content=prompt), HumanMessage(content=user_input)])
        cleaned = resp.content.strip().replace("```json", "").replace("```", "")
        data = json.loads(cleaned)
    except Exception as e:
        print("Analyze error:", str(e))
        data = {"persona": "Normal", "evaluation": "Good"}

    return {
        **state,
        "persona_detected": data["persona"],
        "latest_evaluation": data["evaluation"]
    }

# NODE 2: ASK NEW QUESTION
def ask_new_question(state: InterviewState) -> InterviewState:
    questions = load_questions(state["role"])
    q_type = "behavioral" if state["question_count"] % 2 == 0 else "technical"
    question = random.choice(questions[q_type])

    if state["persona_detected"] == "Confused":
        question = llm.invoke([HumanMessage(content=f"Make this question much simpler and reassuring: {question}")]).content
    elif state["persona_detected"] == "Chatty":
        question = llm.invoke([HumanMessage(content=f"Politely refocus the candidate on this exact question: {question}")]).content

    new_messages = state["messages"] + [{"role": "assistant", "content": question}]
    return {**state, "messages": new_messages, "question_count": state["question_count"] + 1}

# NODE 3: HANDLE SPECIAL
def handle_special(state: InterviewState) -> InterviewState:
    last_q = next((m["content"] for m in reversed(state["messages"]) if m["role"] == "assistant"), "")
    if state["latest_evaluation"] in ["Vague", "Off-topic"] or state["persona_detected"] in ["Chatty", "Edge"]:
        prompt = f"Candidate is {state['persona_detected']}/{state['latest_evaluation']}. Last question: '{last_q}'. Generate a warm, professional response: for Confused, simplify/reassure; for Chatty/Edge, redirect politely; for Vague, ask targeted follow-up."
    else:
        prompt = "Gently guide the candidate back to the interview."

    reply = llm.invoke([HumanMessage(content=prompt)]).content
    return {**state, "messages": state["messages"] + [{"role": "assistant", "content": reply}]}

# NODE 4: FEEDBACK
def generate_feedback(state: InterviewState) -> InterviewState:
    transcript = "\n".join([f"{m['role']}: {m['content']}" for m in state["messages"]])
    prompt = f"""
Mock interview complete for {state['role']}.

Transcript:
{transcript}

Generate spoken feedback as a string:
"Great job! Overall score: X/10.
Strengths: - Bullet1 - Bullet2
Areas for improvement: - Bullet1 - Bullet2
Communication: ...
Technical knowledge: ...
Specific examples: ..."

Keep it structured but natural for voice.
"""
    closing = llm.invoke([HumanMessage(content=prompt)]).content
    return {
        **state,
        "messages": state["messages"] + [{"role": "assistant", "content": closing}],
        "is_finished": True
    }

# DECISION FUNCTION
def decide_next(state: InterviewState):
    if state["question_count"] >= 6:
        return "generate_feedback"
    if state["latest_evaluation"] in ["Vague", "Off-topic"] or state["persona_detected"] in ["Chatty", "Edge"]:
        return "handle_special"
    return "ask_new_question"

# GRAPH
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