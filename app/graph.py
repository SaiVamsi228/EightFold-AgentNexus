import json
import random
from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
import os
from dotenv import load_dotenv

load_dotenv()
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.6)

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

# NODE 1: ANALYZE
def analyze_input(state: InterviewState) -> InterviewState:
    history = "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in state["messages"][-10:]])
    user_input = state["messages"][-1]["content"]

    prompt = f"""
You are analyzing the candidate's latest answer.

Recent conversation:
{history}

Latest user message: "{user_input}"

Return ONLY valid JSON:
{{
  "persona": "Confused|Chatty|Efficient|Edge|Normal",
  "evaluation": "Good|Vague|Off-topic"
}}
"""
    try:
        resp = llm.invoke([SystemMessage(content=prompt)])
        cleaned = resp.content.strip().replace("```json", "").replace("```", "")
        data = json.loads(cleaned)
    except:
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
    if state["latest_evaluation"] in ["Vague", "Off-topic"] or state["persona_detected"] == "Chatty":
        prompt = f"Candidate is vague/off-topic/chatty. Last question was: '{last_q}'. Ask one focused follow-up or polite redirect."
    else:
        prompt = "Gently guide the candidate back."

    reply = llm.invoke([HumanMessage(content=prompt)]).content
    return {**state, "messages": state["messages"] + [{"role": "assistant", "content": reply}]}

# NODE 4: FEEDBACK
def generate_feedback(state: InterviewState) -> InterviewState:
    transcript = "\n".join([f"{m['role']}: {m['content']}" for m in state["messages"]])
    prompt = f"""
Interview over for {state['role']}.
Transcript:
{transcript}

Return JSON with closing statement only (spoken):
"Overall you scored X/10. Strengths: ... Improvements: ..."
"""
    closing = llm.invoke([HumanMessage(content=prompt)]).content
    return {
        **state,
        "messages": state["messages"] + [{"role": "assistant", "content": closing}],
        "is_finished": True
    }

# DECISION FUNCTION (FIXED - NO LAMBDA SYNTAX ERROR)
def decide_next(state: InterviewState):
    if state["question_count"] >= random.randint(6, 9):
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