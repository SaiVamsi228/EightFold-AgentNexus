import json
import random
from typing import TypedDict, List, Optional
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from app.prompts import SYSTEM_ANALYSIS_PROMPT
import os
from dotenv import load_dotenv

load_dotenv()
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.6)

class InterviewState(TypedDict):
    messages: List[dict]
    role: str
    question_count: int
    persona_detected: str
    latest_evaluation: str
    is_finished: bool
    feedback: Optional[dict]

# Load questions
def load_questions(role: str):
    with open("app/questions.json") as f:
        data = json.load(f)
    return data.get(role, data["Software Engineer"])

# Node 1: Analyze persona + answer quality
def analyze_input(state: InterviewState) -> dict:
    history = "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in state["messages"][-8:]])
    user_input = state["messages"][-1]["content"]

    prompt = SYSTEM_ANALYSIS_PROMPT.format(history=history, user_input=user_input)
    try:
        resp = llm.invoke([SystemMessage(content=prompt)])
        cleaned = resp.content.replace("```json", "").replace("```", "").strip()
        result = json.loads(cleaned)
    except:
        result = {"persona": "Normal", "evaluation": "Good"}

    return {
        "persona_detected": result["persona"],
        "latest_evaluation": result["evaluation"]
    }

# Node 2: Decide next action
def decide_next(state: InterviewState) -> str:
    if state["question_count"] >= random.randint(6, 9):
        return "end_interview"
    if state["latest_evaluation"] in ["Vague", "Off-topic"]:
        return "follow_up_or_redirect"
    if state["persona_detected"] == "Chatty":
        return "follow_up_or_redirect"
    return "ask_new_question"

# Node 3: Ask new question (with persona adaptation)
def ask_new_question(state: InterviewState) -> dict:
    questions = load_questions(state["role"])
    q_type = "behavioral" if state["question_count"] % 2 == 0 else "technical"
    question = random.choice(questions[q_type])

    # Persona-based rephrasing
    if state["persona_detected"] == "Confused":
        prompt = f"Make this question simpler and more reassuring: {question}"
        question = llm.invoke([HumanMessage(content=prompt)]).content
    elif state["persona_detected"] == "Chatty":
        prompt = f"Politely interrupt and refocus the candidate on this exact question: {question}"
        question = llm.invoke([HumanMessage(content=prompt)]).content

    return {
        "messages": state["messages"] + [{"role": "assistant", "content": question}],
        "question_count": state["question_count"] + 1
    }

# Node 4: Handle vague/off-topic/chatty
def follow_up_or_redirect(state: InterviewState) -> dict:
    last_q = next((m["content"] for m in reversed(state["messages"]) if m["role"] == "assistant" and "?" in m["content"]), "")
    if state["latest_evaluation"] == "Vague":
        prompt = f"The candidate gave a vague answer to: '{last_q}'\nAsk one specific follow-up question to get a real example."
    else:
        prompt = f"The candidate is going off-topic or too chatty. Politely bring them back to: '{last_q}'"

    response = llm.invoke([HumanMessage(content=prompt)]).content
    return {"messages": state["messages"] + [{"role": "assistant", "content": response}]}

# Node 5: Generate final feedback
def end_interview(state: InterviewState) -> dict:
    transcript = "\n".join([f"{m['role']}: {m['content']}" for m in state["messages"]])
    prompt = f"""
    Mock interview complete for {state['role']}.
    Transcript:
    {transcript}

    Return JSON with:
    - score (1-10)
    - strengths (2-4 bullet strings)
    - improvements (2-4 bullet strings)
    - closing_statement (natural spoken summary)
    """
    try:
        resp = llm.invoke([HumanMessage(content=prompt)])
        data = json.loads(resp.content.replace("```json", "").replace("```", ""))
    except:
        data = {
            "score": 7,
            "strengths": ["Clear communication", "Good energy"],
            "improvements": ["Provide more specific examples", "Structure answers better"],
            "closing_statement": "Great job overall! You scored 7 out of 10."
        }

    closing = data["closing_statement"]
    return {
        "messages": state["messages"] + [{"role": "assistant", "content": closing}],
        "feedback": data,
        "is_finished": True
    }

# Build graph
workflow = StateGraph(InterviewState)
workflow.add_node("analyze", analyze_input)
workflow.add_node("ask_new_question", ask_new_question)
workflow.add_node("follow_up_or_redirect", follow_up_or_redirect)
workflow.add_node("end_interview", end_interview)

workflow.set_entry_point("analyze")
workflow.add_conditional_edges("analyze", decide_next, {
    "ask_new_question": "ask_new_question",
    "follow_up_or_redirect": "follow_up_or_redirect",
    "end_interview": "end_interview"
})
workflow.add_edge("ask_new_question", END)
workflow.add_edge("follow_up_or_redirect", END)
workflow.add_edge("end_interview", END)

app_graph = workflow.compile()