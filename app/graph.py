# app/graph.py (Fixed persona detection: Better prompt, logging, and conditional for Confused/Vague)
import json
import random
from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
import os
from dotenv import load_dotenv
from app.prompts import SYSTEM_ANALYSIS_PROMPT  # Import the separate prompt for consistency

load_dotenv()
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.6)  # Stable model

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

# NODE 1: ANALYZE (FIXED: Use SYSTEM_ANALYSIS_PROMPT + force JSON + better detection for nervous/vague)
def analyze_input(state: InterviewState) -> InterviewState:
    history = "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in state["messages"][-10:]])
    user_input = state["messages"][-1]["content"]

    # Check if first message (role selection)
    if "engineer" in user_input.lower() or "sdr" in user_input.lower() or "retail" in user_input.lower():
        detected = {"persona": "Normal", "evaluation": "Good"}
    else:
        # Use the separate prompt
        analysis_prompt = SYSTEM_ANALYSIS_PROMPT.format(history=history, user_input=user_input)
        try:
            resp = llm.invoke([SystemMessage(content=analysis_prompt), HumanMessage(content=user_input)])
            cleaned = resp.content.strip().replace("```json", "").replace("```", "").replace("json", "")
            detected = json.loads(cleaned)
        except Exception as e:
            print("Analyze error:", str(e))
            detected = {"persona": "Normal", "evaluation": "Good"}

    print(f"DETECTED PERSONA: {detected['persona']}, EVALUATION: {detected['evaluation']} for input: {user_input}")  # LOG FOR DEBUG

    return {
        **state,
        "persona_detected": detected["persona"],
        "latest_evaluation": detected["evaluation"]
    }

# NODE 2: ASK NEW QUESTION (No change, but added logging)
def ask_new_question(state: InterviewState) -> InterviewState:
    questions = load_questions(state["role"])
    q_type = "behavioral" if state["question_count"] % 2 == 0 else "technical"
    question = random.choice(questions[q_type])

    if state["persona_detected"] == "Confused":
        rephrased = llm.invoke([HumanMessage(content=f"Make this question much simpler and reassuring: {question}")]).content
        question = rephrased
        print("REPHRASED FOR CONFUSED:", question)
    elif state["persona_detected"] == "Chatty":
        rephrased = llm.invoke([HumanMessage(content=f"Politely refocus the candidate on this exact question: {question}")]).content
        question = rephrased
        print("REPHRASED FOR CHATTY:", question)

    new_messages = state["messages"] + [{"role": "assistant", "content": question}]
    print(f"NEW QUESTION ASKED (Persona: {state['persona_detected']}): {question}")
    return {**state, "messages": new_messages, "question_count": state["question_count"] + 1}

# NODE 3: HANDLE SPECIAL (Enhanced prompt for Confused/Vague)
def handle_special(state: InterviewState) -> InterviewState:
    last_q = next((m["content"] for m in reversed(state["messages"]) if m["role"] == "assistant"), "")
    if state["latest_evaluation"] in ["Vague", "Off-topic"] or state["persona_detected"] in ["Chatty", "Edge", "Confused"]:
        prompt = f"Candidate is {state['persona_detected']} with {state['latest_evaluation']} answer. Last question: '{last_q}'. Generate a warm, professional response: 
- For Confused: Reassure and simplify/rephrase the question.
- For Vague: Ask a targeted follow-up for specific example.
- For Off-topic/Chatty/Edge: Politely redirect back to the question.
Keep it natural for voice, end with the follow-up question."
    else:
        prompt = "Gently guide the candidate back to the interview with a follow-up."

    reply = llm.invoke([HumanMessage(content=prompt)]).content
    print(f"HANDLE SPECIAL (Persona: {state['persona_detected']}, Eval: {state['latest_evaluation']}): {reply}")
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
    print("FEEDBACK GENERATED:", closing)
    return {
        **state,
        "messages": state["messages"] + [{"role": "assistant", "content": closing}],
        "is_finished": True
    }

# DECISION FUNCTION (FIXED: Trigger handle_special for Confused + Vague/Off-topic)
def decide_next(state: InterviewState):
    print(f"DECISION: Q count={state['question_count']}, Persona={state['persona_detected']}, Eval={state['latest_evaluation']}")
    if state["question_count"] >= 6:
        return "generate_feedback"
    if state["latest_evaluation"] in ["Vague", "Off-topic"] or state["persona_detected"] in ["Chatty", "Edge", "Confused"]:
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