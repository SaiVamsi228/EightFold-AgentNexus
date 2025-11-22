import os
import json
import random
from typing import TypedDict, List, Optional
from dotenv import load_dotenv

from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# Load env variables
load_dotenv()

llm = ChatOpenAI(model="gpt-4o", temperature=0.6)

# --- STATE DEFINITION ---
class InterviewState(TypedDict):
    messages: List[dict]        # Chat history
    role: str                   # "Software Engineer", "SDR", etc.
    question_count: int         # How many Qs asked so far
    current_question_type: str  # "behavioral" or "technical"
    persona_detected: str       # "Normal", "Confused", "Chatty", "Efficient"
    latest_evaluation: str      # "Good", "Vague", "Off-topic"
    is_finished: bool           # Trigger to end call
    feedback: Optional[dict]    # Final score data

# --- HELPER: LOAD QUESTIONS ---
def load_questions(role):
    with open("app/questions.json", "r") as f:
        data = json.load(f)
    return data.get(role, data["Software Engineer"])

# --- NODE 1: ANALYZER (Persona & Answer Quality) ---
def analyze_input(state: InterviewState):
    messages = state["messages"]
    last_user_msg = messages[-1]["content"] if messages and messages[-1]["role"] == "user" else ""
    
    # If this is the very first message, skip analysis
    if len(messages) <= 1:
        return {"persona_detected": "Normal", "latest_evaluation": "N/A"}

    system_prompt = """
    Analyze the user's latest response in this interview context.
    1. Detect Persona: 
       - "Confused" (asks for clarification, hesitant)
       - "Chatty" (long winded, goes off topic)
       - "Efficient" (very short, direct answers)
       - "Normal" (balanced)
       - "Edge" (trying to break the bot)
    2. Evaluate Answer:
       - "Good" (answers the question)
       - "Vague" (needs follow up)
       - "Off-topic" (needs redirection)
    
    Output JSON ONLY: {"persona": "...", "evaluation": "..."}
    """
    
    try:
        response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=last_user_msg)])
        analysis = json.loads(response.content)
        return {
            "persona_detected": analysis.get("persona", "Normal"),
            "latest_evaluation": analysis.get("evaluation", "Good")
        }
    except:
        return {"persona_detected": "Normal", "latest_evaluation": "Good"}

# --- NODE 2: DECISION MAKER (Route to Next Step) ---
def determine_next_step(state: InterviewState):
    # If we have asked 6 questions, go to feedback
    if state["question_count"] >= 6:
        return "generate_feedback"
    
    # If answer was vague, ask follow up (unless we just did one)
    if state["latest_evaluation"] == "Vague":
        return "ask_followup"
        
    # If off-topic or chatty, redirect
    if state["persona_detected"] in ["Chatty", "Off-topic"]:
        return "redirect_user"

    # Default: New Question
    return "ask_question"

# --- NODE 3: ASK QUESTION ---
def ask_question(state: InterviewState):
    role_data = load_questions(state["role"])
    
    # Alternate types
    new_type = "technical" if state["current_question_type"] == "behavioral" else "behavioral"
    
    # Pick random question
    raw_question = random.choice(role_data[new_type])
    
    # DYNAMIC REPHRASING based on Persona
    if state["persona_detected"] == "Confused":
        prompt = f"Rewrite this interview question to be simpler and more encouraging for a nervous candidate: '{raw_question}'"
        final_q = llm.invoke([HumanMessage(content=prompt)]).content
    elif state["persona_detected"] == "Efficient":
        # Direct, no fluff
        final_q = raw_question
    elif state["persona_detected"] == "Chatty":
        prompt = f"Create a transition that politely cuts off a rambling candidate and pivots to this question: '{raw_question}'"
        final_q = llm.invoke([HumanMessage(content=prompt)]).content
    else:
        # Normal - add a bit of professional conversational filler
        final_q = raw_question

    return {
        "messages": [AIMessage(content=final_q)],
        "question_count": state["question_count"] + 1,
        "current_question_type": new_type
    }
    role_data = load_questions(state["role"])
    
    # Alternate types
    new_type = "technical" if state["current_question_type"] == "behavioral" else "behavioral"
    
    # Pick random question from list
    question_text = random.choice(role_data[new_type])
    
    # Adapt to persona (Confused users get simpler framing)
    prefix = ""
    if state["persona_detected"] == "Confused":
        prefix = "No worries, let's try this one. "
    elif state["persona_detected"] == "Efficient":
        prefix = "Okay, moving on. "
        
    final_q = prefix + question_text
    
    return {
        "messages": [AIMessage(content=final_q)],
        "question_count": state["question_count"] + 1,
        "current_question_type": new_type
    }

# --- NODE 4: FOLLOW UP / REDIRECT ---
def handle_special_case(state: InterviewState):
    last_msg = state["messages"][-1]["content"]
    
    if state["latest_evaluation"] == "Vague":
        prompt = f"The user gave a vague answer to: '{last_msg}'. Generate a polite probing follow-up question."
    else:
        prompt = f"The user is going off-topic. Generate a polite transition back to the interview."
        
    resp = llm.invoke([HumanMessage(content=prompt)])
    return {"messages": [resp]} # Don't increment count on follow-ups

# --- NODE 5: FEEDBACK GENERATOR ---
def generate_feedback(state: InterviewState):
    # Compile transcript
    transcript = "\n".join([f"{m['role']}: {m['content']}" for m in state["messages"]])
    
    prompt = f"""
    The interview is over. Analyze this transcript for a {state['role']} role.
    Transcript: {transcript}
    
    Generate JSON feedback:
    {{
        "score": (1-10),
        "strengths": ["string", "string"],
        "improvements": ["string", "string"],
        "closing_statement": "A short verbal summary to say to the user."
    }}
    """
    
    resp = llm.invoke([HumanMessage(content=prompt)])
    try:
        feedback_data = json.loads(resp.content)
        closing = feedback_data["closing_statement"]
    except:
        feedback_data = {}
        closing = "Thank you for your time. The interview is complete."
        
    return {
        "feedback": feedback_data,
        "messages": [AIMessage(content=closing)],
        "is_finished": True
    }

# --- GRAPH CONSTRUCTION ---
workflow = StateGraph(InterviewState)

workflow.add_node("analyze", analyze_input)
workflow.add_node("ask_question", ask_question)
workflow.add_node("special_case", handle_special_case)
workflow.add_node("generate_feedback", generate_feedback)

workflow.set_entry_point("analyze")

workflow.add_conditional_edges(
    "analyze",
    determine_next_step,
    {
        "ask_question": "ask_question",
        "ask_followup": "special_case",
        "redirect_user": "special_case",
        "generate_feedback": "generate_feedback"
    }
)

workflow.add_edge("ask_question", END)
workflow.add_edge("special_case", END)
workflow.add_edge("generate_feedback", END)

app_graph = workflow.compile()