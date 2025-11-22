import json
import random
import re # Added regex for better JSON cleaning
from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
import os
from dotenv import load_dotenv

load_dotenv()

# Use the stable model alias
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash", 
    temperature=0.4 # Lower temp for better JSON consistency
)

class InterviewState(TypedDict):
    messages: List[Dict[str, str]]
    role: str
    question_count: int
    persona_detected: str
    latest_evaluation: str
    is_finished: bool
    feedback: Dict[str, Any]

def load_questions(role: str):
    try:
        with open("app/questions.json") as f:
            data = json.load(f)
        return data.get(role, data["Software Engineer"])
    except:
        return {"behavioral": ["Tell me about yourself."], "technical": ["What is 2+2?"]}

# --- NODE 1: ANALYZE ---
def analyze_input(state: InterviewState) -> InterviewState:
    # Get last few messages for context
    history = "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in state["messages"][-4:]])
    user_input = state["messages"][-1]["content"]

    prompt = f"""
    You are an invisible observer of an interview. Analyze the candidate's latest response.
    
    Context:
    {history}
    
    Latest Candidate Response: "{user_input}"
    
    1. DETECT PERSONA:
    - "Confused": User says "I don't know", "I'm nervous", asks for help, or gives very short/weak answers indicating struggle.
    - "Chatty": User writes 3+ sentences that ramble or go off-topic.
    - "Edge": User is rude, names unrelated things (e.g., "Jim"), or speaks nonsense.
    - "Normal": Standard professional answer.
    
    2. EVALUATE CONTENT:
    - "Good": Relevant answer.
    - "Vague": One or two word answers, or "I guess".
    - "Off-topic": Completely unrelated.
    
    Return ONLY valid JSON:
    {{ "persona": "Confused" | "Chatty" | "Edge" | "Normal", "evaluation": "Good" | "Vague" | "Off-topic" }}
    """
    
    try:
        resp = llm.invoke([HumanMessage(content=prompt)])
        content = resp.content.strip()
        
        # --- FIX: ROBUST JSON CLEANING ---
        # Remove markdown code blocks if present
        if "```" in content:
            content = content.replace("```json", "").replace("```", "")
        
        # Use regex to find the first { and last }
        match = re.search(r'\{.*\}', content, re.DOTALL)
        if match:
            content = match.group(0)
            
        data = json.loads(content)
    except Exception as e:
        print(f"Analyze Error: {e} | Raw output: {content if 'content' in locals() else 'None'}")
        # Fallback
        data = {"persona": "Normal", "evaluation": "Good"}

    print(f"DEBUG: Detected {data['persona']} / {data['evaluation']}")

    return {
        **state,
        "persona_detected": data.get("persona", "Normal"),
        "latest_evaluation": data.get("evaluation", "Good")
    }

# --- NODE 2: ASK NEW QUESTION ---
def ask_new_question(state: InterviewState) -> InterviewState:
    questions = load_questions(state["role"])
    # Simple logic: alternate tech/behavioral
    q_type = "behavioral" if state["question_count"] % 2 == 0 else "technical"
    
    # Pick a random question
    base_question = random.choice(questions[q_type])
    
    # Add to history
    new_messages = state["messages"] + [{"role": "assistant", "content": base_question}]
    
    return {
        **state, 
        "messages": new_messages, 
        "question_count": state["question_count"] + 1
    }

# --- NODE 3: HANDLE SPECIAL ---
def handle_special(state: InterviewState) -> InterviewState:
    last_assistant_msg = next((m["content"] for m in reversed(state["messages"]) if m["role"] == "assistant"), "the previous question")
    
    persona = state["persona_detected"]
    eval_status = state["latest_evaluation"]
    
    prompt = ""
    if persona == "Confused":
        prompt = f"The candidate said they are nervous or stuck on '{last_assistant_msg}'. Be empathetic. Say something reassuring like 'It's okay to take a moment.' Then, ask a much simpler version of the question."
    elif persona == "Edge":
        prompt = f"The candidate said something weird/rude: '{state['messages'][-1]['content']}'. Politely but firmly bring them back to the topic of '{state['role']}' interview."
    elif eval_status == "Vague":
        prompt = f"The candidate gave a vague answer to '{last_assistant_msg}'. Ask a specific follow-up question to get more details."
    else:
        prompt = f"Acknowledge the user's input and gently steer them back to the interview question: {last_assistant_msg}"

    reply = llm.invoke([HumanMessage(content=prompt)]).content
    
    return {**state, "messages": state["messages"] + [{"role": "assistant", "content": reply}]}

# --- NODE 4: FEEDBACK ---
def generate_feedback(state: InterviewState) -> InterviewState:
    transcript = "\n".join([f"{m['role']}: {m['content']}" for m in state["messages"]])
    prompt = f"""
    Interview for {state['role']} is done. 
    Transcript: {transcript}
    
    Provide concise feedback (under 50 words) on their performance.
    """
    closing = llm.invoke([HumanMessage(content=prompt)]).content
    return {
        **state,
        "messages": state["messages"] + [{"role": "assistant", "content": closing}],
        "is_finished": True
    }

# --- DECISION LOGIC ---
def decide_next(state: InterviewState):
    if state["question_count"] >= 6:
        return "generate_feedback"
    
    # FIX: Added "Confused" to the special handling list
    if (state["latest_evaluation"] in ["Vague", "Off-topic"] or 
        state["persona_detected"] in ["Chatty", "Edge", "Confused"]):
        return "handle_special"
        
    return "ask_new_question"

# --- GRAPH SETUP ---
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