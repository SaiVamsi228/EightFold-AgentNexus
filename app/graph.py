import json
import random
from typing import TypedDict, List, Optional
from dotenv import load_dotenv

from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# Load env variables
load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.6)

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
    
    # --- BUG FIX: REMOVED THE LENGTH CHECK HERE ---
    # We always analyze the input now, even if it's the only message we see.

    system_prompt = """
    Analyze the user's latest response in this interview context.
    1. Detect Persona: 
       - "Confused" (asks for clarification, says I don't know, hesitant)
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
        # Clean up json string if LLM adds markdown
        content = response.content.replace("```json", "").replace("```", "")
        analysis = json.loads(content)
        
        return {
            "persona_detected": analysis.get("persona", "Normal"),
            "latest_evaluation": analysis.get("evaluation", "Good")
        }
    except:
        return {"persona_detected": "Normal", "latest_evaluation": "Good"}

# --- NODE 2: DECISION MAKER ---
def determine_next_step(state: InterviewState):
    if state["question_count"] >= 6:
        return "generate_feedback"
    
    if state["latest_evaluation"] == "Vague":
        return "ask_followup"
        
    if state["persona_detected"] in ["Chatty", "Off-topic"]:
        return "redirect_user"

    return "ask_question"

# --- NODE 3: ASK QUESTION ---
def ask_question(state: InterviewState):
    role_data = load_questions(state["role"])
    
    new_type = "technical" if state["current_question_type"] == "behavioral" else "behavioral"
    
    # Pick random question
    raw_question = random.choice(role_data[new_type])
    
    # DYNAMIC REPHRASING based on Persona
    final_q = raw_question
    
    if state["persona_detected"] == "Confused":
        prompt = f"The candidate is confused. Rewrite this question to be simpler, reassuring, and easier to answer: '{raw_question}'"
        final_q = llm.invoke([HumanMessage(content=prompt)]).content
        
    elif state["persona_detected"] == "Efficient":
        final_q = raw_question # Keep it direct
        
    elif state["persona_detected"] == "Chatty":
        prompt = f"The candidate is rambling. Create a polite bridge that interrupts them and pivots to this question: '{raw_question}'"
        final_q = llm.invoke([HumanMessage(content=prompt)]).content

    return {
        "messages": [AIMessage(content=final_q)],
        "question_count": state["question_count"] + 1,
        "current_question_type": new_type
    }

# --- NODE 4: FOLLOW UP / REDIRECT ---
def handle_special_case(state: InterviewState):
    last_msg = state["messages"][-1]["content"]
    
    if state["latest_evaluation"] == "Vague":
        prompt = f"The user gave a vague answer to: '{last_msg}'. Ask a specific follow-up question to dig deeper."
    else:
        prompt = f"The user is going off-topic. Generate a polite transition back to the interview."
        
    resp = llm.invoke([HumanMessage(content=prompt)])
    return {"messages": [resp]} 

# --- NODE 5: FEEDBACK GENERATOR ---
def generate_feedback(state: InterviewState):
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
        content = resp.content.replace("```json", "").replace("```", "")
        feedback_data = json.loads(content)
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