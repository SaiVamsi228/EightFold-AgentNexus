import json
import random
import re
from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3)

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
    retry_count: int
    current_topic_depth: int

# --- QUESTION BANK (Same as before) ---
# ... (Keep your QUESTIONS_DB dictionary here) ...
QUESTIONS_DB = {
  "Software Engineer": {
    "technical": ["Explain process vs thread?", "Design a URL shortener?", "TCP vs UDP?", "Explain Dependency Injection?", "Debug memory leak?", "RESTful API concept?", "SQL vs NoSQL?", "Garbage collection?", "ACID properties?", "Polymorphism?"],
    "behavioral": ["Critical production issue?", "Disagreed with manager?", "Prioritize deadlines?", "Mistake handled?", "Learn tech quickly?"]
  },
  "SDR": {
    "technical": ["Research prospect?", "Gatekeeper strategy?", "Handle objection 'Happy with vendor'?", "Key metrics?", "Qualifying leads?"],
    "behavioral": ["Repeated rejection?", "Difficult sale?", "Missed quota?", "Stay motivated?"]
  },
  "Retail Associate": {
    "technical": ["Return without receipt?", "Coworker stealing?", "Out of stock?", "Messy display?"],
    "behavioral": ["Angry customer?", "Above and beyond?", "Pressure during holidays?"]
  }
}

def load_questions(role: str):
    return QUESTIONS_DB.get(role, QUESTIONS_DB["Software Engineer"])

# --- NODE 1: ANALYZE ( Improved ) ---
def analyze_input(state: InterviewState) -> InterviewState:
    if not state["messages"]:
        return {**state, "persona_detected": "Normal", "latest_evaluation": "Good"}

    user_input = state["messages"][-1]["content"]
    last_q = next((m["content"] for m in reversed(state["messages"]) if m["role"] == "assistant"), "Start of Interview")

    prompt = f"""
    You are an Interview Analyst.
    Context (Last Question): "{last_q}"
    Candidate Response: "{user_input}"
    
    Analyze the response:
    1. PERSONA:
       - "Distracted": Mentions dogs, food, noise, weather, or unrelated life stories.
       - "Confused": Says "I don't understand", "Clarify", "What?".
       - "Efficient": One word answer like "Pass", "Next".
       - "Normal": Attempts to answer the question.
    
    2. EVALUATION:
       - "Off-topic": Completely unrelated (e.g., talking about a dog).
       - "Vague": Too short to be useful.
       - "Good": Relevant attempt.
    
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

# --- NODE 2: ASK NEW QUESTION (Direct Speech) ---
def ask_new_question(state: InterviewState) -> InterviewState:
    questions = load_questions(state["role"])
    q_type = "behavioral" if state["question_count"] % 2 == 0 else "technical"
    
    used = state.get("used_questions", [])
    available = [q for q in questions[q_type] if q not in used]
    if not available: available = questions[q_type]
    
    base_question = random.choice(available)

    # Use LLM to generate the *exact* spoken text to ensure flow
    transition_prompt = ""
    if state["question_count"] == 0:
        transition_prompt = f"You are starting a {state['role']} interview. Say: 'Great. Let's start the {state['role']} interview.' followed immediately by the question: '{base_question}'."
    elif state["persona_detected"] == "Efficient":
        transition_prompt = f"Ask this question directly with no fluff: '{base_question}'"
    else:
        transition_prompt = f"Acknowledge the previous answer briefly (e.g. 'Got it' or 'Makes sense'), then ask: '{base_question}'."

    final_q = llm.invoke([HumanMessage(content=transition_prompt)]).content.strip().replace('"', '')

    return {
        **state,
        "messages": state["messages"] + [{"role": "assistant", "content": final_q}],
        "question_count": state["question_count"] + 1,
        "used_questions": used + [base_question],
        "active_question": base_question
    }

# --- NODE 3: HANDLE SPECIAL (Strict Speech Control) ---
def handle_special(state: InterviewState) -> InterviewState:
    last_q = state.get("active_question", "the topic")
    user_input = state["messages"][-1]["content"]
    persona = state["persona_detected"]

    # Explicit instructions for exact speech
    instruction = ""
    if persona == "Confused":
        instruction = f"The user is confused by '{last_q}'. Explain the concept simply in 1 sentence, then ask them to try again."
    elif persona == "Distracted" or state["latest_evaluation"] == "Off-topic":
        instruction = f"The user is distracted by '{user_input}'. Say something empathetic like 'I understand that can be distracting', but then FIRMLY say: 'Let's get back to the question: {last_q}'."
    elif state["latest_evaluation"] == "Vague":
        instruction = f"The user's answer was too short. Ask: 'Could you give me a specific example for that?'"
    else:
        instruction = f"Politely steer the user back to the question: '{last_q}'."

    prompt = f"""
    You are an AI Interviewer. 
    Instruction: {instruction}
    
    Output ONLY the text you want to speak to the candidate. Do not add "Assistant:" or quotes.
    """

    reply = llm.invoke([HumanMessage(content=prompt)]).content.strip().replace('"', '')
    
    return {**state, "messages": state["messages"] + [{"role": "assistant", "content": reply}]}

# --- NODE 4: FEEDBACK (JSON) ---
def generate_feedback(state: InterviewState) -> InterviewState:
    transcript = "\n".join([f"{m['role']}: {m['content']}" for m in state['messages'] if m['role'] != 'system'])
    
    prompt = f"""
    Act as a Hiring Manager for {state['role']}. Review:
    {transcript}
    
    Return VALID JSON ONLY:
    {{
        "score": "X/10",
        "confidence": "High/Medium/Low",
        "strengths": ["Point 1", "Point 2"],
        "improvements": ["Point 1", "Point 2"],
        "summary": "Executive summary.",
        "recommendation": "Hire/No Hire"
    }}
    """
    try:
        resp = llm.invoke([HumanMessage(content=prompt)]).content.strip()
        clean_json = resp.replace("```json", "").replace("```", "").strip()
        feedback_data = json.loads(clean_json)
    except:
        feedback_data = {"score": "N/A", "summary": "Error analyzing.", "recommendation": "N/A", "confidence": "Low", "strengths": [], "improvements": []}

    return {
        **state,
        "messages": state["messages"] + [{"role": "assistant", "content": "Interview complete. Feedback generated."}],
        "feedback": feedback_data,
        "is_finished": True
    }

# --- ROUTING ---
def decide_next(state: InterviewState):
    if state["question_count"] == 0: return "ask_new_question"
    if state["question_count"] >= 5: return "generate_feedback"
    
    # Prioritize Persona Detection for routing
    if state["persona_detected"] in ["Confused", "Distracted", "Edge"]:
        return "handle_special"
    
    if state["latest_evaluation"] in ["Vague", "Off-topic"]:
        return "handle_special"
        
    return "ask_new_question"

# --- COMPILE ---
workflow = StateGraph(InterviewState)
workflow.add_node("analyze", analyze_input)
workflow.add_node("ask_new_question", ask_new_question)
workflow.add_node("handle_special", handle_special)
workflow.add_node("generate_feedback", generate_feedback)

workflow.set_entry_point("analyze")
workflow.add_conditional_edges("analyze", decide_next, {
    "ask_new_question": "ask_new_question", 
    "handle_special": "handle_special", 
    "generate_feedback": "generate_feedback"
})
workflow.add_edge("ask_new_question", END)
workflow.add_edge("handle_special", END)
workflow.add_edge("generate_feedback", END)

app_graph = workflow.compile()