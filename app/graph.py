import json
import re
from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()

# Temperature 0.4 gives a balance of creativity and adherence to instructions
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.4)

MAX_DEPTH = 3  # Maximum follow-up questions per topic

# --- STATE DEFINITION ---
class InterviewState(TypedDict):
    messages: List[Dict[str, str]]
    role: str
    question_count: int
    persona_detected: str       # "Confused", "Distracted", "Efficient", "Normal", "End_Session"
    latest_evaluation: str      # "Good", "Vague", "Off-topic"
    is_finished: bool
    feedback: str
    used_questions: List[str]
    active_question: str
    retry_count: int
    current_topic_depth: int

# --- NODE 1: ANALYZE INPUT ---
def analyze_input(state: InterviewState) -> InterviewState:
    # CRITICAL GUARD: If history is empty, skip analysis. 
    # This happens on the very first turn when the bot starts the interview.
    if not state["messages"]:
        return {**state, "persona_detected": "Normal", "latest_evaluation": "Good"}

    user_input = state["messages"][-1]["content"]
    
    # 1. IMMEDIATE EXIT CHECK
    if any(keyword in user_input.lower() for keyword in ["end interview", "stop interview", "quit", "finish"]):
        return {**state, "persona_detected": "End_Session", "is_finished": True}

    # 2. HANDLE UNKNOWN ROLE RECOVERY
    if state["role"] == "Unknown":
        role_check_prompt = f"User input: '{user_input}'. Extract the job role. Return ONLY the role name. If none, return 'Unknown'."
        detected_role = llm.invoke([HumanMessage(content=role_check_prompt)]).content.strip().replace('"', '')
        if detected_role != "Unknown" and len(detected_role) < 50:
             return {**state, "role": detected_role, "persona_detected": "Normal", "question_count": 0}

    # 3. NORMAL ANALYSIS
    current_context = state.get("active_question", "Introduction")
    prompt = f"""
    You are an Interview Analyst.
    Role: {state['role']}
    Question asked: "{current_context}"
    User Answer: "{user_input}"
    
    Classify the User:
    1. PERSONA:
       - "Confused": Asks for help, doesn't understand.
       - "Distracted": Talks about pizza, noise, life, irrelevant things.
       - "Efficient": Short, blunt answers.
       - "Normal": Attempts to answer.
    
    2. EVALUATION:
       - "Good": Relevant answer.
       - "Vague": Too short, needs more detail.
       - "Off-topic": Completely unrelated to the question.
    
    Return JSON: {{ "persona": "...", "evaluation": "..." }}
    """
    
    try:
        resp = llm.invoke([HumanMessage(content=prompt)])
        content = resp.content.strip().replace("```json", "").replace("```", "")
        match = re.search(r'\{.*\}', content, re.DOTALL)
        data = json.loads(match.group(0)) if match else {"persona": "Normal", "evaluation": "Good"}
    except:
        data = {"persona": "Normal", "evaluation": "Good"}

    return {
        **state, 
        "persona_detected": data.get("persona", "Normal"), 
        "latest_evaluation": data.get("evaluation", "Good")
    }

# --- NODE 2: ASK DYNAMIC QUESTION ---
def ask_new_question(state: InterviewState) -> InterviewState:
    # Handling "Unknown" at the start
    if state["role"] == "Unknown":
        return {
            **state,
            "messages": state["messages"] + [{"role": "assistant", "content": "Hi! I can help you prepare for your interview. What job role are you targeting today?"}],
        }

    q_type = "behavioral" if state["question_count"] % 2 == 0 else "technical"
    avoid_topics = ", ".join(state.get("used_questions", [])[-3:]) 
    
    prompt = f"""
    Generate a single, challenging {q_type} interview question for a {state['role']}.
    Topics to avoid: {avoid_topics}.
    Return ONLY the question text.
    """
    new_question = llm.invoke([HumanMessage(content=prompt)]).content.strip()
    
    # Contextual Transition
    prefix = ""
    if state["question_count"] == 0:
        prefix = f"Great. Let's start the {state['role']} interview. "
    elif state["latest_evaluation"] == "Good":
        prefix = "Good. Moving on. "
    else:
        prefix = "Okay, let's switch topics. "

    return {
        **state, 
        "messages": state["messages"] + [{"role": "assistant", "content": f"{prefix}{new_question}"}], 
        "question_count": state["question_count"] + 1,
        "used_questions": state.get("used_questions", []) + [new_question],
        "active_question": new_question,
        "retry_count": 0,
        "current_topic_depth": 1
    }

# --- NODE 3: DEEP DIVE (FOLLOW-UP) ---
def ask_follow_up(state: InterviewState) -> InterviewState:
    last_user_msg = state["messages"][-1]["content"]
    current_q = state.get("active_question")
    
    prompt = f"""
    Role: {state['role']}
    Question: "{current_q}"
    User Answer: "{last_user_msg}"
    
    Generate a follow-up question.
    - If they missed details, ask for a specific example.
    - If they were good, challenge a trade-off or implementation detail.
    - Keep it short and sharp.
    """
    
    follow_up = llm.invoke([HumanMessage(content=prompt)]).content.strip()
    
    return {
        **state,
        "messages": state["messages"] + [{"role": "assistant", "content": follow_up}],
        "current_topic_depth": state.get("current_topic_depth", 0) + 1,
        "retry_count": 0
    }

# --- NODE 4: HANDLE SPECIAL CASES ---
def handle_special(state: InterviewState) -> InterviewState:
    target = state.get("active_question", "the interview")
    persona = state["persona_detected"]
    
    # STRICT SCENARIO MAPPING
    scenario = ""
    if persona == "Confused":
        scenario = f"The user is confused about: '{target}'. Explain it simply."
    elif persona == "Distracted":
        scenario = f"The user is talking about unrelated things (pizza, noise, etc). Politely ignore the distraction and repeat the question: '{target}'."
    elif state["latest_evaluation"] == "Vague":
        scenario = f"The user's answer was too vague. Ask for a specific example regarding: '{target}'."
    else:
        scenario = f"The user is off-topic. Steer them back to: '{target}'."
        
    prompt = f"""
    You are a professional Interviewer.
    Scenario: {scenario}
    
    Generate a response to the user. Do NOT say "User is...". Speak directly to the candidate.
    """
    
    reply = llm.invoke([HumanMessage(content=prompt)]).content.strip()
    
    return {
        **state,
        "messages": state["messages"] + [{"role": "assistant", "content": reply}],
        "retry_count": state.get("retry_count", 0) + 1
    }

# --- NODE 5: FEEDBACK GENERATION ---
def generate_feedback(state: InterviewState) -> InterviewState:
    transcript = "\n".join([f"{m['role']}: {m['content']}" for m in state['messages'] if m['role'] != 'system'])
    
    prompt = f"""
    Act as a Hiring Manager for a {state['role']} position.
    Review this transcript:
    {transcript}
    
    Generate a Markdown feedback report:
    # Interview Feedback: {state['role']}
    ## 1. Executive Summary
    ## 2. Strengths
    ## 3. Areas for Improvement
    ## 4. Hiring Recommendation
    """
    
    report = llm.invoke([HumanMessage(content=prompt)]).content
    
    return {
        **state,
        "messages": state["messages"] + [{"role": "assistant", "content": "Thank you! I've prepared your feedback report."}],
        "feedback": report,
        "is_finished": True
    }

# --- ROUTING LOGIC (THE BRAIN) ---
def decide_next(state: InterviewState):
    if state.get("is_finished"): return END
    
    # 1. INITIALIZATION CHECK (This prevents the 'Dense Question' bug)
    if state["question_count"] == 0:
        return "ask_new_question"
        
    # 2. EXIT & LIMIT CHECKS
    if state["persona_detected"] == "End_Session": return "generate_feedback"
    if state["question_count"] >= 5: return "generate_feedback"
    if state.get("retry_count", 0) >= 2: return "ask_new_question"

    # 3. SPECIAL HANDLING (Distraction/Confusion)
    if state["persona_detected"] in ["Confused", "Distracted", "Off-topic"]:
        return "handle_special"

    # 4. NORMAL FLOW
    if state["latest_evaluation"] == "Good":
        if state["persona_detected"] == "Efficient": return "ask_new_question"
        
        # Depth Check
        if state.get("current_topic_depth", 1) < MAX_DEPTH:
            return "ask_follow_up"
        
        return "ask_new_question"

    # Default fallback for Vague/Bad answers
    return "handle_special"

# --- COMPILE ---
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
    END: END
})

workflow.add_edge("ask_new_question", END)
workflow.add_edge("ask_follow_up", END)
workflow.add_edge("handle_special", END)
workflow.add_edge("generate_feedback", END)

app_graph = workflow.compile()