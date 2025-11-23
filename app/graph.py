import json
import re
from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()

# Using a slightly higher temperature for creative question generation
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.4)

# --- STATE DEFINITION ---
class InterviewState(TypedDict):
    messages: List[Dict[str, str]]
    role: str
    question_count: int
    persona_detected: str       # "Confused", "Efficient", "Chatty", "Normal"
    latest_evaluation: str      # "Good", "Vague", "Off-topic"
    is_finished: bool
    feedback: str               # Changed to string to store Markdown report
    used_questions: List[str]   # Track generated questions to avoid repeats
    active_question: str
    retry_count: int
    current_topic_depth: int    # NEW: Tracks follow-ups on the same topic

# --- NODE 1: ANALYZE INPUT ---
def analyze_input(state: InterviewState) -> InterviewState:
    user_input = state["messages"][-1]["content"]
    
    # Handle the "Unknown" role case (User answering "What role?")
    if state["role"] == "Unknown":
        # Simple heuristic or LLM check to extract role
        role_check_prompt = f"User input: '{user_input}'. Extract the job role they want to interview for. If none found, return 'Unknown'. Return ONLY the role name."
        detected_role = llm.invoke([HumanMessage(content=role_check_prompt)]).content.strip().replace('"', '')
        
        if detected_role != "Unknown" and len(detected_role) < 50:
             return {**state, "role": detected_role, "persona_detected": "Normal", "latest_evaluation": "Good"}

    current_context = state.get("active_question", "Introduction")

    prompt = f"""
    You are an Interview Analyst.
    Current Role: {state['role']}
    Context: "{current_context}"
    User Input: "{user_input}"
    
    Analyze:
    1. PERSONA: "Confused" (asks for help), "Efficient" (short/curt), "Chatty" (long stories), "Normal".
    2. EVALUATION: "Good" (relevant), "Vague" (needs details), "Off-topic".
    
    Return JSON: {{ "persona": "...", "evaluation": "..." }}
    """
    
    try:
        resp = llm.invoke([HumanMessage(content=prompt)])
        content = resp.content.strip().replace("```json", "").replace("```", "")
        match = re.search(r'\{.*\}', content, re.DOTALL)
        data = json.loads(match.group(0)) if match else {"persona": "Normal", "evaluation": "Good"}
    except:
        data = {"persona": "Normal", "evaluation": "Good"}

    # Force "Good" if it's just the start
    if state["question_count"] == 0:
        data["evaluation"] = "Good"

    return {
        **state, 
        "persona_detected": data.get("persona", "Normal"), 
        "latest_evaluation": data.get("evaluation", "Good")
    }

# --- NODE 2: ASK DYNAMIC QUESTION ---
def ask_new_question(state: InterviewState) -> InterviewState:
    # 1. Handle Missing Role
    if state["role"] == "Unknown":
        return {
            **state,
            "messages": state["messages"] + [{"role": "assistant", "content": "Hello! I can help you prepare for any interview. What job role are you targeting today?"}],
            # Don't increment count yet
        }

    # 2. Generate Question
    q_type = "behavioral" if state["question_count"] % 2 == 0 else "technical"
    avoid_topics = ", ".join(state.get("used_questions", [])[-3:]) # Only track last 3 to save tokens
    
    prompt = f"""
    Generate a challenging {q_type} interview question for a {state['role']}.
    Topics to avoid (already asked): {avoid_topics}.
    Keep it professional and concise. Return ONLY the question.
    """
    
    new_question = llm.invoke([HumanMessage(content=prompt)]).content.strip()
    
    # 3. Transition Phrase
    prefix = ""
    if state["question_count"] > 0:
        if state["latest_evaluation"] == "Good":
            prefix = "Good answer. Let's move on. "
        else:
            prefix = "Okay, let's switch gears. "

    final_content = f"{prefix}{new_question}" if state["question_count"] > 0 else f"Great, let's start the {state['role']} interview. {new_question}"

    return {
        **state, 
        "messages": state["messages"] + [{"role": "assistant", "content": final_content}], 
        "question_count": state["question_count"] + 1,
        "used_questions": state.get("used_questions", []) + [new_question],
        "active_question": new_question,
        "retry_count": 0,
        "current_topic_depth": 0 # Reset depth for new topic
    }

# --- NODE 3: DEEP DIVE (FOLLOW-UP) ---
def ask_follow_up(state: InterviewState) -> InterviewState:
    last_user_msg = state["messages"][-1]["content"]
    current_q = state.get("active_question")
    
    prompt = f"""
    The user is interviewing for {state['role']}.
    Question: "{current_q}"
    User Answer: "{last_user_msg}"
    
    You are a skeptic interviewer. The user gave a decent answer, but you want to test their depth.
    Generate a specific follow-up question.
    - Ask for a concrete example, a trade-off, or "Why didn't you do X?".
    - Keep it short.
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
    
    instruction = ""
    if persona == "Confused":
        instruction = f"User is confused about '{target}'. Simplify the question or explain the concept, then ask them to try again."
    elif persona == "Distracted":
        instruction = f"User is distracted. Be empathetic, say you'll wait, then gently repeat: '{target}'."
    elif state["latest_evaluation"] == "Vague":
        instruction = f"User answer was too vague. Ask them to provide specific details or an example regarding: '{target}'."
    else:
        instruction = f"User is going off-topic. Politely steer them back to answering: '{target}'."
        
    reply = llm.invoke([SystemMessage(content="You are a professional Interviewer."), HumanMessage(content=instruction)]).content
    
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
    Review this interview transcript:
    {transcript}
    
    Generate a detailed feedback report in Markdown format:
    # Interview Feedback: {state['role']}
    
    ## 1. Executive Summary
    (2-3 sentences on overall impression)
    
    ## 2. Strengths
    * (Bullet point)
    * (Bullet point)
    
    ## 3. Areas for Improvement
    * (Specific technical gaps or communication issues)
    
    ## 4. Hiring Recommendation
    (Strong Hire / Hire / No Hire)
    """
    
    report = llm.invoke([HumanMessage(content=prompt)]).content
    
    return {
        **state,
        "messages": state["messages"] + [{"role": "assistant", "content": "Thank you for your time. I've generated your feedback report."}],
        "feedback": report,
        "is_finished": True
    }

# --- ROUTING LOGIC ---
def decide_next(state: InterviewState):
    if state["is_finished"]: return END
    if state["role"] == "Unknown" and state["question_count"] == 0: return "ask_new_question"
    
    if state["question_count"] >= 5: return "generate_feedback"
    
    # Retry Limit
    if state.get("retry_count", 0) >= 2: return "ask_new_question"

    # Evaluation Logic
    if state["latest_evaluation"] == "Good":
        # If Efficient persona, skip follow-ups
        if state["persona_detected"] == "Efficient":
            return "ask_new_question"
            
        # If we haven't dug deep yet, do a follow-up
        if state.get("current_topic_depth", 0) == 0:
            return "ask_follow_up"
        
        # If we already did a follow-up, move to new question
        return "ask_new_question"

    return "handle_special"

# --- GRAPH COMPILE ---
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