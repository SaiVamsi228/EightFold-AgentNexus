import json
import re
from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()

# Use a lower temperature for logic/classification, higher for creative questions
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.4)

# --- CONFIGURATION ---
MAX_DEPTH = 3  # Maximum follow-up questions per topic

# --- STATE DEFINITION ---
class InterviewState(TypedDict):
    messages: List[Dict[str, str]]
    role: str
    question_count: int
    persona_detected: str       # "Confused", "Efficient", "Chatty", "Normal", "End_Session"
    latest_evaluation: str      # "Good", "Vague", "Off-topic"
    is_finished: bool
    feedback: str
    used_questions: List[str]
    active_question: str
    retry_count: int
    current_topic_depth: int    # Tracks depth of current line of questioning

# --- NODE 1: ANALYZE INPUT ---
def analyze_input(state: InterviewState) -> InterviewState:
    user_input = state["messages"][-1]["content"]
    
    # 1. IMMEDIATE EXIT CHECK
    if any(keyword in user_input.lower() for keyword in ["end interview", "stop interview", "quit", "finish"]):
        return {**state, "persona_detected": "End_Session", "is_finished": True}

    # 2. HANDLE UNKNOWN ROLE (Initial Setup)
    if state["role"] == "Unknown":
        # Extract role from input
        role_check_prompt = f"User input: '{user_input}'. Extract the job role. If none found, return 'Unknown'. Return ONLY the role name."
        detected_role = llm.invoke([HumanMessage(content=role_check_prompt)]).content.strip().replace('"', '')
        
        if detected_role != "Unknown" and len(detected_role) < 50:
             # Force transition to asking the first question immediately
             return {
                 **state, 
                 "role": detected_role, 
                 "persona_detected": "Normal", 
                 "latest_evaluation": "Good",
                 "question_count": 0 # Ensure we start fresh
             }

    # 3. NORMAL ANALYSIS
    current_context = state.get("active_question", "Introduction")
    prompt = f"""
    You are an Interview Analyst.
    Current Role: {state['role']}
    Context (Last Question): "{current_context}"
    User Input: "{user_input}"
    
    Analyze the User Input:
    1. PERSONA: 
       - "Confused" (doesn't understand the question)
       - "Distracted" (talks about pizza, dogs, noise, unrelated life events)
       - "Efficient" (short, direct answers)
       - "Normal" (attempts to answer)
    
    2. EVALUATION: 
       - "Good" (relevant answer)
       - "Vague" (needs more detail)
       - "Off-topic" (unrelated to the question)
    
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
    # Handle the case where role is still unknown
    if state["role"] == "Unknown":
        return {
            **state,
            "messages": state["messages"] + [{"role": "assistant", "content": "Hi! I can help you prepare for any interview. What job role are you targeting today?"}],
        }

    q_type = "behavioral" if state["question_count"] % 2 == 0 else "technical"
    avoid_topics = ", ".join(state.get("used_questions", [])[-3:]) 
    
    prompt = f"""
    Generate a challenging {q_type} interview question for a {state['role']}.
    Topics to avoid: {avoid_topics}.
    Keep it professional and concise. Return ONLY the question.
    """
    new_question = llm.invoke([HumanMessage(content=prompt)]).content.strip()
    
    # Contextual Transition
    prefix = ""
    if state["question_count"] == 0:
        prefix = f"Great. Let's start the {state['role']} interview. "
    elif state["latest_evaluation"] == "Good":
        prefix = "Good answer. Let's move on. "
    else:
        prefix = "Okay, let's switch gears. "

    return {
        **state, 
        "messages": state["messages"] + [{"role": "assistant", "content": f"{prefix}{new_question}"}], 
        "question_count": state["question_count"] + 1,
        "used_questions": state.get("used_questions", []) + [new_question],
        "active_question": new_question,
        "retry_count": 0,
        "current_topic_depth": 1 # Reset depth (1 means we asked the main question)
    }

# --- NODE 3: DEEP DIVE (FOLLOW-UP) ---
def ask_follow_up(state: InterviewState) -> InterviewState:
    last_user_msg = state["messages"][-1]["content"]
    current_q = state.get("active_question")
    
    prompt = f"""
    The user is interviewing for {state['role']}.
    Main Question: "{current_q}"
    User Answer: "{last_user_msg}"
    
    Generate a specific follow-up question to dig deeper.
    If the answer was vague, ask for examples.
    If the answer was good, challenge a specific detail or trade-off.
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
        instruction = f"The user is confused about the question: '{target}'. Explain it simply and ask them to try again."
    elif persona == "Distracted":
        # Specific handling for 'Pizza' or noise
        instruction = f"The user is distracted or chatting about unrelated things (like food or noise). Politely acknowledge it briefly, but firmly bring them back to the question: '{target}'."
    elif state["latest_evaluation"] == "Vague":
        instruction = f"The user's answer was too vague. Ask them to provide specific details or an example regarding: '{target}'."
    else:
        instruction = f"The user went off-topic. Politely steer them back to answering: '{target}'."
        
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
    ## 2. Strengths
    ## 3. Areas for Improvement
    ## 4. Hiring Recommendation (Strong Hire / Hire / No Hire)
    """
    
    report = llm.invoke([HumanMessage(content=prompt)]).content
    
    return {
        **state,
        "messages": state["messages"] + [{"role": "assistant", "content": "Thank you! I've prepared your feedback report."}],
        "feedback": report,
        "is_finished": True
    }

# --- ROUTING LOGIC ---
def decide_next(state: InterviewState):
    if state["persona_detected"] == "End_Session":
        return "generate_feedback"

    if state["role"] == "Unknown":
        return "ask_new_question"

    if state["question_count"] >= 5: 
        return "generate_feedback"
    
    # If the user is confused/distracted, we must handle that before moving on
    if state["persona_detected"] in ["Confused", "Distracted", "Off-topic"]:
        # But don't get stuck in a loop forever
        if state.get("retry_count", 0) >= 2:
             return "ask_new_question"
        return "handle_special"

    # Evaluation Logic
    if state["latest_evaluation"] == "Good":
        if state["persona_detected"] == "Efficient":
            return "ask_new_question"
            
        # DEPTH CHECK: Only ask follow-up if we haven't hit the limit (MAX_DEPTH)
        if state.get("current_topic_depth", 1) < MAX_DEPTH:
            return "ask_follow_up"
        
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
    "generate_feedback": "generate_feedback"
})

workflow.add_edge("ask_new_question", END)
workflow.add_edge("ask_follow_up", END)
workflow.add_edge("handle_special", END)
workflow.add_edge("generate_feedback", END)

app_graph = workflow.compile()