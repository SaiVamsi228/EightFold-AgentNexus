import json
import random
import re
from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
import os
from dotenv import load_dotenv

load_dotenv()

# Using Gemini 2.0 Flash
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash", 
    temperature=0.2  # Lower temperature = stricter JSON adherence
)

class InterviewState(TypedDict):
    messages: List[Dict[str, str]]
    role: str
    question_count: int
    persona_detected: str
    latest_evaluation: str
    is_finished: bool
    feedback: Dict[str, Any]

# --- HARDCODED QUESTIONS (To prevent file reading errors) ---
QUESTIONS_DB = {
  "Software Engineer": {
    "technical": [
      "Explain the difference between a process and a thread.",
      "How would you design a scalable URL shortener?",
      "What is the difference between TCP and UDP?",
      "Explain the concept of Dependency Injection.",
      "How do you debug a memory leak in production?"
    ],
    "behavioral": [
      "Tell me about a time you had to debug a critical production issue.",
      "Describe a situation where you disagreed with a senior engineer.",
      "How do you prioritize tasks when you have multiple deadlines?"
    ]
  },
  "SDR": {
    "technical": [
      "How do you research a prospect before a cold call?",
      "What is your strategy for getting past a gatekeeper?"
    ],
    "behavioral": [
      "Tell me about a time you faced repeated rejection.",
      "Describe your most difficult sale."
    ]
  },
  "Retail Associate": {
    "technical": [
      "How do you handle a customer trying to return an item without a receipt?"
    ],
    "behavioral": [
      "Tell me about a time you dealt with a very difficult customer."
    ]
  }
}

def load_questions(role: str):
    return QUESTIONS_DB.get(role, QUESTIONS_DB["Software Engineer"])

# --- NODE 1: ANALYZE ---
def analyze_input(state: InterviewState) -> InterviewState:
    user_input = state["messages"][-1]["content"]
    
    # PROMPT TUNING: Explicitly catch "Nervous" and "Don't Know"
    prompt = f"""
    Analyze this candidate's response: "{user_input}"
    
    RULES:
    1. If user says "I'm nervous", "I don't know", "I'm stuck", or "Help" -> Persona is "Confused".
    2. If user talks about a totally different topic (e.g. food, sports) -> Persona is "Chatty".
    3. If user answers the question (even poorly) -> Persona is "Normal".
    
    EVALUATION:
    - "Vague": Very short answers (1-3 words) EXCEPT if they are stating their Role.
    - "Good": Complete sentences.
    - "Off-topic": Irrelevant.

    Return JSON: {{ "persona": "...", "evaluation": "..." }}
    """
    
    try:
        resp = llm.invoke([HumanMessage(content=prompt)])
        content = resp.content.strip()
        
        # Clean JSON
        if "```" in content:
            content = content.replace("```json", "").replace("```", "")
        match = re.search(r'\{.*\}', content, re.DOTALL)
        if match:
            content = match.group(0)
            
        data = json.loads(content)
    except:
        data = {"persona": "Normal", "evaluation": "Good"}

    # FORCE OVERRIDE: If the user mentions a role, it's NOT vague, it's Good.
    if any(x in user_input.lower() for x in ["software", "engineer", "sales", "sdr", "retail"]):
        data["evaluation"] = "Good"

    print(f"DEBUG: Input='{user_input}' -> Detected={data}")

    return {
        **state,
        "persona_detected": data.get("persona", "Normal"),
        "latest_evaluation": data.get("evaluation", "Good")
    }

# --- NODE 2: ASK NEW QUESTION ---
def ask_new_question(state: InterviewState) -> InterviewState:
    questions = load_questions(state["role"])
    q_type = "behavioral" if state["question_count"] % 2 == 0 else "technical"
    
    base_question = random.choice(questions[q_type])
    
    # Transition phrasing
    if state["question_count"] > 0:
        base_question = f"Okay. Next question: {base_question}"
    else:
        base_question = f"Great, let's start the {state['role']} interview. {base_question}"
    
    new_messages = state["messages"] + [{"role": "assistant", "content": base_question}]
    
    return {
        **state, 
        "messages": new_messages, 
        "question_count": state["question_count"] + 1
    }

# --- NODE 3: HANDLE SPECIAL ---
def handle_special(state: InterviewState) -> InterviewState:
    last_assistant_msg = next((m["content"] for m in reversed(state["messages"]) if m["role"] == "assistant"), "the interview")
    user_input = state["messages"][-1]["content"]
    persona = state["persona_detected"]
    
    system_instruction = """
    You are a kind Interviewer. Speak directly to the candidate.
    Output ONLY the spoken response. No preamble.
    """

    # CRITICAL FIX: Specific prompts for Confused vs Vague
    if persona == "Confused":
        task_prompt = f"""
        User said: "{user_input}"
        They are nervous or don't know the answer to: "{last_assistant_msg}"
        
        Your Goal:
        1. Say: "That is completely fine, no need to worry."
        2. Ask a MUCH simpler, basic question related to {state['role']}.
        """
    elif state["latest_evaluation"] == "Vague":
        task_prompt = f"""
        User said: "{user_input}"
        Their answer to "{last_assistant_msg}" was too short.
        
        Your Goal:
        1. Ask them to provide more specific details or an example.
        """
    else:
        task_prompt = f"Politely steer the user back to the topic of {state['role']}."

    reply = llm.invoke([
        SystemMessage(content=system_instruction),
        HumanMessage(content=task_prompt)
    ]).content.strip().replace('"', '')

    return {**state, "messages": state["messages"] + [{"role": "assistant", "content": reply}]}

# --- NODE 4: FEEDBACK ---
def generate_feedback(state: InterviewState) -> InterviewState:
    transcript = "\n".join([f"{m['role']}: {m['content']}" for m in state["messages"]])
    prompt = f"""
    You are an Interviewer. The interview is over.
    Transcript: {transcript}
    
    Give the candidate a score out of 10 and 1 piece of advice.
    Speak directly to them.
    """
    closing = llm.invoke([HumanMessage(content=prompt)]).content
    return {
        **state,
        "messages": state["messages"] + [{"role": "assistant", "content": closing}],
        "is_finished": True
    }

# --- DECISION LOGIC ---
def decide_next(state: InterviewState):
    # 1. If it's the START of the interview (count=0), ALWAYS ask a question.
    # This prevents the "Vague" trap on the role selection.
    if state["question_count"] == 0:
        return "ask_new_question"

    # 2. End if too many questions
    if state["question_count"] >= 6:
        return "generate_feedback"
    
    # 3. Handle Personas
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