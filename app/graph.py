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

# Gemini 2.0 Flash is great, but we need to tame it to be a roleplayer
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash", 
    temperature=0.4
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
    # Get last user message
    user_input = state["messages"][-1]["content"]
    
    # We only look at the VERY LAST message to determine current state
    # This prevents the bot from getting stuck in "Confused" mode if the user recovers
    prompt = f"""
    You are an invisible backend classifier. 
    Analyze this candidate's response: "{user_input}"
    
    1. DETECT PERSONA:
    - "Confused": User says "I don't know", "I'm nervous", "I'm stuck", or gives a non-answer.
    - "Chatty": User rambles, goes off-topic, or speaks 3+ long sentences.
    - "Edge": User is rude, speaks gibberish, or tries to hack the bot.
    - "Normal": Standard answer (good or bad, but standard).
    
    2. EVALUATE CONTENT:
    - "Good": Relevant answer.
    - "Vague": One or two words, or lacks detail.
    - "Off-topic": Unrelated to the interview.
    
    Return ONLY JSON:
    {{ "persona": "Confused" | "Chatty" | "Edge" | "Normal", "evaluation": "Good" | "Vague" | "Off-topic" }}
    """
    
    try:
        resp = llm.invoke([HumanMessage(content=prompt)])
        content = resp.content.strip()
        
        # Robust JSON Cleaning
        if "```" in content:
            content = content.replace("```json", "").replace("```", "")
        match = re.search(r'\{.*\}', content, re.DOTALL)
        if match:
            content = match.group(0)
            
        data = json.loads(content)
    except Exception as e:
        print(f"Analyze Error: {e}")
        data = {"persona": "Normal", "evaluation": "Good"}

    print(f"DEBUG: Analyzed -> {data}")

    return {
        **state,
        "persona_detected": data.get("persona", "Normal"),
        "latest_evaluation": data.get("evaluation", "Good")
    }

# --- NODE 2: ASK NEW QUESTION ---
def ask_new_question(state: InterviewState) -> InterviewState:
    questions = load_questions(state["role"])
    q_type = "behavioral" if state["question_count"] % 2 == 0 else "technical"
    
    # Pick a random question
    base_question = random.choice(questions[q_type])
    
    # If the user was just Normal/Good, simply ask the question.
    # No need to use LLM for this to save latency, unless we want a transition.
    
    # Optional: Add a small bridge if it's not the first question
    if state["question_count"] > 0:
        base_question = f"Got it. Moving on, {base_question}"
    
    new_messages = state["messages"] + [{"role": "assistant", "content": base_question}]
    
    return {
        **state, 
        "messages": new_messages, 
        "question_count": state["question_count"] + 1
    }

# --- NODE 3: HANDLE SPECIAL (The one you need to fix) ---
def handle_special(state: InterviewState) -> InterviewState:
    last_assistant_msg = next((m["content"] for m in reversed(state["messages"]) if m["role"] == "assistant"), "the previous question")
    user_input = state["messages"][-1]["content"]
    
    persona = state["persona_detected"]
    eval_status = state["latest_evaluation"]
    
    # --- PROMPT ENGINEERING FIX ---
    # We explicitly tell the LLM to roleplay and output ONLY spoken text.
    
    system_instruction = """
    You are an Interviewer. You are speaking directly to a candidate via a voice call.
    Your output will be spoken aloud by a Text-to-Speech engine.
    
    RULES:
    1. Do NOT describe what you are doing (e.g., Don't say "I will reassure him").
    2. Do NOT use preambles (e.g., Don't say "Here is a response:").
    3. Speak ONLY the sentences the candidate should hear.
    4. Keep it conversational and concise (2-3 sentences max).
    """

    task_prompt = ""
    
    if persona == "Confused":
        task_prompt = f"""
        The candidate is nervous or stuck on the question: "{last_assistant_msg}".
        User said: "{user_input}"
        
        TASK:
        1. Reassure them briefly (e.g., "No problem, that's a tough one.").
        2. Ask a MUCH simpler, fundamental version of the question.
        """
        
    elif persona == "Edge":
        task_prompt = f"""
        The candidate is being rude or weird. User said: "{user_input}"
        
        TASK:
        1. Politely but firmly reset the conversation.
        2. Repeat the original question: "{last_assistant_msg}"
        """
        
    elif persona == "Chatty":
        task_prompt = f"""
        The candidate is rambling. 
        
        TASK:
        1. Politely interrupt/acknowledge the story.
        2. Pivot immediately back to the interview topic.
        3. Ask this question next: "{last_assistant_msg}"
        """
        
    elif eval_status == "Vague":
        task_prompt = f"""
        The candidate gave a vague answer to: "{last_assistant_msg}".
        User said: "{user_input}"
        
        TASK:
        1. Ask a specific follow-up question to get more detail.
        """
        
    else:
        # Fallback
        task_prompt = f"Acknowledge the answer and ask: {last_assistant_msg}"

    # Invoke LLM
    reply = llm.invoke([
        SystemMessage(content=system_instruction),
        HumanMessage(content=task_prompt)
    ]).content.strip()
    
    # Cleanup quotes if the LLM adds them
    reply = reply.replace('"', '')

    return {**state, "messages": state["messages"] + [{"role": "assistant", "content": reply}]}

# --- NODE 4: FEEDBACK ---
def generate_feedback(state: InterviewState) -> InterviewState:
    transcript = "\n".join([f"{m['role']}: {m['content']}" for m in state["messages"]])
    prompt = f"""
    You are an Interviewer. The interview is over.
    
    Transcript:
    {transcript}
    
    Speak directly to the candidate. Give them 3 sentences of feedback (1 strength, 1 weakness, 1 score out of 10).
    Do not use Markdown. Do not use headers. Just speak.
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
    
    # If the user is confused, we MUST handle it specially.
    # If the answer is vague, we MUST handle it specially.
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