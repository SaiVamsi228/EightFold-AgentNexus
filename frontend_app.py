import streamlit as st
import requests
import json
import time

# --- CONFIGURATION ---
# Ensure this URL has NO trailing slash
BACKEND_URL = "https://agent-nexus-be.onrender.com" 
# Your Vapi Public Share URL (Get this from Vapi Dashboard -> Assistant -> Share)
VAPI_SHARE_URL = "https://vapi.ai/share/073fcbe8-ce22-43ac-be1a-1f2c2ff77751"

# --- PAGE SETUP ---
st.set_page_config(page_title="Eightfold Interview Coach", page_icon="🎙️", layout="wide")

# --- CUSTOM CSS ---
st.markdown("""
<style>
    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Modern Card Style */
    .metric-card {
        background-color: #1E1E1E;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #333;
        text-align: center;
    }
    .metric-value {
        font-size: 2em;
        font-weight: bold;
        color: #00D26A;
    }
    .metric-label {
        color: #888;
        font-size: 0.9em;
    }
    
    /* Button Styling */
    .stButton button {
        width: 100%;
        border-radius: 8px;
        height: 50px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=100)
    st.markdown("### ⚙️ Interview Settings")
    
    role = st.selectbox("Target Role", ["Software Engineer", "Sales (SDR)", "Retail Associate"])
    experience = st.slider("Years of Experience", 0, 10, 2)
    resume = st.file_uploader("Upload Resume (Optional)", type=["pdf"])
    
    st.divider()
    st.info("💡 **Tip:** The interview runs in a secure voice room. Come back here for your results.")

# --- MAIN HERO SECTION ---
col1, col2 = st.columns([2, 1])

with col1:
    st.title("Interview Practice Partner")
    st.markdown(f"Ready to practice for **{role}**?")

    st.markdown("---")
    st.subheader("1️⃣ Start Interview")
    st.markdown("Click the button below to enter the secure interview room. Speak with the AI agent, then return here for your feedback.")

    # --- THE FIX: LINK BUTTON ---
    # This opens Vapi in a new tab, bypassing the CSP/Iframe block
    st.link_button("📞 Launch Secure Interview Room", VAPI_SHARE_URL, type="primary", use_container_width=True)

    st.markdown("---")

    # --- FEEDBACK SECTION ---
    st.subheader("2️⃣ Post-Interview Feedback")

    if st.button("Generate Interview Report"):
        with st.spinner("Fetching data from backend..."):
            try:
                # Using default session ID for demo simplicity
                res = requests.get(f"{BACKEND_URL}/get-latest-feedback?call_id=demo_session_final")
                data = res.json()

                if data.get("status") == "Completed":
                    feedback = data["feedback"]

                    # Score Cards
                    m1, m2, m3 = st.columns(3)
                    with m1:
                        st.markdown(f"""<div class="metric-card"><div class="metric-value">{feedback.get('score', 0)}/10</div><div class="metric-label">Overall Score</div></div>""", unsafe_allow_html=True)
                    with m2:
                        st.markdown(f"""<div class="metric-card"><div class="metric-value">{len(data.get('transcript', []))}</div><div class="metric-label">Turns Taken</div></div>""", unsafe_allow_html=True)
                    with m3:
                        st.markdown(f"""<div class="metric-card"><div class="metric-value">High</div><div class="metric-label">Confidence</div></div>""", unsafe_allow_html=True)

                    st.markdown("### 🟢 Strengths")
                    for s in feedback.get("strengths", ["Good communication"]):
                        st.success(s)

                    st.markdown("### 🔴 Areas for Improvement")
                    for i in feedback.get("improvements", ["Be more specific"]):
                        st.warning(i)

                    st.balloons()
                else:
                    st.info("Interview is either in progress or hasn't started yet.")
                    st.caption(f"Backend Status: {data.get('status')}")
                    
            except Exception as e:
                st.error(f"Connection Error: {e}")
                st.caption("Make sure the Backend Render service is active.")

with col2:
    st.markdown("### 📝 Live Transcript")
    st.caption("Conversation history loads here after generation.")
    
    transcript_container = st.container()
    with transcript_container:
        st.markdown("*Transcript pending...*")