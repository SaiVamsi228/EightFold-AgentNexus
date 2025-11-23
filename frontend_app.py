import streamlit as st
import requests
import json
import time

# --- CONFIGURATION ---
BACKEND_URL = "https://agent-nexus-be.onrender.com"  # <--- REPLACE WITH YOUR RAILWAY URL
VAPI_ASSISTANT_ID = "073fcbe8-ce22-43ac-be1a-1f2c2ff77751"      # <--- REPLACE WITH VAPI ASSISTANT ID
VAPI_PUBLIC_KEY = "0ab33e07-7afe-46f7-85dd-e921c7fa28eb"  # <--- REPLACE WITH VAPI PUBLIC KEY

# --- PAGE SETUP ---
st.set_page_config(page_title="Eightfold Interview Coach", page_icon="🎙️", layout="wide")

# --- CUSTOM CSS (Modern UI Polish) ---
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
        color: #00D26A; /* Eightfold Green-ish */
    }
    .metric-label {
        color: #888;
        font-size: 0.9em;
    }
    
    /* Title Style */
    h1 {
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 700;
        color: #FFFFFF;
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
    st.info("💡 **Tip:** Speak clearly. The agent will adapt to your answers.")

# --- MAIN HERO SECTION ---
col1, col2 = st.columns([2, 1])

with col1:
    st.title("Interview Practice Partner")
    st.markdown(f"Ready to practice for **{role}**? Click the microphone button below to start talking.")
    
    # --- REPLACED VAPI WIDGET SECTION ---
    st.markdown("---")
    st.subheader("🎤 Start Interview")

    # We use a height of 300px to ensure the Vapi popup isn't cut off
    # We allow "microphone" in the sandbox parameters
    vapi_code = f"""
    <!DOCTYPE html>
    <html>
    <body>
      <script>
        var vapiInstance = null;
        const assistantId = "{VAPI_ASSISTANT_ID}"; 
        const apiKey = "{VAPI_PUBLIC_KEY}"; 
        const buttonConfig = {{
          position: "center", 
          width: "80px", 
          height: "80px",
          idle: {{ icon: "https://unpkg.com/lucide-static@0.321.0/icons/mic.svg", color: "#fff" }},
          loading: {{ color: "#fff" }},
          active: {{ color: "#00D26A" }} 
        }};

        (function(d, t) {{
          var g = d.createElement(t), s = d.getElementsByTagName(t)[0];
          g.src = "https://cdn.jsdelivr.net/gh/VapiAI/html-script-tag@latest/dist/assets/index.js";
          g.defer = true;
          g.async = true;
          s.parentNode.insertBefore(g, s);

          g.onload = function() {{
            vapiInstance = window.vapiSDK.run({{
              apiKey: apiKey, 
              assistant: assistantId,
              config: buttonConfig
            }});
          }};
        }})(document, "script");
      </script>
    </body>
    </html>
    """
    
    # Renders with specific permission height
    st.components.v1.html(vapi_code, height=300, scrolling=False)
    
    st.markdown("---")
    
    # --- FEEDBACK SECTION ---
    st.subheader("📊 Post-Interview Feedback")
    
    # Button to fetch results manually
    if st.button("Generate Interview Report"):
        with st.spinner("Analyzing conversation data..."):
            try:
                # For demo, we use a default call_id
                res = requests.get(f"{BACKEND_URL}/get-latest-feedback?call_id=test_session_default")
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
                    st.info("Interview in progress or not started. Finish the call to see results!")
            except Exception as e:
                st.error(f"Could not connect to backend: {e}")

with col2:
    st.markdown("### 📝 Live Transcript")
    st.caption("Conversation history will appear here after the interview.")
    
    # Placeholder for transcript
    transcript_container = st.container(height=400)
    with transcript_container:
        st.markdown("*Waiting for interview to start...*")
