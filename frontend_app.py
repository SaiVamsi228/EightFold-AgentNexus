import streamlit as st
import streamlit.components.v1 as components
import requests
import json

# --- CONFIGURATION ---
BACKEND_URL = "https://agent-nexus-be.onrender.com" 
# I extracted this ID from your Share URL: https://vapi.ai/share/073fcbe8-ce22-43ac-be1a-1f2c2ff77751
ASSISTANT_ID = "073fcbe8-ce22-43ac-be1a-1f2c2ff77751" 
# ⚠️ REPLACE THIS WITH YOUR VAPI PUBLIC KEY (Dashboard -> API Keys)
VAPI_PUBLIC_KEY = "0ab33e07-7afe-46f7-85dd-e921c7fa28eb"

# --- PAGE SETUP ---
st.set_page_config(page_title="Eightfold Interview Coach", page_icon="🎙️", layout="wide")

# --- CUSTOM CSS ---
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    .metric-card {
        background-color: #1E1E1E;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #333;
        text-align: center;
        margin-bottom: 10px;
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
</style>
""", unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=100)
    st.markdown("### ⚙️ Interview Settings")
    role = st.selectbox("Target Role", ["Software Engineer", "Sales (SDR)", "Retail Associate"])
    experience = st.slider("Years of Experience", 0, 10, 2)
    st.divider()
    st.info("💡 **Tip:** Allow microphone access when prompted.")

# --- MAIN HERO SECTION ---
col1, col2 = st.columns([2, 1])

with col1:
    st.title("Interview Practice Partner")
    st.markdown(f"Ready to practice for **{role}**?")
    st.markdown("---")
    
    st.subheader("1️⃣ Start Interview")
    st.markdown("Click below to start the voice session directly in this window.")

    # --- CUSTOM VAPI WEB SDK COMPONENT ---
    # This injects the Vapi JS SDK and creates a custom button UI
    vapi_component = f"""
    <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; padding: 20px; background: #1E1E1E; border-radius: 10px; border: 1px solid #333;">
        <button id="vapi-start-btn" style="
            background: linear-gradient(135deg, #00D26A 0%, #00A855 100%);
            border: none;
            border-radius: 8px;
            padding: 15px 40px;
            color: white;
            font-size: 18px;
            font-weight: bold;
            cursor: pointer;
            box-shadow: 0 4px 14px rgba(0, 210, 106, 0.3);
            transition: all 0.2s ease;
        ">📞 Start Voice Session</button>
        <p id="vapi-status" style="margin-top: 15px; color: #888; font-family: sans-serif;">Ready to connect</p>
        <p id="call-id-display" style="margin-top: 5px; color: #555; font-family: monospace; font-size: 12px;"></p>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/@vapi-ai/web@latest/dist/vapi.min.js"></script>
    
    <script>
        const vapi = new Vapi("{VAPI_PUBLIC_KEY}");
        const assistantId = "{ASSISTANT_ID}";
        const btn = document.getElementById("vapi-start-btn");
        const status = document.getElementById("vapi-status");
        const callIdDisplay = document.getElementById("call-id-display");

        let isCallActive = false;

        // BUTTON HANDLER
        btn.addEventListener("click", async () => {{
            if (!isCallActive) {{
                status.innerText = "Connecting...";
                try {{
                    await vapi.start(assistantId);
                }} catch (e) {{
                    status.innerText = "Error: " + e.message;
                }}
            }} else {{
                vapi.stop();
            }}
        }});

        // VAPI EVENTS
        vapi.on("call-start", () => {{
            isCallActive = true;
            btn.innerText = "🟥 End Interview";
            btn.style.background = "linear-gradient(135deg, #FF4B4B 0%, #D32F2F 100%)";
            btn.style.boxShadow = "0 4px 14px rgba(255, 75, 75, 0.3)";
            status.innerText = "🟢 Live | Listening...";
            status.style.color = "#00D26A";
        }});

        vapi.on("call-end", () => {{
            isCallActive = false;
            btn.innerText = "📞 Start Voice Session";
            btn.style.background = "linear-gradient(135deg, #00D26A 0%, #00A855 100%)";
            btn.style.boxShadow = "0 4px 14px rgba(0, 210, 106, 0.3)";
            status.innerText = "Session Ended";
            status.style.color = "#888";
        }});

        vapi.on("speech-start", () => {{
            status.innerText = "🤖 AI is speaking...";
        }});

        vapi.on("speech-end", () => {{
            status.innerText = "🟢 Live | Listening...";
        }});

        vapi.on("message", (message) => {{
            // Capture Call ID when available
            if (message.type === "call-start" || (message.call && message.call.id)) {{
                callIdDisplay.innerText = "Call ID: " + (message.call ? message.call.id : "Active");
            }}
        }});
    </script>
    """
    # Render the HTML component
    components.html(vapi_component, height=250)

    st.markdown("---")

    # --- FEEDBACK SECTION ---
    st.subheader("2️⃣ Post-Interview Feedback")
    
    # Input for Call ID (since Vapi generates unique IDs for every call)
    target_call_id = st.text_input("Enter Call ID (from above)", value="demo_session_final")

    if st.button("Generate Interview Report"):
        with st.spinner("Analyzing interview data..."):
            try:
                # Fetch feedback from your backend
                res = requests.get(f"{BACKEND_URL}/get-latest-feedback?call_id={target_call_id}")
                data = res.json()

                if data.get("status") == "Completed":
                    feedback = data["feedback"]
                    transcript = data.get("transcript", [])

                    # Metrics
                    m1, m2, m3 = st.columns(3)
                    with m1:
                        st.markdown(f"""<div class="metric-card"><div class="metric-value">{feedback.get('score', 'N/A')}</div><div class="metric-label">Score</div></div>""", unsafe_allow_html=True)
                    with m2:
                        st.markdown(f"""<div class="metric-card"><div class="metric-value">{len(transcript)}</div><div class="metric-label">Turns</div></div>""", unsafe_allow_html=True)
                    with m3:
                        st.markdown(f"""<div class="metric-card"><div class="metric-value">{feedback.get('confidence', 'High')}</div><div class="metric-label">Confidence</div></div>""", unsafe_allow_html=True)

                    # Strengths & Weaknesses
                    st.success(f"**Strengths:** {', '.join(feedback.get('strengths', ['Good flow']))}")
                    st.warning(f"**Improvements:** {', '.join(feedback.get('improvements', ['More detail needed']))}")
                    
                    # Transcript Expander
                    with st.expander("View Full Transcript"):
                        for msg in transcript:
                            prefix = "👤 You" if msg['role'] == 'user' else "🤖 AI"
                            st.markdown(f"**{prefix}:** {msg['content']}")
                else:
                    st.info("Interview currently in progress or no data found for this ID.")
                    
            except Exception as e:
                st.error(f"Connection Error: {e}")

with col2:
    st.markdown("### 📝 Notes & Live Status")
    st.info("The audio interface is running in the secure component on the left.")
    st.caption("Once the interview finishes, copy the Call ID displayed in the box to generate your personalized report.")