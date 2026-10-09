import streamlit as st
from PIL import Image
from crewai import Agent, Task, Crew, LLM

from tools.classifier_tool import predict_mri
from tools.gradcam_tool import generate_gradcam
from tools.escalation_tool import check_and_escalate
from tools.knowledge_tool import query_knowledge_base

st.set_page_config(page_title="Brain MRI AI Agent", page_icon="🧠", layout="wide")

st.title("🧠 Brain MRI Classification & Decision Support Agent")
st.warning("⚠️ **Educational & Research Disclaimer**: This software is strictly an educational research tool and must NOT be used for clinical diagnoses or direct patient care.")

if "escalated_cases" not in st.session_state:
    st.session_state.escalated_cases = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "latest_results" not in st.session_state:
    st.session_state.latest_results = None

st.sidebar.header("⚙️ Agent Controls")
confidence_threshold = st.sidebar.slider("Escalation Threshold", 0.50, 0.95, 0.70, 0.05)

tab1, tab2, tab3 = st.tabs(["📷 Scan Analysis", "💬 Interactive Chat", "📋 Escalation Queue"])

with tab1:
    uploaded_file = st.file_uploader("Upload Brain MRI Scan", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Uploaded Scan", use_container_width=True)
            
        if st.button("🚀 Run Analysis", type="primary"):
            with st.spinner("Analyzing scan..."):
                pred_results = predict_mri(image)
                gradcam_img = generate_gradcam(image)
                escalation_info = check_and_escalate(pred_results, threshold=confidence_threshold)
                retrieved_context = query_knowledge_base(pred_results['top_prediction'])
                
                st.session_state.latest_results = {
                    "pred_results": pred_results,
                    "gradcam_img": gradcam_img,
                    "escalation_info": escalation_info,
                    "retrieved_context": retrieved_context
                }

        if st.session_state.latest_results is not None:
            res = st.session_state.latest_results
            with col2:
                st.image(res["gradcam_img"], caption="Grad-CAM Feature Heatmap", use_container_width=True)
                
            st.markdown("---")
            st.subheader("📊 Network Outputs")
            st.write(f"**Predicted Class:** `{res['pred_results']['top_prediction']}` | **Confidence:** `{res['pred_results']['confidence']*100:.2f}%`")
            st.json(res["pred_results"]["all_probabilities"])
            
            if res["escalation_info"]["flagged"]:
                st.error(f"🚨 **Flagged**: {res['escalation_info']['reason']}")
            else:
                st.success("✅ Confidence threshold satisfied.")
                
            st.markdown("---")
            st.subheader("🤖 Synthesized Agent Report")
            
            groq_key = st.secrets.get("GROQ_API_KEY", "")
            if not groq_key:
                st.error("Missing GROQ_API_KEY in Streamlit secrets.")
            else:
                groq_llm = LLM(model="groq/openai/gpt-oss-120b", api_key=groq_key)
                
                report_agent = Agent(
                    role="Medical Communicator",
                    goal="Translate PyTorch outputs into clear, educational non-diagnostic descriptions.",
                    backstory="You summarize neural network findings using reference knowledge. You NEVER diagnose directly.",
                    llm=groq_llm
                )
                
                report_task = Task(
                    description=f"Synthesize output: {res['pred_results']['top_prediction']} ({res['pred_results']['confidence']*100:.2f}%). Context: {res['retrieved_context']}",
                    expected_output="Clean markdown summary.",
                    agent=report_agent
                )
                
                crew = Crew(agents=[report_agent], tasks=[report_task])
                with st.spinner("Generating agent explanation..."):
                    agent_report = crew.kickoff()
                    st.markdown(agent_report.raw)

with tab2:
    st.header("💬 Follow-up Chat")
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
    if user_query := st.chat_input("Ask a question..."):
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)
            
        groq_key = st.secrets.get("GROQ_API_KEY", "")
        if groq_key and st.session_state.latest_results is not None:
            res = st.session_state.latest_results
            chat_llm = LLM(model="groq/openai/gpt-oss-120b", api_key=groq_key)
            
            chat_agent = Agent(
                role="Knowledge Assistant",
                goal="Answer questions based on MRI findings.",
                backstory="Helpful AI tutor.",
                llm=chat_llm
            )
            
            chat_task = Task(
                description=f"Context: {res['pred_results']['top_prediction']}. Question: {user_query}",
                expected_output="Concise educational answer.",
                agent=chat_agent
            )
            
            chat_crew = Crew(agents=[chat_agent], tasks=[chat_task])
            with st.chat_message("assistant"):
                reply = chat_crew.kickoff().raw
                st.markdown(reply)
                st.session_state.chat_history.append({"role": "assistant", "content": reply})

with tab3:
    st.header("📋 Pending Escalations")
    if not st.session_state.escalated_cases:
        st.info("No cases pending review.")
    else:
        for idx, case in enumerate(st.session_state.escalated_cases, 1):
            with st.expander(f"Case #{idx} - {case['prediction']} ({case['confidence_percentage']})"):
                st.write(f"**Reason:** {case['reason']}")