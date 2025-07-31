# streamlit_app.py
import streamlit as st
import requests
import json
import os
from dotenv import load_dotenv

load_dotenv()

# Load API URL and API Key from environment
API_URL = os.getenv("API_URL", "http://localhost:8000/hackrx/run")
API_KEY = os.getenv("HACKRX_API_KEY")

st.set_page_config(page_title="HackRx RAG Assistant", layout="wide")
st.title("📄 Intelligent Policy QA System")

st.markdown("Upload a policy PDF URL and ask questions. Get answers with reasoning and matched clauses.")

# Input URL and questions
doc_url = st.text_input("📎 Enter PDF Blob URL")
questions_input = st.text_area("💬 Ask your questions (one per line)", height=150)

if st.button("🚀 Run Query"):
    if not doc_url or not questions_input.strip():
        st.warning("Please enter both PDF URL and at least one question.")
    else:
        questions = [q.strip() for q in questions_input.strip().split("\n") if q.strip()]
        payload = {
            "documents": doc_url,
            "questions": questions
        }

        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        }

        with st.spinner("Contacting backend and processing..."):
            try:
                res = requests.post(API_URL, json=payload, headers=headers)
                res.raise_for_status()
                results = res.json().get("answers", [])

                for item in results:
                    with st.expander(f"🟢 {item['question']}"):
                        st.markdown(f"**Answer:** {item['answer']}")
                        st.markdown(f"**Reasoning:** {item['reasoning']}")
                        st.markdown(f"**Matched Clauses:** {', '.join(item['clauses'])}")
            except Exception as e:
                st.error(f"❌ Error: {e}")
