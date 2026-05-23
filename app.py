# Streamlit app for our RAG Assistant
import os
from typing import Any, Dict

import streamlit as st

from rag_project_retrieval import run_llm

st.set_page_config(page_title="RAG Assistant", page_icon=":robot_face:")
st.title("RAG Assistant with LangChain", text_alignment="center")

with st.sidebar:
    st.subheader("Session")
    if st.button("Clear Session", use_container_width=True):
        st.session_state.pop("messages", None)
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "ai",
            "content": "Hello! I'm your RAG assistant. Ask me anything, and I'll do my best to help you with the information I can retrieve.",
            "source": [],
        }
    ]


for mssg in st.session_state.messages:
    with st.chat_message(mssg["role"]):
        st.markdown(mssg["content"])
        if mssg["source"]:
            with st.expander("**Sources:**"):
                for source in mssg["source"]:
                    st.markdown(f"- {source}")


prompt = st.chat_input("Ask a question")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt, "source": []})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("ai"):
        try:
            with st.spinner("Generating response..."):
                result: Dict[str, Any] = run_llm(prompt)
                answer = str(
                    result.get("answer", "")
                    or "Sorry, I couldn't find an answer to your question based on the retrieved context. Please try asking something else or provide more details."
                )
                sources = result.get("sources", [])
                st.markdown(answer)
                if sources:
                    with st.expander("**Sources:**"):
                        for source in sources:
                            st.markdown(f"- {source}")

                st.session_state.messages.append(
                    {"role": "ai", "content": answer, "source": sources}
                )

        except Exception as e:
            st.markdown(f"An error occurred: {str(e)}")
