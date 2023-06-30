import streamlit as st
# from models.predict_model import *

with st.sidebar:
    st.write("# 🤖 Language Models")
    "[![GitHub Repo](https://github.com/codespaces/badge.svg)](https://github.com/Foxxy-HCMUS/e2eqa)"

st.title("💬 Question-Answering System")
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():

    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    msg = {
        "role": "assistant", 
        "content": "test", #get_answer_e2e(prompt)
    }
    st.session_state.messages.append(msg)
    st.chat_message("assistant").write(msg["content"])
    
    