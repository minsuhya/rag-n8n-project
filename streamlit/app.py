import streamlit as st
import requests
import json

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.title("MkDocs RAG 챗봇 with n8n")
st.markdown("MkDocs 문서를 기반으로 질문에 답변하는 챗봇입니다. n8n 워크플로우를 통해 처리됩니다.")

N8N_WORKFLOW_URL = "http://n8n:5678/api/v1/workflows/run"

with st.form("chat_form"):
    user_input = st.text_input("질문을 입력하세요:", placeholder="MkDocs에 대해 궁금한 점을 물어보세요!")
    submit_button = st.form_submit_button("전송")

if submit_button and user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    try:
        response = requests.post(
            N8N_WORKFLOW_URL,
            json={"question": user_input},
            headers={"Content-Type": "application/json"}
        )
        if response.status_code == 200:
            result = response.json()
            answer = result.get("answer", "답변을 생성하지 못했습니다.")
            st.session_state.chat_history.append({"role": "assistant", "content": answer})
        else:
            st.error(f"n8n 워크플로우 호출 실패: {response.status_code}")
    except Exception as e:
        st.error(f"오류 발생: {str(e)}")

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.write(message["content"]) 