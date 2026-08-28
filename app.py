import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parent
ENV_PATH = PROJECT_ROOT / '.env'
load_dotenv(dotenv_path=ENV_PATH, override=True)

st.set_page_config(page_title="OpenAI 챗봇", page_icon="💬")
st.title("💬 OpenAI 기본 챗봇")

api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_KEY")
if not api_key:
    st.error(f"API 키를 찾지 못했습니다: {ENV_PATH}")
    st.stop()

client = OpenAI(api_key=api_key)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("메시지를 입력하세요")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("응답 생성 중..."):
            response = client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                messages=st.session_state.messages,
            )
            answer = response.choices[0].message.content or "응답이 없습니다."
            st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
