import os
import sys
from pathlib import Path

import numpy as np
import streamlit as st

ROOT_DIR = next(parent for parent in Path(__file__).resolve().parents if (parent / "project_env.py").is_file())
sys.path.insert(0, str(ROOT_DIR))
from project_env import load_project_env

load_project_env()

from custom_chatbot import ModularRAG
from PIL import Image

# page title
st.set_page_config(page_title="🦜🕸️ 반도체 도메인 보고서 기반 챗봇")
st.title("🦜🕸️ 반도체 도메인 보고서 기반 챗봇")

documents_dir = "data/paper"

# 새로운 문서를 추가했다면, force_reload를 True로 변경하고, documents_dsamsung_chatbot_v2escription을 수정하세요.
documents_description = "차세대 반도체를 위한 글로벌 장비 개발 동향"

force_reload = False


@st.cache_resource
def init_chatbot():
    chatbot = ModularRAG(documents_dir, documents_description, force_reload)
    return chatbot


# Streamlit app은 app code를 계속 처음부터 재실행하는 방식으로 페이지를 갱신합니다.
# Chatbot을 state에 포함시키지 않으면 매 질문마다 chatbot을 다시 초기화 합니다.
if "chatbot" not in st.session_state:
    with st.spinner("챗봇 초기화 중입니다, 최대 3분까지 소요됩니다."):
        chatbot = init_chatbot()
        st.session_state.chatbot = chatbot
    st.write("챗봇 초기화를 완료했습니다.")

if "messages" not in st.session_state:
    st.session_state.messages = []

st.markdown(
    """
- 예시 질문 (문서 활용): 삼성전자 반도체의 장점을 설명해주세요.
- 예시 질문 (웹 검색 활용): 삼성전자 신규 제품을 소개해주세요.
    """
)

for conversation in st.session_state.messages:
    with st.chat_message(conversation["role"]):
        if "image" in conversation.keys() and conversation["image"]:
            st.image(conversation["content"])
        else:
            st.write(conversation["content"])

# React to user input
if prompt := st.chat_input("질문을 입력하면 챗봇이 답변을 제공합니다."):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

if prompt is not None:
    response = st.session_state.chatbot.run(prompt)
    generation = response
    with st.chat_message("assistant"):
        st.markdown(generation)
        st.session_state.messages.append(
            {"role": "assistant", "content": generation, "image": False}
        )
