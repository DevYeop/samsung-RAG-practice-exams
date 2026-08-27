# =============================================================================
# 04. 최소 대화 메모리 챗봇
# =============================================================================
#
# [이 파일의 학습 목표]
# RAG 없이도 LLM이 같은 대화에서 이전 질문과 답변을 기억하게 만듭니다.
#
# 일반 체인은 매번 독립적으로 실행됩니다. 따라서 두 번째 질문만 전달하면 첫 번째 질문에서
# 알려 준 내용을 알 수 없습니다. `RunnableWithMessageHistory`는 체인을 실행하기 전에 이전
# 대화를 넣고, 실행이 끝난 뒤 현재 질문과 답변을 다시 대화 기록에 저장합니다.
#
# 핵심 흐름:
# 질문 -> session_id로 이전 대화 조회 -> prompt -> LLM -> 답변 -> 대화 기록에 저장
# =============================================================================

from langchain_openai import ChatOpenAI
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

import os
import sys
from pathlib import Path

ROOT_DIR = next(parent for parent in Path(__file__).resolve().parents if (parent / "project_env.py").is_file())
sys.path.insert(0, str(ROOT_DIR))
from project_env import load_project_env, require_env

load_project_env()
require_env("OPENAI_API_KEY")


llm = ChatOpenAI(model="gpt-4o-mini")

# 세션별 대화 기록을 저장하는 딕셔너리입니다.
# 같은 session_id는 같은 기록을 사용하고, 다른 session_id는 별도의 대화로 취급합니다.
store = {}

def get_session_history(session_id):
    """session_id에 해당하는 대화 기록을 반환합니다."""
    if session_id not in store: # session_id가 없으면 새로 생성
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id] # session_id에 해당하는 대화 기록 반환

# MessagesPlaceholder 위치에 해당 세션 이전의 HumanMessage와 AIMessage가 들어갑니다.
prompt = ChatPromptTemplate.from_messages([
    ("system", "사용자와 나눈 이전 대화를 참고하여 자연스럽게 답변하세요."), # 이건 유저프롬프트보다 더 상위 지침인 시스템 프롬프트로써 작동하게 됩니다
    MessagesPlaceholder(variable_name="chat_history"), # 여기에 이전 대화내역들이 들어갑니다.
    {"question": "제 이름은 민수입니다."},
    ("human", "{question}"), #유저프롬프트로 채ㅑ팅내역에 저장이 되게됩니다
])

chain = prompt | llm | StrOutputParser()

# 기존 체인을 메모리 기능으로 감쌉니다.
# ⭐이로서 chain에 저장기능이 추가되어, 같은 session_id를 가진 호출은 이전 대화를 기억하게 됩니다.
chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history, # session_id를 받아서 InMemoryChatMessageHistory를 반환하는 함수
    
    # 사용자가 입력한 질문이 들어있는 key (하단의 "제 이름은 민수입니다."라는 value에 대한 키를 말한다)
    #
    # RunnableWithMessageHistory는 이 값을 현재 사용자 메시지로 인식하고,
    # 체인 실행이 끝난 뒤 대화 기록에 HumanMessage로 저장합니다.
    input_messages_key="question", 
    
    # 상단의 MessagesPlaceholder에서 variable_name로 지정한 key
    # 그 결과, 해당 위치에 이전 대화내역이 들어가게 됩니다.
    history_messages_key="chat_history", 
)

# 같은 session_id를 사용해야 두 호출이 하나의 대화로 연결됩니다.
config = {"configurable": {"session_id": "user_A"}}

# chain에 저장기능을 입힌 chain_with_history를 사용하면, 이전 대화가 기억됩니다.
first_response = chain_with_history.invoke(
    {"question": "제 이름은 민수입니다."},
    config=config, # config에 session_id를 지정하면, 같은 session_id를 가진 호출은 같은 대화로 취급됩니다.
)
print("첫 번째 답변:", first_response)

second_response = chain_with_history.invoke(
    {"question": "제 이름이 무엇이라고 했나요?"},
    config=config,
)
print("두 번째 답변:", second_response)

print("채팅을 시작합니다.")
print("'종료' 또는 'exit'를 입력하면 프로그램이 종료됩니다.")

while True:
    # strip()로 공백 제거 후 질문을 입력받습니다.
    question = input("\n사용자: ").strip()

    # 종료 명령어를 입력하면 반복문을 끝냅니다.
    if question.lower() in ["종료", "exit", "quit"]:
        print("채팅을 종료합니다.")
        break

    # 빈 문자열을 입력한 경우에는 다시 입력받습니다.
    if not question:
        print("질문을 입력해 주세요.")
        continue

    response = chain_with_history.invoke(
        {"question": question},
        config=config,
    )

    print("AI:", response)
