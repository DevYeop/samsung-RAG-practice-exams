# =============================================================================
# LangChain LCEL 체이닝 이해하기
# =============================================================================
#
# [학습 목표]
# 1. prompt | llm | StrOutputParser()가 무엇을 만드는지 이해합니다.
# 2. chain.invoke()가 호출될 때 데이터가 어떤 순서와 타입으로 이동하는지 확인합니다.
# 3. 각 객체의 매개변수 개수가 달라도 연결되는 이유를 이해합니다.
# 4. 체이닝 코드와 각 단계를 직접 호출하는 코드가 사실상 같은 흐름임을 확인합니다.
#
# 핵심 한 문장:
# LCEL의 | 연산자는 함수를 즉시 실행하는 연산자가 아니라,
# 여러 Runnable을 순서대로 실행할 RunnableSequence 객체를 만드는 연산자입니다.
# =============================================================================

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableSequence

import os
import sys
from pathlib import Path

ROOT_DIR = next(parent for parent in Path(__file__).resolve().parents if (parent / "project_env.py").is_file())
sys.path.insert(0, str(ROOT_DIR))
from project_env import load_project_env, require_env

load_project_env()
require_env("OPENAI_API_KEY")

# =============================================================================
# 1. 체인에 사용할 구성 요소 만들기
# =============================================================================

# ChatPromptTemplate도 Runnable입니다.
# ⭐Runnable 이란? : 입력을 받아 출력을 반환하는 객체
#
# 입력:
#     {"topic": "RAG"}
#
# 출력:
#     ChatPromptValue
#
# ChatPromptValue는 단순 문자열이 아니라,
# 시스템 메시지와 사용자 메시지를 묶어 가진 프롬프트 객체입니다.
prompt = ChatPromptTemplate.from_messages([
    ("system", "당신은 기술을 쉽게 설명하는 강사입니다."),
    ("human", "{topic}을 한 문단으로 쉽게 설명해 주세요."),
])

# ChatOpenAI도 Runnable입니다.
#
# 대표적인 입력: 프롬프트로 전해지는 아래의 것들
#     ChatPromptValue
#     또는 메시지 리스트
#     또는 문자열
#
# 출력:
#     AIMessage
#
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0, # 창의성 낮게, 사실적 답변 높게. 
)
print("type(llm)")
print(type(llm))
# 타입이 ChatOpenAI이지만, Runnable 인터페이스를 구현하고 있기 때문에
# llm.invoke(input) 형태로 호출할 수 있습니다. ctrl + 클릭으로 ChatOpenAI 클래스를 클릭하면 상속구조를 역추적 가능
#
# [상속구조]
# ChatOpenAI
#    ↓
# BaseChatOpenAI
#    ↓
# BaseChatModel
#    ↓
# BaseLanguageModel
#    ↓
# RunnableSerializable
#    ↓
# Runnable

# StrOutputParser도 Runnable입니다.
#
# 입력:
#     AIMessage 또는 문자열
#
# 출력:
#     str
#
# ChatOpenAI가 반환한 AIMessage 객체에서
# 최종 답변 문자열을 꺼내기 위해 사용합니다.
parser = StrOutputParser()


# =============================================================================
# 2. | 연산자로 체인 만들기
# =============================================================================

# 여기서는 아직 LLM이 실행되지 않습니다.
#
# prompt | llm | parser
#
# 는 다음 실행 순서를 기억하는 RunnableSequence 객체를 만듭니다.
#
#     입력
#       ↓
#     prompt
#       ↓
#     llm
#       ↓
#     parser
#       ↓
#     최종 출력
chain = prompt | llm | parser


print("=" * 70)
print("1. chain 객체 확인")
print("=" * 70)

print("chain의 타입:", type(chain))
print("RunnableSequence인가?:", isinstance(chain, RunnableSequence))


# =============================================================================
# 3. chain.invoke() 실행
# =============================================================================

print("\n" + "=" * 70)
print("2. LCEL 체인 실행")
print("=" * 70)

result = chain.invoke({
    "topic": "RAG"
})

print("최종 결과 타입:", type(result))
print("최종 결과:")
print(result)


# =============================================================================
# 4. 체인을 사용하지 않고 각 단계를 직접 호출하기
# =============================================================================
#
# 아래 코드는 사실상 chain.invoke() 내부 흐름을 사람이 직접 작성한 것입니다.
#
# 중요한 점:
# 각 단계는 모두 동일한 형태의 실행 인터페이스를 가집니다.
#
#     output = runnable.invoke(input)
#
# 각 클래스의 생성자 매개변수나 내부 구현은 달라도,
# 연결할 때 사용하는 공통 규칙은 invoke(input)입니다.
# =============================================================================

print("\n" + "=" * 70)
print("3. 각 단계를 직접 실행")
print("=" * 70)


# 1단계: dict → ChatPromptValue
prompt_value = prompt.invoke({
    "topic": "RAG"
})

print("\n[1단계: prompt]")
print("입력 타입 : dict")
print("출력 타입 :", type(prompt_value))
print("출력 내용 :")
print(prompt_value)


# 2단계: ChatPromptValue → AIMessage
ai_message = llm.invoke(prompt_value)

print("\n[2단계: llm]")
print("입력 타입 :", type(prompt_value))
print("출력 타입 :", type(ai_message))
print("출력 내용 :")
print(ai_message)


# 3단계: AIMessage → str
parsed_result = parser.invoke(ai_message)

print("\n[3단계: parser]")
print("입력 타입 :", type(ai_message))
print("출력 타입 :", type(parsed_result))
print("출력 내용 :")
print(parsed_result)


# =============================================================================
# 5. 타입의 이동 경로
# =============================================================================
#
# 이 체인에서 데이터 타입은 다음과 같이 변합니다.
#
# dict
#   ↓ ChatPromptTemplate
# ChatPromptValue
#   ↓ ChatOpenAI
# AIMessage
#   ↓ StrOutputParser
# str
#
# 앞 단계의 출력 타입을 다음 단계가 입력으로 받을 수 있어야 합니다.
#
# 각 객체가 Runnable이라는 공통 실행 규약을 따르기 때문에 연결됩니다.
#
# 공통 실행 규약:
#     runnable.invoke(input)
#
# 실제 클래스마다 받을 수 있는 input의 종류는 다르지만,
# 호출하는 사용법은 invoke 하나로 통일되어 있습니다.
# =============================================================================


# =============================================================================
# 6. 파이썬 함수와 비교하기
# =============================================================================
#
# 일반 함수 합성으로 표현하면 아래와 비슷합니다.
#
    # result = 
    # parser(
    #     llm(
    #         prompt(input_data)
    #     )
    # ) 
# promt, llm, parser 순으로 함수가 실행되고,
# 각각의 함수가 실행되고 return 값을 다음 함수의 인자로 전달하게 됨

# 개념적으로 이와 같은 흐름을 띈다
input_data = {
    "topic": "인공지능"
}

step1 = prompt.invoke(input_data)
step2 = llm.invoke(step1) # 아웃풋, 답변
step3 = StrOutputParser().invoke(step2)

result = step3

# LCEL에서는 위와 같은 흐름을 왼쪽에서 오른쪽으로 읽을 수 있게 표현합니다.
#
#     chain = prompt | llm | parser
#     result = chain.invoke(input_data)
#
# 즉, LCEL은 여러 처리 단계를 읽기 쉬운 파이프라인으로 표현하는 문법입니다.
# =============================================================================

   
