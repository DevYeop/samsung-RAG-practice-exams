# =============================================================================
# 03. RAG 검색부터 답변까지 체인으로 연결
# =============================================================================
#
# [이 파일의 학습 목표]
# 질문 하나만 넣으면 '문서 검색 -> context 생성 -> 프롬프트 완성 -> LLM 호출
# -> 문자열 출력'이 모두 실행되는 전체 RAG 체인을 만듭니다.
#
# 1단계는 모든 과정을 직접 적었고, 2단계는 답변 생성 부분만 체인으로 묶었습니다.
# 3단계는 검색기(Retrieval)까지 체인 안에 넣어,
# 실제 앱이나 함수에서 반복 사용하기 좋은 형태로 만듭니다.
#
# [전체 처리 흐름]
# 문서 준비 단계는 앞의 파일과 같습니다.
#
# 1. PDF를 읽습니다.
# 2. PDF를 작은 청크로 나눕니다.
# 3. 청크를 임베딩해 벡터 DB에 저장합니다.
# 4. 벡터 DB를 retriever로 바꿉니다.
#
# 질문을 넣은 뒤부터는 다음처럼 실행됩니다.
#
#                     +-> retriever -> Document 목록 -> format_docs -> context
# 사용자 질문 ------+
#                     +-> RunnablePassthrough -----------------------> question
#
# context와 question이 하나의 딕셔너리로 모이면 다음 단계로 흐릅니다.
#
# {"context": 검색된 문서, "question": 원래 질문}
#     -> prompt -> llm -> StrOutputParser -> 최종 답변 문자열
#
# [이 버전이 잘 맞는 용도]
# - 질문만 바꾸어 같은 RAG 과정을 반복 실행할 때
# - 챗봇, API, Streamlit 앱 등에 RAG 기능을 하나의 처리 단위로 연결할 때
# - LangChain의 LCEL 문법과 Runnable 개념을 복습할 때
#
# [장점]
# - `rag_chain.invoke(질문)` 한 줄로 검색부터 답변까지 실행할 수 있습니다.
# - 반복 코드가 줄고, 완성된 체인을 다른 코드에서 재사용하기 쉽습니다.
#
# [단점]
# - 코드는 짧지만, 한 질문이 여러 경로로 나뉘다가 다시 합쳐지는 과정을 머릿속으로
#   추적해야 하므로 1단계보다 어렵게 느껴질 수 있습니다.
# - 중간값인 docs와 context를 확인하려면 별도로 중간에 로깅 및 디버깅하는 작업이 필요할 수 있습니다.
# - 어느 단계에서 오류가 났는지 찾을 때 LCEL의 데이터 전달 규칙을 알아야 합니다.
#
# [1·2단계와의 차이]
# 1단계: 모든 과정을 변수와 invoke로 직접 작성합니다. 가장 쉽게 흐름을 볼 수 있습니다.
# 2단계: `prompt | llm | parser`만 체인입니다. docs와 context는 사용자가 직접 만듭니다.
# 3단계: `retriever | format_docs`도 체인에 포함됩니다. 입력은 질문 문자열 하나면 됩니다.
#
# [읽을 때 기억할 핵심]
# - `RunnablePassthrough()`는 어려운 처리를 하지 않습니다. 받은 입력을 그대로 넘깁니다.
# - 딕셔너리의 `context`와 `question`은 동일한 질문에서 동시에 만들어집니다.
# - `context`는 검색기를 거쳐 변환된 값이고, `question`은 원본을 그대로 유지한 값입니다.
# =============================================================================


from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# RunnablePassthrough는 입력받은 값을 바꾸지 않고 그대로 다음 단계로 넘깁니다.
from langchain_core.runnables import RunnablePassthrough

import os
import sys
from pathlib import Path

ROOT_DIR = next(parent for parent in Path(__file__).resolve().parents if (parent / "project_env.py").is_file())
sys.path.insert(0, str(ROOT_DIR))
from project_env import load_project_env, require_env

load_project_env()
require_env("OPENAI_API_KEY")


# -----------------------------------------------------------------------------
# LLM 생성
# -----------------------------------------------------------------------------
llm = ChatOpenAI(model="gpt-4o-mini")

# -----------------------------------------------------------------------------
# 1. PDF 불러오기
# -----------------------------------------------------------------------------
# 기존 방식: 실행한 작업 폴더를 기준으로 PDF를 찾기 때문에 실행 위치에 따라 실패할 수 있습니다.
# loader = PyPDFLoader("./Maximizing Muscle Hypertrophy.pdf")

# 현재 파이썬 파일이 있는 폴더를 기준으로 PDF의 절대 경로를 만듭니다.
# 따라서 프로젝트 루트나 다른 폴더에서 이 파일을 실행해도 같은 PDF를 찾습니다.
SCRIPT_DIR = Path(__file__).resolve().parent
PDF_PATH = SCRIPT_DIR / "Maximizing Muscle Hypertrophy.pdf"

if not PDF_PATH.is_file():
    raise FileNotFoundError(f"PDF 파일을 찾을 수 없습니다: {PDF_PATH}")

loader = PyPDFLoader(str(PDF_PATH))
pages = loader.load()


# -----------------------------------------------------------------------------
# 2. PDF 문서 잘게 나누기
# -----------------------------------------------------------------------------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

splits = text_splitter.split_documents(pages)


# -----------------------------------------------------------------------------
# 3. 문서를 임베딩하여 벡터 DB에 저장
# -----------------------------------------------------------------------------
vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=OpenAIEmbeddings()
)

# -----------------------------------------------------------------------------
# 4. 벡터 DB를 검색기로 변환
# -----------------------------------------------------------------------------
retriever = vectorstore.as_retriever()

# -----------------------------------------------------------------------------
# 검색된 Document 객체들을 문자열로 합치는 함수
# -----------------------------------------------------------------------------
# retriever의 출력은 Document 객체의 목록이지만, 프롬프트에는 글을 넣어야 합니다.
# `format_docs()`는 각 Document의 `page_content`를 꺼낸 뒤 빈 줄로 구분해 하나의 긴 문자열로 만듭니다.
def format_docs(docs):
    return "\n\n".join(
        doc.page_content for doc in docs
    )

# -----------------------------------------------------------------------------
# 7. 프롬프트 템플릿
# -----------------------------------------------------------------------------
# `{context}`와 `{question}`은 실행할 때 값이 채워지는 빈칸입니다.
# - context에는 retriever가 찾고 format_docs가 합친 문서가 들어갑니다.
# - question에는 RunnablePassthrough가 그대로 전달한 사용자 질문이 들어갑니다.
prompt = ChatPromptTemplate.from_template("""
다음 문서를 참고하여 질문에 답변하세요.

문서:
{context}

질문:
{question}

답변:
""")

# -----------------------------------------------------------------------------
# 8. RAG 체인 생성
# -----------------------------------------------------------------------------
# 이 파일의 핵심 부분입니다.
#
# [첫 번째 단계: 질문 하나를 두 경로로 나누기]
# 딕셔너리의 각 키는 같은 입력 질문을 받아 자신의 방식으로 값을 만듭니다.
#
# `"context": retriever | format_docs`
# 1. 질문 문자열이 retriever에 들어갑니다.
# 2. retriever가 관련 Document 목록을 돌려줍니다.
# 3. Document 목록이 format_docs로 넘어갑니다.
# 4. format_docs가 하나의 context 문자열로 바꿉니다.
#
# `"question": RunnablePassthrough()`
# 1. 같은 질문 문자열을 받습니다.
# 2. 수정하거나 분석하지 않고 그대로 돌려줍니다.
#
# 두 경로의 결과가 모이면 다음과 같은 딕셔너리가 됩니다.
#
# {
#     "context": "검색된 문서들을 합친 글",
#     "question": "사용자가 처음에 넣은 질문"
# }
#
# [두 번째 단계: 답변 만들기]
# 1. prompt가 딕셔너리의 두 값을 템플릿에 채웁니다.
# 2. llm이 완성된 프롬프트를 읽고 답변을 생성합니다.
# 3. StrOutputParser가 생성된 답변을 가공합니다.
rag_chain = (
    {
        # 하나의 입력 질문을 두 갈래로 보냅니다.
        # - context 경로: 문서를 검색해서 참고 자료를 만듭니다.
        ## -- "근육 성장을 위해 가장 중요한 요소는 무엇인가요?"에 유사한 문서를 검색합니다.
        # - question 경로: 원본 질문을 그대로 유지합니다.
        ## -- "근육 성장을 위해 가장 중요한 요소는 무엇인가요?"를 그대로 다음 단계인 prompt로 넘깁니다.
        "context": retriever | format_docs,
        "question": RunnablePassthrough() # 마치 "근육 성장을 위해 가장 중요한 요소는 무엇인가요?"값이 적용되어 있는 셈
    }
    | prompt # 프롬프트 템플릿에 context와 question을 채웁니다.
    | llm # 완성된 프롬프트를 읽고 답변을 생성합니다.
    | StrOutputParser() # StrOutputParser가 답변 글만 꺼내 str 형태로 돌려줍니다.
)

# -----------------------------------------------------------------------------
# 9. 질문만 넣어서 전체 RAG 실행
# -----------------------------------------------------------------------------
# 2단계에서는 `{"context": ..., "question": ...}` 딕셔너리를 직접 만들어 넣었습니다.
# 3단계에서는 질문 문자열 하나만 넣습니다. rag_chain이 필요한 context를 스스로 검색합니다.
#
# 입력: str 형태의 질문
# 출력: StrOutputParser가 만든 str 형태의 최종 답변
response = rag_chain.invoke(
    "근육 성장을 위해 가장 중요한 요소는 무엇인가요?"
)

# response가 이미 문자열이므로 `.content`를 붙이지 않고 바로 출력합니다.
print(response)
