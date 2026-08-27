# =============================================================================
# 02. RAG 답변 과정을 체인으로 연결
# =============================================================================
#
# [이 파일의 학습 목표]
# 1단계에서 직접 작성했던 '프롬프트 만들기 -> LLM 호출 -> 답변 글 꺼내기'를
# LangChain의 체인으로 연결하는 방법을 배웁니다.
#
# 이 단계에서 가장 중요한 표현은 다음 한 줄입니다.
#
#     chain = prompt | llm | StrOutputParser()
#
# 파이썬의 `|`는 원래 '파이프'라고 읽습니다. 여기서는 앞 단계의 결과를 뒤 단계의
# 입력으로 전달한다는 뜻입니다. 물이 관을 따라 순서대로 흐른다고 생각하면 쉽습니다.
#
# [전체 처리 흐름]
# 1. PDF를 읽습니다.
# 2. PDF를 작은 문서 조각으로 나눉니다.
# 3. 문서 조각을 임베딩해 벡터 DB에 저장합니다.
# 4. 사용자 질문과 관련된 문서를 검색합니다.             <- 직접 작성
# 5. 검색된 문서를 문자열로 합칩니다.               <- 직접 작성
# 6. context와 question을 딕셔너리로 chain에 넣습니다.    <- 여기서부터 체인
# 7. prompt가 완성된 메시지를 만듭니다.
# 8. llm이 답변합니다.
# 9. StrOutputParser가 답변 객체에서 글만 꺼냅니다.
#
# [이 버전이 잘 맞는 용도]
# - 1단계의 RAG 흐름을 이해한 뒤 LangChain 체인 문법을 처음 배울 때
# - 프롬프트와 LLM을 바꾸며 여러 번 실험할 때
# - 검색 결과는 직접 확인하면서 답변 생성 부분만 재사용하고 싶을 때
#
# [장점]
# - 문자열 조합보다 프롬프트의 입력 자리를 명확하게 관리할 수 있습니다.
# - AIMessage에서 `.content`를 직접 꺼내지 않아도 최종 결과가 문자열로 나옵니다.
# - 여러 처리 단계를 이름이 있는 하나의 chain으로 재사용할 수 있습니다.
#
# [단점]
# - 코드가 짧아지는 대신, 데이터가 어떤 형태로 넘어가는지 처음에는 덜 보일 수 있습니다.
# - 검색과 context 생성은 여전히 체인 밖에 있어, 질문 하나만으로 전체가 실행되지는 않습니다.
#
# [1단계와의 차이]
# 1단계: f-string으로 prompt를 만들고 `llm.invoke(prompt)`를 직접 호출합니다.
# 2단계: 템플릿에 context와 question을 넣으면 나머지 답변 과정을 chain이 실행합니다.
#
# [3단계와의 차이]
# 2단계: 사용자가 docs와 context를 직접 만들어 chain에 넣습니다.
# 3단계: 질문 하나만 넣으면 검색과 context 생성까지 chain 안에서 실행됩니다.
# =============================================================================

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate

# StrOutputParser는 LLM의 AIMessage 객체에서 답변 글만 꺼내 평범한 문자열로 바꿉니다.
from langchain_core.output_parsers import StrOutputParser

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
# PDF 읽기
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
# 문서 분할
# -----------------------------------------------------------------------------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

splits = text_splitter.split_documents(pages)

# -----------------------------------------------------------------------------
# 벡터 DB 생성
# -----------------------------------------------------------------------------
vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=OpenAIEmbeddings()
)

# -----------------------------------------------------------------------------
# 검색기 생성
# -----------------------------------------------------------------------------
retriever = vectorstore.as_retriever()

# -----------------------------------------------------------------------------
# 프롬프트 템플릿 생성
# -----------------------------------------------------------------------------
# 템플릿은 '빈칸이 있는 양식'입니다.
# `{context}`와 `{question}`은 아직 실제 내용이 아니라, 나중에 값이 들어갈 자리를 표시합니다.
# 1단계의 f-string은 즉시 문자열을 완성했지만, ChatPromptTemplate은 실행할 때 값을 채웁니다.
prompt = ChatPromptTemplate.from_template("""
다음 문서를 참고하여 답변하세요.

문서:
{context}

질문:
{question} 
""")
# -> 프롬프트 완성하게 되고
# 그 완성된거를 다음 체인한테 넘겨주는 것

# -----------------------------------------------------------------------------
# 체인 생성
# -----------------------------------------------------------------------------
# LCEL(LangChain Expression Language)은 `|`를 사용해 여러 처리 단계를 연결하는 문법입니다.
#
# 데이터는 다음 순서로 이동합니다.
# 1. prompt: 딕셔너리에서 context와 question을 꺼내 완성된 프롬프트를 만듭니다.
# 2. llm: 완성된 프롬프트를 읽고 AIMessage 형태로 답변합니다.
# 3. StrOutputParser: AIMessage에서 답변 글만 꺼내 str 형태로 돌려줍니다.
#
# 주의: 현재 단계에서는 chain 안에 retriever가 없습니다.
# 때문에 사용자가 context를 직접 만들어 넣어야 합니다.
chain = prompt | llm | StrOutputParser()

# 인공지능 답변중에서
# content를 뽑아서 순수 문자열로 그냥 반환하겠다


# 답변이 생성되서 -> 다음으로 전달 (그걸 받는 주인공이 아웃풋파서)
# 답변을 재가공해서, 우리가 원하던 형태로 가공을 해주고 


# -----------------------------------------------------------------------------
# 사용자 질문과 문서 검색
# -----------------------------------------------------------------------------
# 여기까지는 1단계와 같습니다. 질문 문자열을 만든 뒤 retriever를 직접 호출합니다.
question = "근육 성장을 위해 가장 중요한 것은?"

# `docs`는 검색된 Document 객체들의 목록입니다.
docs = retriever.invoke(question)

# -----------------------------------------------------------------------------
# 체인 실행
# -----------------------------------------------------------------------------
# `chain.invoke()`의 입력은 템플릿의 빈칸 이름과 같은 키를 가진 딕셔너리여야 합니다.
#
# - `context` 키: 검색된 Document의 page_content를 합친 긴 문자열
# - `question` 키: 사용자가 입력한 질문 문자열
#
# chain은 이 딕셔너리를 받은 뒤 `prompt -> llm -> StrOutputParser`를 순서대로 실행합니다.
# 최종 `response`는 이미 StrOutputParser를 통과했으므로 AIMessage가 아닌 평범한 문자열입니다.
response = chain.invoke({
    # 리스트 내포와 join을 사용해 Document 목록을 프롬프트에 넣을 글로 바꿉니다.
    "context": "\n\n".join(
        doc.page_content for doc in docs
    ),
    "question": question
})

# 결과출력. response가 이미 문자열이므로 1단계처럼 `.content`를 붙이지 않습니다.
print(response)
