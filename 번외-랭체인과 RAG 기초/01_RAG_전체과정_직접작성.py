# =============================================================================
# 01. RAG 전체 과정 직접 작성
# =============================================================================
#
# [전체 처리 흐름]
# 1. PDF를 읽어 Document 객체로 변환합니다.
# 2. 긴 문서를 검색하기 좋은 작은 조각으로 나눕니다. (청킹 chunking)
# 3. 각 조각의 내용을 숫자 목록인 임베딩으로 바꾸어 벡터 DB에 저장합니다.
# --> 텍스트 to 수치화 -> 임베딩, 벡터화
# 4. 사용자의 질문도 임베딩으로 바꾸고, 비슷한 문서 조각을 찾습니다.
# --> 사용자 질문도 벡터화 시키고, 그걸 기반으로 유사도 검색을 하면서
# ---> 그와 유관한 문서를 찾게되는 원리입니다.
# ----> 인간이 유사하다라는 걸 판단하는 원리랑 vs 컴퓨터가 지금 유사하다를 판단하는 원리가 완전히 달랐죠
# 5. 찾은 문서 조각을 하나의 context 문자열로 합칩니다.
# ---> conext(맥락, 정보, 기억, 상황, 정황)=> 결국 다 데이터! 입니다
# 6. context를 프롬프트에 추가시킵니다.
# ---> 원래 우리가 날렸던 프롬프트에, rag시스템을 통해 추출된 문서정보가
# ---> context(data, 문서정보, 관련정보) 들어가게 되는 것이죠.
# ---> 즉, 기존 프롬프트를 증강(Argumented)시켜주는 역할을 합니다.
# 7. 완성된 프롬프트를 LLM에 보냅니다.
# 8. LLM의 답변 내용을 출력합니다.
#
# [장점]
# - 데이터가 이동하는 순서가 코드에 그대로 드러납니다.
# - docs, context, prompt, response를 각각 확인하기 쉽습니다.
# - LangChain의 체인 문법을 모르더라도 파이썬 기초만 알면 흐름을 읽을 수 있습니다.
#
# [단점]
# - 질문을 할 때마다 검색, 문서 합치기, 프롬프트 작성, LLM 호출을
#   직접 써야 하므로 반복 사용하기에는 코드가 길어집니다.
# - 프롬프트가 일반 문자열이라서 입력 변수가 늘어나면 관리하기 불편해집니다.
#
# [2단계 파일과의 차이]
# 이 파일은 프롬프트 문자열을 직접 만들고 LLM을 직접 호출합니다.
# 2단계에서는 이 부분을 `프롬프트 | LLM | 출력 변환기`라는 체인으로 묶습니다.
# 따라서 1단계는 '원리 이해', 2단계는 '반복 코드 정리'에 더 적합합니다.
# =============================================================================

# ChatOpenAI: 채팅 형태의 OpenAI LLM을 LangChain에서 사용하게 해 줍니다.
# OpenAIEmbeddings: 글의 의미를 컴퓨터가 비교할 수 있는 숫자 목록으로 바꿉니다.
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# PyPDFLoader: PDF를 읽어 LangChain의 Document 객체 목록으로 바꿉니다.
from langchain_community.document_loaders import PyPDFLoader

# Chroma: 임베딩된 문서를 저장하고 의미가 비슷한 문서를 찾는 벡터 DB입니다.
from langchain_chroma import Chroma

# RecursiveCharacterTextSplitter: 긴 글을 문단과 문장 경계를 고려해 작게 나눕니다.
from langchain_text_splitters import RecursiveCharacterTextSplitter

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
# PDF를 읽을 로더(loader)를 준비
# 기존 방식: 실행한 작업 폴더를 기준으로 PDF를 찾기 때문에 실행 위치에 따라 실패할 수 있습니다.
# loader = PyPDFLoader("./Maximizing Muscle Hypertrophy.pdf")

# 현재 파이썬 파일이 있는 폴더를 기준으로 PDF의 절대 경로를 만듭니다.
# 따라서 프로젝트 루트나 다른 폴더에서 이 파일을 실행해도 같은 PDF를 찾습니다.
SCRIPT_DIR = Path(__file__).resolve().parent
PDF_PATH = SCRIPT_DIR / "Maximizing Muscle Hypertrophy.pdf"

if not PDF_PATH.is_file():
    raise FileNotFoundError(f"PDF 파일을 찾을 수 없습니다: {PDF_PATH}")

loader = PyPDFLoader(str(PDF_PATH))

# `load()`를 호출해야 실제 PDF 내용을 읽습니다.
# `pages`는 Document 객체의 목록(페이지 만큼, document객체가 여러개) 들어있습니다.
# 각 Document의 `page_content`에는 본문이, `metadata`에는 페이지 번호 등이 들어 있습니다.
pages = loader.load()

# -----------------------------------------------------------------------------
# 문서 분할
# -----------------------------------------------------------------------------
# PDF 전체를 한번에 검색하기보다 작은 조각으로 나누면 질문과 관련된 부분을
# 더 정밀하게 찾을 수 있습니다. 이렇게 나눈 조각을 '청크(chunk)'라고 부릅니다.
text_splitter = RecursiveCharacterTextSplitter(
    # 청크 하나의 목표 크기입니다.
    # '대략, 약 1천자'를 기준으로 나눈다라고 생각하면 쉽습니다. 
    chunk_size=1000,


    # 인접한 청크가 200만큼 겹치게 합니다.
    # 문장이 청크 경계에서 끊기더라도 앞뒤 맥락을 잃을 가능성을 줄입니다.
    # 웹툰에서 이전화의 마지막 몇 컷을, 현재화 초반에 겹쳐서 제공하듯이
    # 문장이 짤리면서 맥락이 끊기는 것을 방지하기 위해서 오버랩을 해주는 것과 비슷합니다.
    chunk_overlap=200
)

# `split_documents()`는 pages의 각 Document를, 위에서 설정한 값을 참조하여 작은 Document로 나눕니다.
# `splits`는 분할된 보다 더 작은형태의 Document(문서조각) 객체들의 리스트입니다.
splits = text_splitter.split_documents(pages)

# -----------------------------------------------------------------------------
# 벡터 DB 생성
# -----------------------------------------------------------------------------
# 임베딩은 문장의 '의미'를 여러 개의 숫자로 표현한 것입니다.
# ex) "나는 오늘 점심으로 라면을 먹었다"라는 문장은
# [0.123, 0.456, 0.789, ...]처럼 수천 개의 숫자 목록으로 바뀌면서, 벡터(Vector)로 변환됩니다. 
#
# 의미가 비슷한 글은 숫자 공간에서도 가깝게 배치된다는 아이디어를 사용합니다.
# ex) 우리가 벡터와 유사도에 대한 개념을 쉽게 이해하기 위해 본 3차원 차트 이미지에서,
#     dog와 wolf는 가깝게, dog와 car는 멀리 떨어져 있었던 것처럼.
# 자료 링크: https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQ5NyCLd8kjuM-nlnWMdPOSImUiitot-4aa8_4r_LJM2w&s=10
#
# `from_documents()`는 다음 작업을 순서대로 처리합니다.
# 1. splits의 문서조각들을 OpenAIEmbeddings로 임베딩합니다.
# 2. 임베딩, 원본 본문, 메타데이터를 Chroma에 저장합니다.
#
# 현재는 `persist_directory`가 없으므로 실행할 때마다 새로 만듭니다.
# 학습용으로는 간단하지만,
# 저장할 문서가 많아질수록 그만큼 비용과 시간이 반복되는 단점이 있습니다.
vectorstore = Chroma.from_documents(
    documents=splits, # 조각조각낸 문서들을 
    embedding=OpenAIEmbeddings() # 해당 임베딩 모델을 통해서 백터화하겠다!
)

# -----------------------------------------------------------------------------
# 검색기 생성
# -----------------------------------------------------------------------------
# 벡터 DB는 '저장과 비교'를 담당하고, retriever는 '질문을 받아 관련 문서를 찾기'를
# 담당합니다. `as_retriever()`는 vectorstore를 LangChain의 표준 검색기 형태로 바꿉니다.
retriever = vectorstore.as_retriever()

# 아래 코드처럼 사실, 서칭되는 문서 개수를 직접 지정할 수도 있습니다.
# retriever = vectorstore.as_retriever(
#     search_kwargs={"k": 20}
# )

# -----------------------------------------------------------------------------
# 사용자 질문
# -----------------------------------------------------------------------------
# `question`은 평범한 파이썬 문자열입니다.
# 실제 챗봇에서는 입력창에서 받은 글이 이 변수에 들어간다고 생각하면 됩니다.
# question = "근육 성장을 위해 가장 중요한 것은?"
# question = "근비대 기본 원칙이 무엇이고, 슈퍼세트와 드롭세트의 차이점에 대해서 알려줘"
question = "근비대 기본 원칙은?"



# -----------------------------------------------------------------------------
# 관련 문서 검색
# -----------------------------------------------------------------------------
# `invoke()`는 '이 입력을 넣어 실행해 달라'는 LangChain의 공통 호출 방식입니다.
# retriever는 내부적으로 질문을 임베딩한(백터화한 뒤) 뒤,
# 저장된 청크 임베딩(문서가 백터화된 것들)과 비교합니다.
# `docs`는 질문과 의미가 가깝다고 판단된 Document 객체들의 목록입니다.
docs = retriever.invoke(question)

# -----------------------------------------------------------------------------
# 검색된 문서를 하나의 문자열로 합치기
# -----------------------------------------------------------------------------
# LLM은 Document 객체 목록을 그대로 읽는 것이 아니라
# 그저 글로 작성된 프롬프트를 읽을 뿐 입니다.
# 그래서 각 Document의 `page_content`만 꺼내어하나의 큰 문자열로 합쳐줌으로써
# 프롬프트를 증강(Argumented) 시켜줍니다.
# `"\n\n"`은 문서 조각 사이에 빈 줄을 넣어 경계를 구분하는 역할을 합니다.
    
context_list = []

for i, doc in enumerate(docs):
    print(f"\n===== 선택된 청크 {i + 1} =====")
    print("metadata:", doc.metadata)
    print("page_content:")
    print(repr(doc.page_content))  # 줄바꿈까지 raw하게 확인
    print("=" * 50)

    context_list.append(doc.page_content)

context = "\n\n".join(context_list)


context = "\n\n".join(
    doc.page_content
    for doc in docs
) # 청크들이 모조리 합해진 문자열



# -----------------------------------------------------------------------------
# 프롬프트 직접 생성
# -----------------------------------------------------------------------------
# 프롬프트는 LLM에게 보내는 전체 요청문입니다.
# f-string의 `{context}`와 `{question}` 자리에 위에서 만든 실제 문자열이 바로 들어갑니다.
prompt = f"""
## role (역할)
다음 문서를 참고하여 답변하세요.

문서: 
{context} 

질문:
{question}
"""
# ⭐위의 프롬프트가 완성되는 것을 보면 알 수 있듯이 RAG 를 통해서 관련 context를 기존 프롬프트에 첨부해 줌으로써써, 기존 프롬프트를 증강(Argumented)시켜주고 있다.

# -----------------------------------------------------------------------------
# LLM 호출
# -----------------------------------------------------------------------------
# 완성된 prompt 문자열을 llm에 넣어 답변을 생성합니다.
response = llm.invoke(prompt)

# 결과 출력
print(response.content)
# print(response)
