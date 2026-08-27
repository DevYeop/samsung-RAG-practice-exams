#!/usr/bin/env python
# coding: utf-8

# # Hybrid searching을 활용한 RAG 챗봇 고도화
# 
# ### 실습 목표
# 
# 이 실습은 RAG(Retrieval-Augmented Generation) 과정에서 Dense(LLama 임베딩 기반)와 Sparse(BM25 기반) 검색을 모두 경험해보고, 두 검색 결과를 결합한 후 HuggingFace Cross Encoder 기반의 재정렬(Reranking)을 적용하여 최종적으로 RAG 챗봇의 답변 품질을 향상시키는 방법을 학습하는 데 목적이 있습니다.
# 
# ### 실습 목차
# 1. 문서 로드 및 전처리 (PDF 파일 로드 및 청크 분할)
# 2. Dense Retrieval (LLama 임베딩 기반)
# 3. Sparse Retrieval (BM25 기반)
# 4. 두 검색 결과 결합 및 Cross Encoder Reranking
# 5. RAG 챗봇 워크플로우 구성 및 실행

# ### 0. 환경 설정
# 
# 필요한 라이브러리를 아래와 같이 불러옵니다.

# In[ ]:


import os
from time import time
from langchain.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_core.documents import Document
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langgraph.graph import END, StateGraph
from typing import List
from typing_extensions import TypedDict


# 참고: 실습에서는 Ollama를 통해 llama 3.1 8B 모델을 사용하므로,
# 아래와 같이 해당 모델을 미리 다운로드합니다.
# 
# ```bash
# !ollama pull llama3.1
# ```

# ### 1. 문서 로드 및 전처리
# - PDF 문서 로드 및 청크 분할:
# 지정된 데이터 경로의 PDF 파일을 `PyPDFLoader`로 불러오고, `RecursiveCharacterTextSplitter`를 이용해 청크 크기 500, 겹침 100으로 분할합니다.

# In[ ]:


# 데이터 경로 및 문서 파일명 지정 (파일 경로에 맞게 수정)
data_dir = "data"
pdf_data_name = "RE177_2023년 국내외 인공지능 산업 동향 연구_2장.pdf"

# PDF 문서 로드
loader = PyPDFLoader(os.path.join(data_dir, pdf_data_name))
docs = loader.load()

# 텍스트 분할: 청크 크기 500, 겹침 100
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
docs = text_splitter.split_documents(docs)

print(f"총 Document 개수: {len(docs)}")


# ### 2. Dense Retrieval (LLama 임베딩 기반)
# - LLama 임베딩 벡터 DB 구축:
# `OllamaEmbeddings`를 사용하여 Dense 임베딩을 생성한 후, Chroma 벡터 스토어에 저장합니다. 임베딩 생성 소요시간을 측정하여 출력합니다.
# - Dense 검색 실행:
# 생성된 벡터 스토어에서 쿼리(“프런티어 모델 포럼에 대해 설명해”)에 대해 상위 5개의 문서를 `similarity_search_with_score`를 통해 회수하고, 각 문서의 유사도 점수를 출력합니다.

# In[ ]:


# Dense Embedding: LLama Embedding Layer 기반 벡터 DB 생성 및 검색
print("=== Step 1: Dense Retrieval (LLama Embedding) ===")
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain.vectorstores import Chroma


# In[ ]:


# LLama embedding layer를 활용해 벡터 DB 구축 (Chroma 사용)
llama_embeddings = OllamaEmbeddings(model="llama3.1")
start = time()
dense_vectorstore = Chroma.from_documents(
    docs, 
    embedding=llama_embeddings, 
    persist_directory="./vectorstore/dense"
)
end = time()
print(f"Embedding 소요 시간: {end - start:.5f} sec")
# 상위 5개 문서를 반환하도록 retriever 생성
dense_retriever = dense_vectorstore.as_retriever(search_kwargs={"k": 5})


# In[ ]:


query = "프런티어 모델 포럼에 대해 설명해"

# 소요시간 확인
start = time()

# Dense retrieval 시, 문서와 유사도 점수를 함께 확인
dense_results_with_scores = dense_vectorstore.similarity_search_with_score(query, k=5)

end = time()

print(f"Dense Retrieval running time: {end - start:.5f} sec\n")

print("Dense Retrieval 결과 (점수 포함):")

for i, (doc, score) in enumerate(dense_results_with_scores, start=1):
    print(f"[Dense] Rank {i} | Score: {score:.4f}")
    print("내용:", doc.page_content)
    print("-" * 50)

print("=" * 70)


# ### 3. Sparse Retrieval (BM25 기반)
# - BM25Retriever 활용:
# `BM25Retriever`를 사용하여 동일 쿼리에 대해 Sparse 방식의 검색을 수행합니다. BM25는 점수 정보를 반환하지 않으므로, 문서 내용만 출력합니다.

# In[ ]:


# 3. Sparse Retrieval: BM25 기반 검색
print("=== Step 2: Sparse Retrieval (BM25) ===")
from langchain.retrievers import BM25Retriever

# BM25Retriever를 이용해 sparse 검색 수행
bm25_retriever = BM25Retriever.from_documents(docs)
bm25_retriever.k = 5  # 상위 5개 문서 회수

# 소요시간 확인
start = time()

bm25_results = bm25_retriever.invoke(query)

end = time()

print(f"Sparse Retrieval running time: {end - start:.5f} sec\n")

print("BM25 Retrieval 결과:")
for i, doc in enumerate(bm25_results, start=1):
    # BM25 결과에는 Score 정보가 포함되지 않음
    print(f"[BM25] Rank {i}")
    print("내용:", doc.page_content)
    print("-" * 50)

print("=" * 70)


# ### 4. 두 검색 결과 결합 및 Reranking
# - 결과 결합:
# Dense 검색 결과는 (문서, 점수) 튜플 형태이므로 문서 부분만 추출하고, BM25 결과는 그대로 결합합니다.
# - Cross Encoder Reranker 적용:
# `HuggingFaceCrossEncoder`를 이용해 Cross Encoder 모델을 초기화한 후, `CrossEncoderReranker`를 통해 결합된 문서를 재정렬하여 상위 3개의 문서를 선택합니다.

# In[ ]:


# 4. 두 검색 결과 결합 및 Reranking (상위 3개 선택)
print("=== Step 3: Combined Retrieval & Reranking ===")

# Dense 검색 결과는 (doc, score) 튜플이므로, 문서 부분만 추출합니다.
dense_docs = [doc for doc, _ in dense_results_with_scores]

# BM25 결과는 이미 Document 객체입니다.
combined_docs = dense_docs + bm25_results

# Reranker 설정: HuggingFace 기반 CrossEncoder 모델 사용 (상위 3개 선택)
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

top_n = 3

cross_encoder = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-v2-m3")
reranker = CrossEncoderReranker(model=cross_encoder, top_n=top_n)

# 결합된 문서에 대해 reranking 수행
top_docs = reranker.compress_documents(combined_docs, query)
print("Reranker를 통한 상위 3개 문서:")
for i, doc in enumerate(top_docs, start=1):
    print(f"Rank {i}")
    print("내용:", doc.page_content)
    print("-" * 50)

print("=" * 70)


# In[ ]:


combined_docs


# In[ ]:


# combined_docs 리스트에 있는 각 문서에 대해 반복합니다.
for combined_doc in combined_docs:
    text_pair = (query, combined_doc.page_content)
    
    # 생성한 텍스트 쌍에 대한 유사도 점수를 계산합니다.
    scores = cross_encoder.score([text_pair])
    
    # 리스트에 단 하나의 텍스트 쌍만 전달했으므로, 첫 번째 점수를 추출합니다.
    similarity_score = scores[0]
    
    # 현재 문서에 대한 유사도 점수를 출력합니다.
    print(f"문서 유사도 점수: {similarity_score:.4f}")


# ### 5. RAG 챗봇 워크플로우 구성 및 실행
# - 그래프 구성:
# 각 기능(답변 생성, 데이터 검색, 검색된 데이터를 바탕으로 답변 생성 등)을 그래프 노드로 정의하고, 조건에 따라 분기하는 워크플로우를 구성합니다.
# - 라우팅 체인:
# ChatOllama를 활용하여 질문의 특성에 따라 RAG 사용 여부를 결정하는 라우터 체인을 구성합니다.
# - 실행:
# 구성된 그래프를 컴파일하고, 사용자의 입력에 따라 챗봇이 답변을 생성합니다.

# In[ ]:


llm = ChatOllama(model="llama3.1")
route_llm = ChatOllama(model="llama3.1", format="json")

# # Dense Retrieval을 위한 Chroma 벡터 스토어 (OllamaEmbeddings 활용)
# loader = PyPDFLoader(os.path.join(data_dir, pdf_data_name))
# docs = loader.load()
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
# docs = text_splitter.split_documents(docs)
# print(f"총 Document 개수: {len(docs)}")

# # Dense 임베딩 생성 (OllamaEmbeddings 이용)
# llama_embeddings = OllamaEmbeddings(model="llama3.1")
# dense_vectorstore = Chroma.from_documents(
#     docs,
#     embedding=llama_embeddings,
#     persist_directory="./vectorstore/dense"
# )

# # BM25 Retriever 생성 (Sparse 검색)
# bm25_retriever = BM25Retriever.from_documents(docs)
# bm25_retriever.k = 5 

# # Cross Encoder Reranker 생성 (두 검색 결과 결합 후 재정렬)
# top_n_reranked_docs = 3
# cross_encoder = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-v2-m3")
# reranker = CrossEncoderReranker(model=cross_encoder, top_n=top_n_reranked_docs)


# In[ ]:


# --------------------------------------------------
# 1. State 타입 정의 (질문, 생성된 답변, 검색 데이터 저장)
class State(TypedDict):
    question: str
    generation: str
    data: str

# StateGraph 객체 생성
workflow = StateGraph(State)

# --------------------------------------------------
# 2. 단순 답변 생성 (검색 없이 LLM 호출)
def answer(state: State) -> State:
    print("---단순 답변 생성---")
    question = state["question"]
    return {"question": question, "generation": llm.invoke(question).content, "data": ""}

# --------------------------------------------------
# 3. Hybrid 검색 및 Reranking
def rerank(query: str, _docs: List[Document]) -> List[Document]:
    # 입력 쿼리와 문서를 기반으로 reranker를 적용하여 재정렬합니다.
    reranked_docs = reranker.compress_documents(_docs, query)
    return reranked_docs

def retrieval(state: State):
    """
    Dense와 BM25 검색 결과를 모두 회수한 후 결합하고,
    이를 reranker를 통해 재정렬하여 최종 검색 데이터를 생성합니다.
    """
    print("---데이터 검색---")
    question = state["question"]
    
    # Dense 검색 결과: 상위 5개 문서를 (문서, 점수) 튜플로 회수 후 문서 부분만 추출
    dense_results_with_scores = dense_vectorstore.similarity_search_with_score(question, k=5)
    dense_docs = [doc for doc, _ in dense_results_with_scores]
    
    # BM25 검색 결과: 상위 5개 문서를 회수 (점수 정보 없음)
    bm25_results = bm25_retriever.invoke(question)
    
    # 두 검색 결과를 결합합니다.
    combined_docs = dense_docs + bm25_results
    
    # 결합된 문서를 reranker로 재정렬합니다.
    reranked_docs = rerank(question, combined_docs)
    return {"question": question, "data": reranked_docs}

# --------------------------------------------------
# 4. 검색된 데이터를 바탕으로 최종 답변 생성
def answer_with_retrieved_data(state: State):
    """
    hybrid 검색 및 reranking 결과(문서 목록)를 바탕으로 LLM을 통해 최종 답변을 생성합니다.
    """
    print("---검색된 데이터를 바탕으로 답변 생성---")
    question = state["question"]
    data = state["data"]
    # 프롬프트 템플릿 내 플레이스홀더 {context}와 {question}을 그대로 사용합니다.
    messages_with_context = [
        ("system", "사용자가 제공한 정보를 바탕으로 질문에 정확하게 답변하세요."),
        ("human", "정보:\n{context}\n\n질문: {question}")
    ]
    prompt_with_context = ChatPromptTemplate.from_messages(messages_with_context)
    qa_chain = prompt_with_context | llm | StrOutputParser()
    # invoke 시, 실제 값으로 {context}와 {question}을 전달합니다.
    generation = qa_chain.invoke({"context": data, "question": question})
    return {"question": question, "data": data, "generation": generation}

# --------------------------------------------------
# 5. 라우팅 체인 구성 (질문에 따라 'rag' 또는 'plain_answer' 경로 선택)
route_system_message = """당신은 사용자의 질문에 RAG 사용 여부를 결정하는 전문가입니다.
인공지능 산업 관련 질문이면 'rag'를, 그렇지 않으면 'plain_answer'를 선택하세요.
답변은 반드시 'route' 키만 있는 JSON 형식으로 생성해야 합니다."""
route_user_message = "{question}"
route_prompt = ChatPromptTemplate.from_messages(
    [("system", route_system_message), ("human", route_user_message)]
)
route_llm = ChatOllama(model="llama3.1", format="json", temperature=0)
router_chain = route_prompt | route_llm | JsonOutputParser()

def init_answer(state: State) -> State:
    "초기 질문의 경로를 결정합니다."
    question = state["question"]
    route = router_chain.invoke({"question": question})["route"]
    return {"question": question, "generation": route, "data": ""}

def route_question(state: State) -> str:
    # 라우터 체인 결과를 소문자화 및 좌우 공백 제거 후 반환합니다.
    return state["generation"].lower().strip()

# --------------------------------------------------
# 6. 그래프 구성 및 실행
workflow.add_node("init_answer", init_answer)
workflow.add_node("rag", retrieval)
workflow.add_node("plain_answer", answer)
workflow.add_node("answer_with_retrieval", answer_with_retrieved_data)

workflow.set_entry_point("init_answer")
workflow.add_edge("plain_answer", END)
workflow.add_edge("answer_with_retrieval", END)
workflow.add_edge("rag", "answer_with_retrieval")

workflow.add_conditional_edges(
    "init_answer",
    route_question,
    {"rag": "rag", "plain_answer": "plain_answer"}
)

# 그래프 컴파일 및 (주피터 노트북 환경에서) 구조 시각화
graph = workflow.compile()
# display(Image(graph.get_graph().draw_mermaid_png()))

# --------------------------------------------------
# 7. 챗봇 실행 루프
while True:
    question = input("질문을 입력해주세요 (종료를 원하시면 '종료' 입력): ")
    if question.strip() == "종료":
        break
    response = graph.invoke({"question": question})
    print("Assistant:", response["generation"])

