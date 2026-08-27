import os
from pathlib import Path
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

# 기존 방식: data_dir = "data"
BASE_DIR = Path(__file__).resolve().parent
data_dir = BASE_DIR / "data"
pdf_data_name = "RE177_2023년 국내외 인공지능 산업 동향 연구_2장.pdf"

# 기존 방식: loader = PyPDFLoader(os.path.join(data_dir, pdf_data_name))
loader = PyPDFLoader(data_dir / pdf_data_name)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
docs = text_splitter.split_documents(docs)

print(f"총 Document 개수: {len(docs)}")

print("=== Step 1: Dense Retrieval (LLama Embedding) ===")
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain.vectorstores import Chroma

# llama_embeddings = OllamaEmbeddings(model="llama3.1")
llama_embeddings = OllamaEmbeddings(model="nomic-embed-text")
start = time()
dense_vectorstore = Chroma.from_documents(
    docs, 
    embedding=llama_embeddings, 
    # 기존 방식: persist_directory="./vectorstore/dense"
    persist_directory=str(BASE_DIR / "vectorstore" / "dense")
)
end = time()
print(f"Embedding 소요 시간: {end - start:.5f} sec")
                              
dense_retriever = dense_vectorstore.as_retriever(search_kwargs={"k": 5})

query = "프런티어 모델 포럼에 대해 설명해"

start = time()

dense_results_with_scores = dense_vectorstore.similarity_search_with_score(query, k=5)

end = time()

print(f"Dense Retrieval running time: {end - start:.5f} sec\n")

print("Dense Retrieval 결과 (점수 포함):")

for i, (doc, score) in enumerate(dense_results_with_scores, start=1):
    print(f"[Dense] Rank {i} | Score: {score:.4f}")
    print("내용:", doc.page_content)
    print("-" * 50)

print("=" * 70)

print("=== Step 2: Sparse Retrieval (BM25) ===")
from langchain.retrievers import BM25Retriever

bm25_retriever = BM25Retriever.from_documents(docs)
bm25_retriever.k = 5               

start = time()

bm25_results = bm25_retriever.invoke(query)

end = time()

print(f"Sparse Retrieval running time: {end - start:.5f} sec\n")

print("BM25 Retrieval 결과:")
for i, doc in enumerate(bm25_results, start=1):
                                 
    print(f"[BM25] Rank {i}")
    print("내용:", doc.page_content)
    print("-" * 50)

print("=" * 70)

print("=== Step 3: Combined Retrieval & Reranking ===")

dense_docs = [doc for doc, _ in dense_results_with_scores]

combined_docs = dense_docs + bm25_results

from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

top_n = 3

cross_encoder = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-v2-m3")
reranker = CrossEncoderReranker(model=cross_encoder, top_n=top_n)

top_docs = reranker.compress_documents(combined_docs, query)
print("Reranker를 통한 상위 3개 문서:")
for i, doc in enumerate(top_docs, start=1):
    print(f"Rank {i}")
    print("내용:", doc.page_content)
    print("-" * 50)

print("=" * 70)

combined_docs

for combined_doc in combined_docs:
    text_pair = (query, combined_doc.page_content)

    scores = cross_encoder.score([text_pair])

    similarity_score = scores[0]

    print(f"문서 유사도 점수: {similarity_score:.4f}")

llm = ChatOllama(model="llama3.1")
route_llm = ChatOllama(model="llama3.1", format="json")

class State(TypedDict):
    question: str
    generation: str
    data: str

workflow = StateGraph(State)

def answer(state: State) -> State:
    print("---단순 답변 생성---")
    question = state["question"]
    return {"question": question, "generation": llm.invoke(question).content, "data": ""}

def rerank(query: str, _docs: List[Document]) -> List[Document]:
                                            
    reranked_docs = reranker.compress_documents(_docs, query)
    return reranked_docs

def retrieval(state: State):
    """
    Dense와 BM25 검색 결과를 모두 회수한 후 결합하고,
    이를 reranker를 통해 재정렬하여 최종 검색 데이터를 생성합니다.
    """
    print("---데이터 검색---")
    question = state["question"]

    dense_results_with_scores = dense_vectorstore.similarity_search_with_score(question, k=5)
    dense_docs = [doc for doc, _ in dense_results_with_scores]

    bm25_results = bm25_retriever.invoke(question)

    combined_docs = dense_docs + bm25_results

    reranked_docs = rerank(question, combined_docs)
    return {"question": question, "data": reranked_docs}

def answer_with_retrieved_data(state: State):
    """
    hybrid 검색 및 reranking 결과(문서 목록)를 바탕으로 LLM을 통해 최종 답변을 생성합니다.
    """
    print("---검색된 데이터를 바탕으로 답변 생성---")
    question = state["question"]
    data = state["data"]
                                                         
    messages_with_context = [
        ("system", "사용자가 제공한 정보를 바탕으로 질문에 정확하게 답변하세요."),
        ("human", "정보:\n{context}\n\n질문: {question}")
    ]
    prompt_with_context = ChatPromptTemplate.from_messages(messages_with_context)
    qa_chain = prompt_with_context | llm | StrOutputParser()
                                                    
    generation = qa_chain.invoke({"context": data, "question": question})
    return {"question": question, "data": data, "generation": generation}

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
                                         
    return state["generation"].lower().strip()

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

graph = workflow.compile()

while True:
    question = input("질문을 입력해주세요 (종료를 원하시면 '종료' 입력): ")
    if question.strip() == "종료":
        break
    response = graph.invoke({"question": question})
    print("Assistant:", response["generation"])
