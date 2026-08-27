import io
import os
from pathlib import Path
import time
from typing import List, Dict, Any, Tuple, Optional
import pandas as pd
# from IPython.display import Image, display
from langchain.document_loaders import PyPDFLoader
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_chroma import Chroma
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_core.documents import Document
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict

# get_ipython().system('ollama pull llama3.1')
# os.system("ollama pull llama3.1")
# os.system("ollama pull nomic-embed-text")

llm = ChatOllama(model="llama3.1")
route_llm = ChatOllama(model="llama3.1", format="json")

# 기존 방식: data_dir = "data"
BASE_DIR = Path(__file__).resolve().parent
data_dir = BASE_DIR / "data"

pdf_data_name = "RE177_2023년 국내외 인공지능 산업 동향 연구_2장.pdf"

# embeddings = OllamaEmbeddings(model="llama3.1")
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# 기존 방식: doc_path = os.path.join(data_dir, pdf_data_name)
doc_path = data_dir / pdf_data_name
loader = PyPDFLoader(doc_path)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=10)
docs = text_splitter.split_documents(docs)

len_docs = [len(doc.page_content) for doc in docs]
print(f"총 Document 개수: {len(docs)}")
print(f"Document 별 길이: {len_docs}")
print(f"Document 별 길이 평균: {sum(len_docs) / len(len_docs)}")

top_n_reranked_docs = 3
        
cross_encoder = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-v2-m3")

reranker = CrossEncoderReranker(model=cross_encoder, top_n=top_n_reranked_docs)

# 기존 IPython 방식:
# get_ipython().run_cell_magic('time', '', '# 저장된 VectorStore를 불러올 때는 아래 주석을 해제하여 사용합니다.\n# vectorstore = Chroma(\n#     embedding_function=embeddings,\n#     persist_directory="./vectorstore/chroma"\n# )\n\nvectorstore = Chroma.from_documents(\n    docs,\n    embedding=embeddings,\n    # 기존 방식: persist_directory="./vectorstore/chroma"\n    persist_directory=str(BASE_DIR / "vectorstore" / "chroma")\n)\n\n# Reranker를 통해 한번 더 정렬할 예정이므로, 탐색할 k값을 top_n의 5배로 설정합니다.\ndb_retriever = vectorstore.as_retriever(\n    search_kwargs={"k": top_n_reranked_docs * 5}\n)\n')
########################################################################################
start_time = time.perf_counter()

vectorstore = Chroma.from_documents(
    docs,
    embedding=embeddings,
    # 기존 방식: persist_directory="./vectorstore/chroma"
    persist_directory=str(BASE_DIR / "vectorstore" / "chroma"),
)

db_retriever = vectorstore.as_retriever(
    search_kwargs={"k": top_n_reranked_docs * 5}
)

elapsed_time = time.perf_counter() - start_time
print(f"VectorStore 생성 시간: {elapsed_time:.2f}초")
########################################################################################

class State(TypedDict):

    question: str
    generation: str
    data: str

workflow = StateGraph(State)

def answer(state: State) -> State:
                                
    print("---답변 생성---")                         
    question = state["question"]

    return {"question": question, "generation": llm.invoke(question).content}

def rerank(query: str, _docs: List[Document]) -> List[Document]:
    reranked_docs = reranker.compress_documents(_docs, query)
    return reranked_docs

def retrieval(state: State):
    """
    데이터 검색을 수행합니다.

    Args:
        state (dict): 현재 그래프 상태

    Returns:
        state (dict): 검색된 데이터를 포함한 새로운 State
    """
    print("---데이터 검색---")                         
    question = state["question"]

    retrieval_chain = (
        db_retriever
        | (lambda _retrieved_docs: rerank(question, _retrieved_docs))
    )
    data = retrieval_chain.invoke(question)

    return {"question": question, "data": data}

def answer_with_retrieved_data(state: State):
    """
    검색된 데이터를 바탕으로 답변을 생성합니다.

    Args:
        state (dict): 현재 그래프 상태

    Returns:
        state (dict): LLM의 답변을 포함한 새로운 State
    """

    print(
        "---검색된 데이터를 바탕으로 답변 생성---"
    )                         

    question = state["question"]
    data = state["data"]

    messages_with_contexts = [
        ("system", "사용자가 입력하는 정보를 바탕으로 질문에 답하세요."),
        ("human", "정보: {context}.\n{question}."),
    ]
    prompt_with_context = ChatPromptTemplate.from_messages(messages_with_contexts)

    qa_chain = prompt_with_context | llm | StrOutputParser()

    generation = qa_chain.invoke({"context": data, "question": question})
    return {"question": question, "data": data, "generation": generation}

route_system_message = """당신은 사용자의 질문에 RAG 사용 여부를 결정하는 전문가입니다.
인공지능 산업과 관련된 질문이라면 'rag'를 선택하고, 그렇지 않다면 'plain_answer'를 선택하세요.
답변은 'route' 키 하나만 있는 JSON으로 답변하고, 다른 텍스트나 설명을 생성하지 마세요."""
route_user_message = "{question}"
route_prompt = ChatPromptTemplate.from_messages(
    [("system", route_system_message), ("human", route_user_message)]
)

route_llm = ChatOllama(model="llama3.1", format="json", temperature=0)
router_chain = route_prompt | route_llm | JsonOutputParser()

print(router_chain.invoke({"question": "인공지능 산업 현황 및 전망에 대해 알려줘"}))
print(router_chain.invoke({"question": "오늘 저녁 뭐 먹을까?"}))

def init_answer(state: State) -> str:
    "초기 질문의 경로를 결정합니다."
    question = state["question"]
    route = router_chain.invoke({"question": question})["route"]
    return {"question": question, "generation": route}

def route_question(state: State) -> str:
    route = state["generation"]
    return route.lower().strip()

workflow.add_node("init_answer", init_answer)
workflow.add_node("rag", retrieval)

workflow.add_node("plain_answer", answer)
workflow.add_node("answer_with_retrieval", answer_with_retrieved_data)

workflow.set_entry_point("init_answer")

workflow.add_edge(
    "plain_answer", END
)                                               
workflow.add_edge("answer_with_retrieval", END)
workflow.add_edge("rag", "answer_with_retrieval")

workflow.add_conditional_edges(
    "init_answer",
    route_question,
                                                               
    {
        "rag": "rag",
        "plain_answer": "plain_answer",
    },
)

graph = workflow.compile()
# display(Image(graph.get_graph().draw_mermaid_png()))

# 일반 Python 방식
graph_image_path = BASE_DIR / "workflow_graph.png"
graph_image_data = graph.get_graph().draw_mermaid_png()
graph_image_path.write_bytes(graph_image_data)

print(f"워크플로 그래프 저장 완료: {graph_image_path}")

while True:
    question = input("질문을 입력해주세요 (종료를 원하시면 '종료'를 입력해주세요.): ")
    if question == "종료":
        break
    else:

        response = graph.invoke({"question": question})
        print("Assistant: ", response["generation"])
