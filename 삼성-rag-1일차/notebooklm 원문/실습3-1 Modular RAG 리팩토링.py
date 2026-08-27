#!/usr/bin/env python
# coding: utf-8

# # Modular RAG 리팩토링

# ## 실습 목표
# ---
# 기존에 구현한 Adaptive RAG 챗봇을 모듈러 시스템으로 리팩토링 하여 여러 컴포넌트를 상황에 따라 자유롭게 이어붙일 수 있는 구조로 개선합니다.

# ## 실습 목차
# ---
# 1. **환경 설정**: 실습에 필요한 라이브러리를 불러오고, 환경 변수, Reranker, 유틸리티 함수 등 RAG 로직을 구현하기 위한 환경을 구성합니다.
# 
# 2. **Adaptive RAG 리팩토링**: 앞서 구현한 Adaptive RAG의 각 Node와 조건부 간선을 모듈화 합니다.

# ## 1. 환경 설정
# 필요한 라이브러리를 불러옵니다.

# In[ ]:


import contextlib
import io
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT_DIR = next(parent for parent in Path(__file__).resolve().parents if (parent / "project_env.py").is_file())
sys.path.insert(0, str(ROOT_DIR))
from project_env import load_project_env, require_env

load_project_env()
require_env("TAVILY_API_KEY")

import pandas as pd

# Added imports for reranker
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_chroma import Chroma
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.documents import Document
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict


# Ollama를 통해 llama 3.1 8B 모델을 불러옵니다.

# In[ ]:


get_ipython().system('ollama pull llama3.1')


# In[ ]:


llm = ChatOllama(model="llama3.1", temperature=0)
route_llm = ChatOllama(model="llama3.1", format="json", temperature=0)
embeddings = OllamaEmbeddings(model="llama3.1")


# Graph State를 정의합니다.

# In[ ]:


class State(TypedDict):
    # 그래프 상태의 속성을 정의합니다.
    # 질문, LLM이 생성한 텍스트, 데이터, 코드를 저장합니다.
    question: str
    generation: str
    data: str
    code: str


# ### 1.1 웹 검색 API 초기화

# 웹 검색을 자동화 하기 위해 Tavily Search API Key를 등록합니다.
# 
# 1. 먼저, 아래 링크에 접속한 후 'Sign-in' 버튼을 눌러 로그인 화면으로 이동합니다.
#    - https://app.tavily.com/sign-in
# 2. 여러분의 Tavily 계정으로 로그인한 후, 아래 링크에 접속하여 default API Key를 복사하여 아래 코드에 적용합니다.
#    - https://app.tavily.com/home

# In[ ]:


# Tavily API key는 tvly- 로 시작하는 문자열입니다.
# API Key를 입력했다면, 이 셀을 실행해서 API Key를 환경 변수에 등록합니다.


# ### 1.2 RAG 챗봇에 사용할 데이터 로드
# RAG 챗봇의 데이터베이스에 등록할 PDF 문서와 엑셀 데이터를 불러옵니다.

# In[ ]:


excel_data_name = "한국지능정보사회진흥원_인공지능 학습용 데이터 구축 현황_20210104.csv"
pdf_data_name = "RE177_2023년 국내외 인공지능 산업 동향 연구_2장.pdf"

# 데이터를 불러오고, 이름과 컬럼명을 저장합니다.
data_dir = "./data"
df_ai_train_data_dist = pd.read_csv(
    os.path.join(data_dir, excel_data_name), index_col=None
)

# 데이터를 저장한 변수명을 LLM에 제공하여 이 변수를 활용하는 코드를 작성하게 할 수 있습니다.
df_name = "df_ai_train_data_dist"
df_columns = ", ".join(df_ai_train_data_dist.columns)


# ### 1.3 Reranking 모듈 초기화

# In[ ]:


# Initialize reranker
top_n_reranked_docs = 3
cross_encoder = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-v2-m3")
reranker = CrossEncoderReranker(model=cross_encoder, top_n=top_n_reranked_docs)

vectorstore = Chroma(
    embedding_function=embeddings, persist_directory="./vectorstore/chroma"
)

db_retriever = vectorstore.as_retriever()


# ### 1.4 유틸리티성 함수 정의

# In[ ]:


# LLM이 생성한 코드를 파싱하는 함수를 정의합니다.
def python_code_parser(input: str) -> str:
    # LLM은 대부분 ``` 블럭 안에 코드를 출력합니다. 이를 활용합니다.
    # ```python (코드) ```, 혹은 ``` (코드) ``` 형태로 출력됩니다. 두 경우 모두에 대응하도록 코드를 작성합니다.
    processed_input = input.replace("```python", "```").strip()
    parsed_input_list = processed_input.split("```")

    # 만약 ``` 블럭이 없다면, 입력 텍스트 전체가 코드라고 간주합니다.
    # 아닐 경우 이어지는 코드 실행 과정에서 예외 처리를 통해 오류를 확인할 수 있습니다.
    if len(parsed_input_list) == 1:
        return processed_input

    # 코드 부분만 추출합니다.
    # LLM은 여러 코드 블럭에 걸쳐 필요한 코드를 출력할 수 있으므로, 코드가 있는 홀수 번째 텍스트를 모두 저장합니다.
    parsed_code_list = []
    for i in range(1, len(parsed_input_list), 2):
        parsed_code_list.append(parsed_input_list[i])

    # 코드 부분을 하나로 합칩니다.
    return "\n".join(parsed_code_list)


# 생성한 코드를 실행하는 함수를 정의합니다.
def run_code(input_code: str):
    output = io.StringIO()
    try:
        with contextlib.redirect_stdout(output):
            # 코드가 실행하면서 출력한 모든 결과를 캡쳐합니다.
            exec(input_code, {"df_ai_train_data_dist": df_ai_train_data_dist})
    except Exception as e:
        # 에러가 발생할 경우, 이를 StringIO 객체에 저장합니다.
        print(f"Error: {e}", file=output)
    # StringIO 객체에 저장된 값을 반환합니다.
    return output.getvalue()


# ## 2. Adaptive RAG 리팩토링
# ---
# 앞서 구현한 Adaptive RAG 챗봇의 기능은 유지한 채, 각 기능을 모듈화 하여 유지보수가 용이하도록 리팩토링 합니다.

# ### 2.1. Node에 대응하는 모듈 구성
# 
# 단일 기능을 나타내는 Node 단위로 모듈을 구성합니다.

# In[ ]:


# LangGraph 기반 RAG 시스템을 구성하기 위해, 각 모듈이 상속해야 할 Base 메서드를 정의합니다.
class BaseRAGModule:
    # 각 모듈은 LangGraph Workflow의 노드 역할을 합니다.
    node_id: str

    # 각 모듈은 다음 노드로 전달할 데이터를 반환하는 `run` 메서드를 가져야 합니다.
    def run(self, state: State) -> State:
        raise NotImplementedError

    def __call__(self, state: State) -> State:
        return self.run(state)


# 챗봇의 각 기능을 모듈로 나누어 구현합니다.

# In[ ]:


# 챗봇의 각 기능을 모듈로 나누어 구현합니다.
class VectorDBRetrieverRAGModule(BaseRAGModule):
    node_id = "vector_db_retriever"

    def __init__(self, db_retriever, reranker):
        self.db_retriever = db_retriever
        self.reranker = reranker

    def _rerank(self, query: str, docs: List[Document]) -> List[Document]:
        return self.reranker.compress_documents(docs, query)

    def run(self, state: State) -> State:
        def get_retrieved_text(docs):
            return "\n".join([doc.page_content for doc in docs])

        print("---벡터 DB 검색---")
        question = state["question"]

        retrieval_chain = (
            self.db_retriever
            | (lambda docs: self._rerank(question, docs))
            | get_retrieved_text
        )
        state["data"] = retrieval_chain.invoke(question)
        return state


# In[ ]:


class WebRetrieverRAGModule(BaseRAGModule):
    node_id = "web_retriever"

    def run(self, state: State) -> State:
        print("---웹 검색---")
        tavily_search_tool = TavilySearchResults(max_results=1)
        results = tavily_search_tool.invoke({"query": state["question"]})
        contents_list = [r["content"] for r in results]
        state["data"] = contents_list
        return state


# In[ ]:


class DataRetreiverRAGModule(BaseRAGModule):
    node_id = "data_retriever"

    def __init__(self):
        self.df_name = df_name
        self.df_columns = df_columns

    # 데이터를 쿼리하는 코드를 생성하고, 실행하고, 그 결과를 포함한 State를 반환합니다.
    def run(self, state: State) -> State:
        # 데이터를 쿼리하는 코드를 생성하고, 실행하고, 그 결과를 포함한 State를 반환합니다.

        print("---데이터 쿼리---")  # 현재 상태를 확인하기 위한 Print문
        question = state["question"]


        # Retrieval
        system_message = """당신은 주어진 데이터를 분석하는 데이터 분석가입니다.
        주어진 DataFrame에서 데이터를 출력하여 주어진 질문에 답할 수 있는 파이썬 코드를 작성하세요.
        {df_name} DataFrame에 액세스할 수 있습니다.
        `{df_name}` DataFrame에는 다음과 같은 열이 있습니다: {df_columns}
        데이터는 이미 로드되어 있으므로 데이터 로드 코드를 생략해야 합니다."""

        message_with_data_info = [
            ("system", system_message),
            ("human", "{question}"),
        ]

        prompt_with_data_info = ChatPromptTemplate.from_messages(message_with_data_info)

        # 체인을 구성합니다.
        code_generate_chain = (
            prompt_with_data_info
            | llm
            | StrOutputParser()
            | python_code_parser
        )
        code = code_generate_chain.invoke({"question": question, "df_name": self.df_name, "df_columns": self.df_columns})
        data = run_code(code)
        return {"question": question, "code": code, "data": data, "generation": code}


# In[ ]:


class LLMAnswerRAGModule(BaseRAGModule):
    node_id = "llm_answer"

    def run(self, state: State) -> State:
        # Merge logic for data-based answer, retrieved-data answer, and plain answer generation.
        question = state["question"]
        data = state.get("data", "")

        # If there is data, generate answers using two different prompts and merge their result.
        if data:
            print("---데이터 기반 답변 생성 (통합 로직)---")
            # 첫 번째 데이터 기반 체인: answer_with_data 방식
            reasoning_with_data = [
                (
                    "system",
                    "당신은 데이터를 바탕으로 질문에 답하는 데이터 분석가입니다. 사용자가 입력한 데이터를 바탕으로, 질문에 대답하세요.",
                ),
                ("human", "데이터: {data}\n{question}"),
            ]
            chain_data = (
                ChatPromptTemplate.from_messages(reasoning_with_data)
                | llm
                | StrOutputParser()
            )
            generation = chain_data.invoke({"data": data, "question": question})

        else:
            # 데이터가 없는 경우 단순히 질문만으로 답변 생성 (plain answer)
            print("---답변 생성 (데이터 없음)---")
            generation = llm.invoke(question).content

        return {
            "question": question,
            "code": state.get("code", ""),
            "data": data,
            "generation": generation,
        }


# ### 2.2 조건부 간선 모듈 구성
# 

# In[ ]:


# 조건부 간선을 구현하기 위한 클래스를 정의합니다.
class BaseConditionalEdgeModule:
    edge_id: str

    def run(self, state: State) -> str:
        raise NotImplementedError


# In[ ]:


class SelfReflectionRAGModule(BaseConditionalEdgeModule):
    edge_id = "self_reflection"

    def judge_answer(self, state: State) -> str:
        print("---답변 퀄리티 검증 (Self-RAG)---")
        try:
            hallucinated = self._is_hallucinated(state)
        except KeyError:
            hallucinated = False
            print("---주어진 답변의 진실 여부를 판단할 수 없습니다.---")
        else:
            status = "진실이 아닙니다" if hallucinated else "진실입니다"
            print(f"---주어진 답변은 {status}.---")

        try:
            supportive = self._is_answer_supportive(state)
        except KeyError:
            supportive = True
            print("---주어진 답변의 지원 여부를 판단할 수 없습니다.---")
        else:
            status = "지원적입니다" if supportive else "지원적이지 않습니다"
            print(f"---주어진 답변은 {status}.---")

        try:
            useful = self._is_answer_useful(state)
        except KeyError:
            useful = True
            print("---주어진 답변의 유용성 여부를 판단할 수 없습니다.---")
        else:
            status = "유용합니다" if useful else "유용하지 않습니다"
            print(f"---주어진 답변은 {status}.---")

        if (supportive or useful ) and hallucinated is False:
            return "yes"
        else:
            return "no"

    # 지지성 평가
    def _is_answer_supportive(self, state: State) -> bool:
        # 생성된 텍스트가 질문과 관련이 있는지 확인합니다.
        question = state["question"]
        generation = state["generation"]
        system_message = """당신은 AI의 답변이 사용자의 질문에 대한 해답인지 평가하는 평가자입니다."""
        # 다른 모듈과 달리, 정보와 AI의 답변을 모두 사용자 프롬프트에 추가합니다. 이는 실험을 통해 더 좋은 결과가 나와서 선택한 방법입니다.
        # 또한 함수 이름과 달리 'supportive'가 아니라 'answer' Key에 답변을 저장하라는 지시가 있습니다.
        # 이는 다른 지시사항에 있는 텍스트와 일관성을 유지하기 위함입니다.
        user_message = """사용자의 질문: {question}
        AI의 답변: {generation}
        AI의 답변이 사용자의 질문에 대한 해답이면 'yes', 아니라면 'no'를 선택하세요.
        'answer' key 하나만 있는 JSON 형식으로 답변하고, 다른 텍스트나 설명을 생성하지 마세요."""

        message_list = [("system", system_message)]
        message_list.append(("human", user_message))

        relevant_judge_prompt = ChatPromptTemplate.from_messages(message_list)
        # 로직 선택용 ChatOllama 객체를 생성합니다. format="json" 인자를 적용하여 출력 양식을 json으로 강제합니다.
        # 같은 질문에 항상 같은 대답을 유도하기 위해 temperature를 0으로 설정합니다.
        router_chain = relevant_judge_prompt | route_llm | JsonOutputParser()

        result = router_chain.invoke({"question": question, "generation": generation})
        if result["answer"] == "yes":
            return True
        return False

    # 유용성 평가
    def _is_answer_useful(self, state: State) -> bool:
        # 생성된 텍스트가 질문에 대한 해답인지 확인합니다.
        question = state["question"]
        generation = state["generation"]
        system_message = (
            """당신은 AI의 답변이 사용자에게 유용한지 평가하는 평가자입니다."""
        )
        user_message = """사용자의 질문: {question}
        AI의 답변: {generation}
        AI의 답변이 사용자에게 유용하다면 'yes', 아니라면 'no'를 선택하세요.
        'useful' key 하나만 있는 JSON 형식으로 답변하고, 다른 텍스트나 설명을 생성하지 마세요."""

        message_list = [("system", system_message)]
        message_list.append(("human", user_message))

        useful_judge_prompt = ChatPromptTemplate.from_messages(message_list)
        # 로직 선택용 ChatOllama 객체를 생성합니다. format="json" 인자를 적용하여 출력 양식을 json으로 강제합니다.
        # 같은 질문에 항상 같은 대답을 유도하기 위해 temperature를 0으로 설정합니다.
        router_chain = useful_judge_prompt | route_llm | JsonOutputParser()

        result = router_chain.invoke({"question": question, "generation": generation})
        if result["useful"] == "yes":
            return True

    # 할루시네이션 평가
    def _is_hallucinated(self, state: State) -> bool:
        # 생성된 텍스트가 질문에 대한 해답인지 확인합니다.
        generation = state["generation"]
        docs = state["data"]
        system_message = """당신은 주어진 근거 문서를 바탕으로 AI의 답변이 진실인지 여부를 평가하는 평가자입니다."""
        user_message = """근거 문서: {docs}
        AI의 답변: {generation}
        근거 문서를 바탕으로 AI의 답변이 진실이라면 True, 아니라면 False를 선택하세요.
        'answer` key 하나만 있는 JSON 형식으로 답변하고, 다른 텍스트나 설명을 생성하지 마세요."""

        message_list = [("system", system_message)]
        message_list.append(("human", user_message))

        hallucination_judge_prompt = ChatPromptTemplate.from_messages(message_list)
        # 로직 선택용 ChatOllama 객체를 생성합니다. format="json" 인자를 적용하여 출력 양식을 json으로 강제합니다.
        # 같은 질문에 항상 같은 대답을 유도하기 위해 temperature를 0으로 설정합니다.
        router_chain = hallucination_judge_prompt | route_llm | JsonOutputParser()

        result = router_chain.invoke({"generation": generation, "docs": docs})
        print(result)
        if result["answer"] == "True":
            return True
        return False


# ### 2.3 Adaptive RAG 구현
# ---
# Self-RAG 챗봇에 웹 검색 기능을 추가하여 Adaptive RAG 챗봇을 구현하고 사용합니다.

# In[ ]:


class ModularRAG:
    def __init__(self, llm, route_llm, embeddings):
        # Initialize LLMs, embeddings, retriever and vectorstore from globals
        self.llm = llm
        self.route_llm = route_llm
        self.embeddings = embeddings
        # Vectorstore와 retriever는 submodule에서 각각 초기화 하므로, 여기서 저장하지 않습니다.

        # Initialize submodules once
        self.data_retriever = DataRetreiverRAGModule()
        self.web_retriever = WebRetrieverRAGModule()
        self.llm_answer = LLMAnswerRAGModule()
        self.self_reflection = SelfReflectionRAGModule()

        # Build workflow graph
        self.workflow = StateGraph(State)
        # Add nodes: our modules and a lambda node for plain answer generation
        self.workflow.add_node("data_retriever", self.data_retriever)
        self.workflow.add_node("web_retriever", self.web_retriever)
        self.workflow.add_node("llm_answer", self.llm_answer)
        self.workflow.add_node(
            "plain_answer",
            lambda state: {
                **state,
                "generation": self.llm.invoke(state["question"]).content,
            },
        )
        # Set entry point to start with data retrieval
        self.workflow.set_entry_point("data_retriever")
        # Conditional edge after data retrieval: if data is relevant then go to llm_answer, else try web retrieval
        self.workflow.add_conditional_edges(
            "data_retriever",
            lambda state: "yes" if self._is_data_relevant(state) else "no",
            {"yes": "llm_answer", "no": "web_retriever"},
        )
        # Conditional edge after web retrieval: if relevance is confirmed, go to llm_answer; otherwise use plain answer.
        self.workflow.add_conditional_edges(
            "web_retriever",
            lambda state: "yes" if self._is_data_relevant(state) else "no",
            {"yes": "llm_answer", "no": "plain_answer"},
        )
        # Both answer nodes lead to termination.
        self.workflow.add_edge("llm_answer", END)
        self.workflow.add_edge("plain_answer", END)
        
        self.workflow = self.workflow.compile()
    # 정보 평가 (reuse legacy logic)
    def _is_data_relevant(self, state: State) -> bool:
        # LLM이 생성한 텍스트가 문서와 관련이 있는지 확인합니다.
        print("--- is_data_relevant ---")
        question = state["question"]
        data = state["data"]
        system_message = """당신은 검색된 문서와 사용자의 질문의 연관성을 평가하는 평가자입니다.
다음은 검색된 문서입니다: \n{data}\n.
문서와 사용자가 입력할 질문이 연관이 있다면 `yes`, 그렇지 않다면 `no`를 선택하세요.
답변은 'relevant' key 하나만 있는 JSON으로 답변하고, 다른 텍스트나 설명을 생성하지 마세요."""
        message_list = [("system", system_message)]
        message_list.append(("human", "{question}"))
        relevant_judge_prompt = ChatPromptTemplate.from_messages(message_list)
        router_chain = relevant_judge_prompt | self.route_llm | JsonOutputParser()
        result = router_chain.invoke({"question": question, "data": data})
        print(result)
        return result.get("relevant", "no") == "yes"

    def run(self, question: str) -> str:
        # Create the initial state and invoke the workflow
        init_state = {"question": question, "data": "", "code": "", "generation": ""}
        final_state = self.workflow.invoke(init_state)
        # Evaluate answer quality using the SelfReflection module
        quality = self.self_reflection.judge_answer(final_state)
        if quality != "yes":
            print(
                "---Answer did not pass self-reflection. Falling back to plain answer.---"
            )
            final_state["generation"] = self.llm.invoke(question).content
        return final_state["generation"]


# In[ ]:


# Instantiate using global objects defined above
modular_rag = ModularRAG(llm, route_llm, embeddings)


# 모델 구성을 끝마쳤으면, 한번 사용해 봅시다.

# - 예시 질문 (문서 데이터 활용): 인공지능 산업 현황 및 전망에 대해 알려줘
# - 예시 질문 (엑셀 데이터 활용): 인공지능 학습용 데이터 구축 현황을 알려줘
# - 예시 질문 (인터넷 검색): OpenAI o3-mini 모델을 설명해줘
# - 예시 질문 (데이터 무관): 저녁 메뉴 추천해줘

# In[ ]:


# Note. LLM 특성상 의도하지 않은 방식으로 작동하지 않을 가능성이 있습니다.
while True:
    question = input(
        "질문을 입력해주세요 (종료를 원하시면 '종료'를 입력해주세요.): "
    )
    if question == "종료":
        break
    answer = modular_rag.run(question)
    print("Assistant:", answer)
