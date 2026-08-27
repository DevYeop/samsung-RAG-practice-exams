#!/usr/bin/env python
# coding: utf-8

# # Modular RAG 리팩토링 - LlamaIndex

# ## 실습 목표
# ---
# LlamaIndex Workflow의 각 Step의 Logic을 별도의 Class로 분리하는 리팩토링을 통해 유지보수 및 확장성을 높이는 방법을 학습합니다.

# ## 실습 목차
# ---
# 1. **환경 설정**: 실습에 필요한 라이브러리를 불러오고, 환경 변수, Reranker, 유틸리티 함수 등 RAG 로직을 구현하기 위한 환경을 구성합니다.
# 
# 2. **Adaptive RAG 리팩토링**: 앞서 구현한 Adaptive RAG의 각 Node와 조건부 간선을 모듈화 합니다.

# ## 1. 환경 설정
# 필요한 라이브러리를 불러옵니다.

# In[ ]:


import asyncio
import contextlib
import io
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT_DIR = next(parent for parent in Path(__file__).resolve().parents if (parent / "project_env.py").is_file())
sys.path.insert(0, str(ROOT_DIR))
from project_env import load_project_env, require_env

load_project_env()

import pandas as pd
from llama_index.core import (
    ChatPromptTemplate,
    Settings,
    SimpleDirectoryReader,
    VectorStoreIndex,
    get_response_synthesizer,
)
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.postprocessor.llm_rerank import LLMRerank
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.schema import NodeWithScore
from llama_index.core.selectors import LLMSingleSelector
from llama_index.core.tools import ToolMetadata
from llama_index.core.workflow import Event, StartEvent, StopEvent, Workflow, step
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.tools.tavily_research import TavilyToolSpec
from llama_index.utils.workflow import draw_all_possible_flows
from pydantic import BaseModel, Field


# Ollama를 통해 llama 3.1 8B 모델을 불러옵니다.

# In[ ]:


get_ipython().system('ollama pull llama3.1')


# In[ ]:


# Configure logging
logging.basicConfig(level=logging.INFO)

Settings.llm = Ollama(model="llama3.1")
Settings.embedding = OllamaEmbedding(model_name="llama3.1")


# Workflow 구성에 활용할 Pydantic 데이터 구조를 정의합니다.

# In[ ]:


class Route(BaseModel):
    """질문에 대해 사용할 도구의 이름을 저장하는 데이터 구조"""

    route_name: str


class Answer(BaseModel):
    """LLM의 답변을 저장하는 데이터 구조"""

    answer: str


# ### 1.1 웹 검색 API 초기화
# 
# 웹 검색을 자동화 하기 위해 Tavily Search API Key를 등록합니다.
# 
# 1. 먼저, 아래 링크에 접속한 후 'Sign-in' 버튼을 눌러 로그인 화면으로 이동합니다.
#    - https://app.tavily.com/sign-in
# 2. 여러분의 Tavily 계정으로 로그인한 후, 아래 링크에 접속하여 default API Key를 복사하여 아래 코드에 적용합니다.
#    - https://app.tavily.com/home

# In[ ]:


# Tavily API key는 tvly- 로 시작하는 문자열입니다.
# API Key를 입력했다면, 이 셀을 실행해서 API Key를  등록합니다.
tvly_api_key = require_env("TAVILY_API_KEY")


tavily_tool = TavilyToolSpec(
    api_key=tvly_api_key,
)


# 검색이 잘 되는지 확인합니다.

# In[ ]:


tavily_tool.search("인공지능 산업 동향", max_results=3)


# ### 1.2 RAG 챗봇에 사용할 데이터 로드
# RAG 챗봇의 데이터베이스에 등록할 PDF 문서와 엑셀 데이터를 불러옵니다.

# In[ ]:


# Configuration constants
SIMILARITY_CUTOFF = 0.3

excel_data_name = "한국지능정보사회진흥원_인공지능 학습용 데이터 구축 현황_20210104.csv"
pdf_data_name = "RE177_2023년 국내외 인공지능 산업 동향 연구_2장.pdf"


# 데이터를 불러오고, 이름과 컬럼명을 저장합니다.
data_dir = "./data"
documents = SimpleDirectoryReader(data_dir).load_data()

df_ai_train_data_dist = pd.read_csv(
    os.path.join(data_dir, excel_data_name), index_col=None
)
# embed_model에 저희가 생성한 OllamaEmbedding을 넣어줍니다.
vector_index = VectorStoreIndex.from_documents(
    documents, embed_model=Settings.embedding
)

# `as_query_engine` 메서드를 통해 한번에 Query Engine을 제작하지 않고, 세부사항을 조절하기 위해 여러 단계로 나누어서 진행합니다.

# 1. Retriever를 생성하면서 상황에 맞게 조정합니다.
# Top k 인자를 5로 설정해서 한번에 유사도가 높은 최대 5개의 문서를 검색하도록 설정합니다.
retriever = VectorIndexRetriever(index=vector_index, similarity_top_k=5)

# 2. 답변을 생성하는 객체를 생성합니다.
response_synthesizer = get_response_synthesizer()

# 3. 유사도 점수 기준 후처리 객체를 생성합니다.
# 유사도 0.5 이상인 문서만을 반환하도록 설정합니다.
postprocessor = SimilarityPostprocessor(similarity_cutoff=SIMILARITY_CUTOFF)

# 3. Retriever와 답변 생성 객체를 활용해 Query Engine을 구성합니다.
query_engine_tuned = RetrieverQueryEngine(
    retriever=retriever,
    response_synthesizer=response_synthesizer,
    node_postprocessors=[postprocessor],
)

Settings.query_engine = query_engine_tuned


# ### 1.3 유틸리티성 함수 정의

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
        logging.error(f"Error in running code: {e}")
        print(f"Error: {e}", file=output)
    # StringIO 객체에 저장된 값을 반환합니다.
    return output.getvalue()


# ## 2. Adaptive RAG 리팩토링
# ---
# Self-RAG 챗봇에 웹 검색 기능을 추가하여 Adaptive RAG 챗봇을 구현하고 사용합니다.

# Adaptive RAG 흐름에서 생기는 이벤트 목록은 다음과 같습니다:
# - StartEvent: RAG 흐름을 시작합니다. 사용자의 입력을 받고, 질문을 판단한 다음 적합한 루트로 이동합니다.
# - StopEvent: RAG 흐름이 종료됩니다.
#   
# - PlainAnswerReqEvent: 단순 LLM 답변을 요청합니다. 
# - ExcelDataReqEvent: 엑셀 데이터 기반 답변을 요청합니다.
# - WebSearchReqEvent: 웹 검색 기반 답변을 요청합니다.
# - RetrievalReqEvent: 내부 데이터베이스 검색 기반 답변을 요청합니다.
# - RetrievalDoneEvent: 검색 결과가 생성되었습니다. 
# 여기서 데이터베이스는 Event Handler에서 관리하므로 Event에 포함할 필요는 없습니다.
# 
# 이전에 구현한 Adaptive RAG와 동일하게 `Event Class`를 정의합니다.

# In[ ]:


# Adaptive RAG 흐름에서 생기는 이벤트 목록은 다음과 같습니다:
# - PlainAnswerReqEvent: 단순 LLM 답변을 요청합니다.
# - ExcelDataReqEvent: 엑셀 데이터 기반 답변을 요청합니다.
# - WebSearchReqEvent: 웹 검색 기반 답변을 요청합니다.
# - RetrievalReqEvent: 내부 데이터베이스 검색 기반 답변을 요청합니다.
# - RetrievalDoneEvent: 내부 데이터베이스 검색 결과가 생성되었습니다.
class PlainAnswerReqEvent(Event):
    question: str


class ExcelDataReqEvent(Event):
    question: str


class WebSearchReqEvent(Event):
    question: str


class RetrievalReqEvent(Event):
    question: str


class RetrievalDoneEvent(Event):
    question: str
    docs: List[NodeWithScore]
    answer: str


# ### 2.1 Step에 대응하는 모듈 Class 구성
# 
# 용이한 유지보수를 위해, 주요 Step에 대응하는 모듈을 Workflow Class 외부에 정의합니다.

# In[ ]:


# Note. Staticmethod 대신 Class Object를 Workflow에 저장하여 구현하는 것도 가능합니다.
class PlainAnswerStep:
    @staticmethod
    async def execute(answer_llm, question: str) -> StopEvent:
        messages = [
            ("system", "당신은 주어진 질문에 답하는 전문가입니다. 질문에 대답하세요."),
            ("user", question),
        ]
        prompt = ChatPromptTemplate.from_messages(messages).message_templates
        answer = await answer_llm.achat(messages=prompt)
        return StopEvent(result=answer.raw.answer)


# In[ ]:


class ExcelDataStep:
    @staticmethod
    async def execute(
        llm,
        answer_llm,
        df_name: str,
        df_columns: str,
        question: str,
        python_code_parser,
        run_code,
    ) -> StopEvent:
        code_gen_msg = (
            "당신은 엑셀 데이터를 바탕으로 질문에 답하는 데이터 분석가입니다.\n"
            f"{df_name} DataFrame에 액세스할 수 있습니다.\n"
            f"DataFrame의 열: {df_columns}\n"
            "파일을 불러오는 코드는 생략하세요."
        )
        prompt_messages = [("system", code_gen_msg), ("user", question)]
        prompt = ChatPromptTemplate.from_messages(prompt_messages).message_templates
        raw_code_resp = await llm.achat(prompt)
        raw_code = raw_code_resp.raw["message"]["content"]
        code = python_code_parser(raw_code)
        # Execute generated code
        data = run_code(code)
        if data.startswith("Error"):
            return StopEvent(result=data)
        # Generate answer based on data
        reasoning_msg = "당신은 엑셀 데이터를 바탕으로 질문에 답하는 데이터 분석가입니다. 데이터를 참고하여 답변하세요."
        reasoning_messages = [
            ("system", reasoning_msg),
            ("user", f"데이터: {data}\n{question}"),
        ]
        reasoning_prompt = ChatPromptTemplate.from_messages(
            reasoning_messages
        ).message_templates
        answer_resp = await answer_llm.achat(reasoning_prompt)
        return StopEvent(result=answer_resp.raw.answer)


# In[ ]:


class RetrievalDoneStep:
    """Step for processing the retrieval done event."""

    @staticmethod
    async def execute(event, llm, evaluate_prompt) -> StopEvent | PlainAnswerReqEvent:
        question = event.question
        raw_docs = event.docs

        # Rerank answer using LLMRerank
        reranker = LLMRerank(top_n=3, llm=llm)
        reranked_docs = reranker.postprocess_nodes(raw_docs, query_str=event.question)
        docs = [doc.text for doc in reranked_docs]
        docs_text = "\n".join(docs)
        answer = event.answer

        # Use helper for evaluations
        hallucination_messages = [
            (
                "system",
                "당신은 주어진 근거 문서를 바탕으로 AI의 답변이 진실인지 여부를 평가하는 평가자입니다.",
            ),
            (
                "user",
                f"근거 문서: {docs_text} AI의 답변: {answer} 근거 문서를 바탕으로 AI의 답변이 진실이라면 'yes', 아니라면 'no'를 선택하세요.",
            ),
        ]
        non_hallucinated = await evaluate_prompt(hallucination_messages)

        supportive_messages = [
            (
                "system",
                "당신은 주어진 근거 문서를 바탕으로 AI의 답변이 지원적인지 여부를 평가하는 평가자입니다.",
            ),
            (
                "user",
                f"근거 문서: {docs_text} AI의 답변: {answer} 근거 문서를 바탕으로 AI의 답변이 지원적이라면 'yes', 아니라면 'no'를 선택하세요.",
            ),
        ]
        supportive = await evaluate_prompt(supportive_messages)

        useful_messages = [
            (
                "system",
                "당신은 주어진 근거 문서를 바탕으로 AI의 답변이 유용한지 여부를 평가하는 평가자입니다.",
            ),
            (
                "user",
                f"근거 문서: {docs_text} AI의 답변: {answer} 근거 문서를 바탕으로 AI의 답변이 유용하다면 'yes', 아니라면 'no'를 선택하세요.",
            ),
        ]
        useful = await evaluate_prompt(useful_messages)

        if non_hallucinated == "no" or (supportive == "no" and useful == "no"):
            return PlainAnswerReqEvent(question=question)
        return StopEvent(result=answer)


# Event를 다 정의했으면, 해당 Event를 처리할 수 있는 Workflow를 정의합니다.

# In[ ]:


# Note. LlamaIndex는 Workflow를 async로 정의하는 것을 권장합니다.
class AdaptiveRAGFlow(Workflow):
    llm = Settings.llm
    embedding = Settings.embedding
    route_llm = llm.as_structured_llm(output_cls=Route)
    answer_llm = llm.as_structured_llm(output_cls=Answer)
    df_name = "df_ai_train_data_dist"
    df_columns = ", ".join(df_ai_train_data_dist.columns)
    tavily_tool = tavily_tool

    @step
    async def start(
        self, event: StartEvent
    ) -> (
        PlainAnswerReqEvent | ExcelDataReqEvent | WebSearchReqEvent | RetrievalReqEvent
    ):
        # RAG Workflow를 시작합니다. 사용자의 질문을 받아서 어떤 방식으로 답변을 생성할지 결정합니다.
        question = event.question

        # 앞서 살펴본 라우팅 프롬프트가 아니라, 정보를 바탕으로 LlamaIndex의 도구를 선택하는 기능을 활용합니다.
        choices = [
            ToolMetadata(
                description="인공지능 산업 동향과 관련된 질문에 사용할 도구", name="rag"
            ),
            ToolMetadata(
                description="인공지능 데이터 프로필과 관련된 질문에 사용할 도구",
                name="excel_data",
            ),
            ToolMetadata(
                description="웹 검색을 통해 정보를 수집해야 할 질문에 사용할 도구",
                name="web_search",
            ),
            ToolMetadata(
                description="인공지능 산업 동향, 데이터 프로필과 관련 없으며 정보를 수집할 필요가 없는 질문에 사용할 도구",
                name="plain_answer",
            ),
        ]
        selector = LLMSingleSelector.from_defaults()
        selector_result = selector.select(choices, question)

        route_index = selector_result.selections[0].index
        route_ans = choices[route_index].name

        if route_ans == "rag" or route_ans == "RAG":
            return RetrievalReqEvent(question=question)
        elif route_ans == "excel_data":
            return ExcelDataReqEvent(question=question)
        elif route_ans == "web_search":
            return WebSearchReqEvent(question=question)
        else:
            return PlainAnswerReqEvent(question=question)

    @step
    async def plain_answer_req(self, event: PlainAnswerReqEvent) -> StopEvent:
        # 외부에 정의한 모듈을 불러오도록 수정했습니다.
        question = event.question
        return await PlainAnswerStep.execute(self.answer_llm, question)

    @step
    async def excel_data_req(self, event: ExcelDataReqEvent) -> StopEvent:
        # 외부에 정의한 모듈을 불러오도록 수정했습니다.
        question = event.question
        return await ExcelDataStep.execute(
            self.llm,
            self.answer_llm,
            self.df_name,
            self.df_columns,
            question,
            python_code_parser,  # existing helper
            run_code,  # existing helper
        )

    @step
    async def web_search_req(
        self, event: WebSearchReqEvent
    ) -> RetrievalDoneEvent | PlainAnswerReqEvent:
        question = event.question
        # 웹 검색 결과를 받아옵니다.
        search_result = self.tavily_tool.search(question, max_results=3)

        web_vector_index = VectorStoreIndex.from_documents(
            search_result, embed_model=Settings.embedding
        )
        web_engine = web_vector_index.as_query_engine()
        query_result = await web_engine.aquery(question)
        source_nodes = query_result.source_nodes

        # 검색된 문서가 없다면 LLM을 사용하여 답변을 생성합니다.
        if len(source_nodes) == 0:
            return PlainAnswerReqEvent(question=question)

        return RetrievalDoneEvent(
            question=question, docs=source_nodes, answer=str(query_result)
        )

    @step
    async def retrieval_req(
        self, event: RetrievalReqEvent
    ) -> RetrievalDoneEvent | WebSearchReqEvent:
        # 내부 데이터베이스를 사용하여 답변을 생성합니다.
        question = event.question
        query_result = await Settings.query_engine.aquery(question)
        source_nodes = query_result.source_nodes

        # 검색된 문서가 없다면 Web을 사용하여 답변을 생성합니다.
        if len(source_nodes) == 0:
            return WebSearchReqEvent(question=question)

        answer = str(query_result)

        # 검색된 문서가 있다면 검증을 요청합니다.
        return RetrievalDoneEvent(question=question, docs=source_nodes, answer=answer)

    # 한 Step에 대응하는 기능이 아니므로 step decorator가 없습니다.
    async def evaluate_prompt(self, messages: List[Tuple[str, str]]) -> str:
        """Helper to run a prompt and return answer or default to 'yes' on error."""
        prompt = ChatPromptTemplate.from_messages(messages).message_templates
        try:
            result = await self.answer_llm.achat(prompt)
            return result.raw.answer
        except ValueError:
            return "yes"

    @step
    async def retrieval_done(
        self, event: RetrievalDoneEvent
    ) -> StopEvent | PlainAnswerReqEvent:
        # Refactored: delegate processing to the external static method.
        return await RetrievalDoneStep.execute(event, self.llm, self.evaluate_prompt)


# 모델 구성을 끝마쳤으면 모델 그래프를 생성하고 시각화합니다.

# In[ ]:


# 모든 가능한 흐름을 그립니다.
draw_all_possible_flows(AdaptiveRAGFlow, filename="adaptive_rag_flow.html")

flow = AdaptiveRAGFlow(verbose=True, timeout=300)


# 시각화한 그래프는 실습 메인 화면에서 `adaptive_rag_flow.html` 파일을 열어서 확인하실 수 있습니다.
# 
# 이제 챗봇을 사용해봅시다. 평균적으로 질문 하나에 약 1분 정도 소요됩니다.
# 
# - 예시 질문 (인공지능 산업 동향 연구 문서 활용): 인공지능 산업 현황 및 전망에 대해 알려줘
# - 예시 질문 (데이터 활용): 인공지능 학습 데이터의 종류를 요약해줘
# - 예시 질문 (웹 검색): Rust로 공약수를 구하는 방법에 대해 검색해서 설명해줘
# - 예시 질문 (데이터 무관, 웹 검색이 되는 경우도 많음): 오늘 저녁 뭐 먹을까?

# In[ ]:


while True:
    question = input(
        "질문을 입력해주세요 (종료를 원하시면 '종료'를 입력해주세요.): "
    )
    if question == "종료":
        break
    else:
        # flow.run is async, so we await it inside an async function
        result = await flow.run(question=question)
        print("Assistant: ", result)


# In[ ]:


# Note. 주피터 노트북 환경이 아니라 파이썬 환경 등에서 이 코드를 실행할 때는, 아래와 비슷한 비동기 처리가 필요합니다.
# async def main():
#     while True:
#         question = input(
#             "질문을 입력해주세요 (종료를 원하시면 '종료'를 입력해주세요.): "
#         )
#         if question == "종료":
#             break
#         else:
#             # flow.run is async, so we await it inside an async function
#             result = await flow.run(question=question)
#             print("Assistant: ", result)


# if __name__ == "__main__":
#     asyncio.run(main())


# ### 추가 과제
# 
# 아직 Workflow Class 내부에 구현된 `web_search_req` 메서드의 로직을 외부로 추출해보세요.
