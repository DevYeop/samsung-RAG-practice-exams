import contextlib
import io
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

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
from env_config import load_environment, require_env

# get_ipython().system('ollama pull llama3.1')
# os.system("ollama pull llama3.1")
# os.system("ollama pull nomic-embed-text")

llm = ChatOllama(model="llama3.1", temperature=0)
route_llm = ChatOllama(model="llama3.1", format="json", temperature=0)
# embeddings = OllamaEmbeddings(model="llama3.1")
embeddings = OllamaEmbeddings(model="nomic-embed-text")

class State(TypedDict):

    question: str
    generation: str
    data: str
    code: str

load_environment()
require_env("TAVILY_API_KEY")

excel_data_name = "한국지능정보사회진흥원_인공지능 학습용 데이터 구축 현황_20210104.csv"
pdf_data_name = "RE177_2023년 국내외 인공지능 산업 동향 연구_2장.pdf"

# 기존 방식: data_dir = "./data"
BASE_DIR = Path(__file__).resolve().parent
data_dir = BASE_DIR / "data"
df_ai_train_data_dist = pd.read_csv(
    data_dir / excel_data_name, index_col=None
)

df_name = "df_ai_train_data_dist"
df_columns = ", ".join(df_ai_train_data_dist.columns)

top_n_reranked_docs = 3
cross_encoder = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-v2-m3")
reranker = CrossEncoderReranker(model=cross_encoder, top_n=top_n_reranked_docs)

vectorstore = Chroma(
    # 기존 방식: persist_directory="./vectorstore/chroma"
    embedding_function=embeddings,
    persist_directory=str(BASE_DIR / "vectorstore" / "chroma")
)

db_retriever = vectorstore.as_retriever()

def python_code_parser(input: str) -> str:

    processed_input = input.replace("```python", "```").strip()
    parsed_input_list = processed_input.split("```")

    if len(parsed_input_list) == 1:
        return processed_input

    parsed_code_list = []
    for i in range(1, len(parsed_input_list), 2):
        parsed_code_list.append(parsed_input_list[i])

    return "\n".join(parsed_code_list)

def run_code(input_code: str):
    output = io.StringIO()
    try:
        with contextlib.redirect_stdout(output):
                                         
            exec(input_code, {"df_ai_train_data_dist": df_ai_train_data_dist})
    except Exception as e:
                                            
        print(f"Error: {e}", file=output)
                                
    return output.getvalue()

class BaseRAGModule:
                                           
    node_id: str

    def run(self, state: State) -> State:
        raise NotImplementedError

    def __call__(self, state: State) -> State:
        return self.run(state)

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

class WebRetrieverRAGModule(BaseRAGModule):
    node_id = "web_retriever"

    def run(self, state: State) -> State:
        print("---웹 검색---")
        tavily_search_tool = TavilySearchResults(max_results=1)
        results = tavily_search_tool.invoke({"query": state["question"]})
        contents_list = [r["content"] for r in results]
        state["data"] = contents_list
        return state

class DataRetreiverRAGModule(BaseRAGModule):
    node_id = "data_retriever"

    def __init__(self):
        self.df_name = df_name
        self.df_columns = df_columns

    def run(self, state: State) -> State:

        print("---데이터 쿼리---")                         
        question = state["question"]

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

        code_generate_chain = (
            prompt_with_data_info
            | llm
            | StrOutputParser()
            | python_code_parser
        )
        code = code_generate_chain.invoke({"question": question, "df_name": self.df_name, "df_columns": self.df_columns})
        data = run_code(code)
        return {"question": question, "code": code, "data": data, "generation": code}

class LLMAnswerRAGModule(BaseRAGModule):
    node_id = "llm_answer"

    def run(self, state: State) -> State:
                                                                                                
        question = state["question"]
        data = state.get("data", "")

        if data:
            print("---데이터 기반 답변 생성 (통합 로직)---")
                                                 
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
                                                       
            print("---답변 생성 (데이터 없음)---")
            generation = llm.invoke(question).content

        return {
            "question": question,
            "code": state.get("code", ""),
            "data": data,
            "generation": generation,
        }

class BaseConditionalEdgeModule:
    edge_id: str

    def run(self, state: State) -> str:
        raise NotImplementedError

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

    def _is_answer_supportive(self, state: State) -> bool:
                                     
        question = state["question"]
        generation = state["generation"]
        system_message = """당신은 AI의 답변이 사용자의 질문에 대한 해답인지 평가하는 평가자입니다."""

        user_message = """사용자의 질문: {question}
        AI의 답변: {generation}
        AI의 답변이 사용자의 질문에 대한 해답이면 'yes', 아니라면 'no'를 선택하세요.
        'answer' key 하나만 있는 JSON 형식으로 답변하고, 다른 텍스트나 설명을 생성하지 마세요."""

        message_list = [("system", system_message)]
        message_list.append(("human", user_message))

        relevant_judge_prompt = ChatPromptTemplate.from_messages(message_list)

        router_chain = relevant_judge_prompt | route_llm | JsonOutputParser()

        result = router_chain.invoke({"question": question, "generation": generation})
        if result["answer"] == "yes":
            return True
        return False

    def _is_answer_useful(self, state: State) -> bool:
                                     
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

        router_chain = useful_judge_prompt | route_llm | JsonOutputParser()

        result = router_chain.invoke({"question": question, "generation": generation})
        if result["useful"] == "yes":
            return True

    def _is_hallucinated(self, state: State) -> bool:
                                     
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

        router_chain = hallucination_judge_prompt | route_llm | JsonOutputParser()

        result = router_chain.invoke({"generation": generation, "docs": docs})
        print(result)
        if result["answer"] == "True":
            return True
        return False

class ModularRAG:
    def __init__(self, llm, route_llm, embeddings):
                                                                             
        self.llm = llm
        self.route_llm = route_llm
        self.embeddings = embeddings

        self.data_retriever = DataRetreiverRAGModule()
        self.web_retriever = WebRetrieverRAGModule()
        self.llm_answer = LLMAnswerRAGModule()
        self.self_reflection = SelfReflectionRAGModule()

        self.workflow = StateGraph(State)
                                                                              
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
                                                      
        self.workflow.set_entry_point("data_retriever")
                                                                                                                  
        self.workflow.add_conditional_edges(
            "data_retriever",
            lambda state: "yes" if self._is_data_relevant(state) else "no",
            {"yes": "llm_answer", "no": "web_retriever"},
        )
                                                                                                                        
        self.workflow.add_conditional_edges(
            "web_retriever",
            lambda state: "yes" if self._is_data_relevant(state) else "no",
            {"yes": "llm_answer", "no": "plain_answer"},
        )
                                                
        self.workflow.add_edge("llm_answer", END)
        self.workflow.add_edge("plain_answer", END)
        
        self.workflow = self.workflow.compile()
                                
    def _is_data_relevant(self, state: State) -> bool:
                                          
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
                                                          
        init_state = {"question": question, "data": "", "code": "", "generation": ""}
        final_state = self.workflow.invoke(init_state)
                                                                 
        quality = self.self_reflection.judge_answer(final_state)
        if quality != "yes":
            print(
                "---Answer did not pass self-reflection. Falling back to plain answer.---"
            )
            final_state["generation"] = self.llm.invoke(question).content
        return final_state["generation"]

modular_rag = ModularRAG(llm, route_llm, embeddings)

while True:
    question = input(
        "질문을 입력해주세요 (종료를 원하시면 '종료'를 입력해주세요.): "
    )
    if question == "종료":
        break
    answer = modular_rag.run(question)
    print("Assistant:", answer)
