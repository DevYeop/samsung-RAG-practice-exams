import contextlib
import io
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from langchain_chroma import Chroma
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langgraph.graph import END, StateGraph
from langchain_ollama import ChatOllama, OllamaEmbeddings
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

def retrieve_web(state: State) -> State:
    print("---웹 검색---")
    tavily_search_tool = TavilySearchResults(max_results=1)
                                   
    results = tavily_search_tool.invoke({"query": state["question"]})

    contents_list = [r['content'] for r in results]
    state["data"] = contents_list
    return state

retrieved_state = retrieve_web({"question": "LangChain과 LlamaIndex의 특징과 차이점은 무엇인가요?"})
print(retrieved_state["data"])

def is_data_relevant(state: State) -> dict:
                                      
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

    router_chain = relevant_judge_prompt | route_llm | JsonOutputParser()

    result = router_chain.invoke({"question": question, "data": data})
    print(result)
    return result

def is_answer_supportive(state: State) -> dict:
                                 
    question = state["question"]
    generation = state["generation"]
    system_message = (
        """당신은 AI의 답변이 사용자의 질문에 대한 해답인지 평가하는 평가자입니다."""
    )

    user_message = """사용자의 질문: {question}
AI의 답변: {generation}
AI의 답변이 사용자의 질문에 대한 해답이면 'yes', 아니라면 'no'를 선택하세요.
'answer' key 하나만 있는 JSON 형식으로 답변하고, 다른 텍스트나 설명을 생성하지 마세요."""

    message_list = [("system", system_message)]
    message_list.append(("human", user_message))

    relevant_judge_prompt = ChatPromptTemplate.from_messages(message_list)

    router_chain = relevant_judge_prompt | route_llm | JsonOutputParser()

    result = router_chain.invoke({"question": question, "generation": generation})
    return result

def is_answer_useful(state: State) -> dict:
                                 
    question = state["question"]
    generation = state["generation"]
    system_message = """당신은 AI의 답변이 사용자에게 유용한지 평가하는 평가자입니다."""
    user_message = """사용자의 질문: {question}
AI의 답변: {generation}
AI의 답변이 사용자에게 유용하다면 'yes', 아니라면 'no'를 선택하세요.
'useful' key 하나만 있는 JSON 형식으로 답변하고, 다른 텍스트나 설명을 생성하지 마세요."""

    message_list = [("system", system_message)]
    message_list.append(("human", user_message))

    useful_judge_prompt = ChatPromptTemplate.from_messages(message_list)

    router_chain = useful_judge_prompt | route_llm | JsonOutputParser()

    result = router_chain.invoke({"question": question, "generation": generation})
    return result

def is_hallucinated(state: State) -> dict:
                                 
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
    return result

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

def query(state: State) -> State:

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
        {"question": RunnablePassthrough()}
        | prompt_with_data_info
        | llm
        | StrOutputParser()
        | python_code_parser
    )
    code = code_generate_chain.invoke(question)
    data = run_code(code)
    return {"question": question, "code": code, "data": data, "generation": code}

def answer_with_data(state: State) -> State:
                              
    print("---데이터 기반 답변 생성---")                         
    question = state["question"]
    data = state["data"]

    reasoning_system_message = "당신은 데이터를 바탕으로 질문에 답하는 데이터 분석가입니다. 사용자가 입력한 데이터를 바탕으로, 질문에 대답하세요."
    reasoning_user_message = "데이터: {data}\n{question}"

    reasoning_with_data = [
        ("system", reasoning_system_message),
        ("human", reasoning_user_message),
    ]
    reasoning_with_data_chain = (
        ChatPromptTemplate.from_messages(reasoning_with_data) | llm | StrOutputParser()
    )

    generation = reasoning_with_data_chain.invoke({"data": data, "question": question})
    return {
        "question": question,
        "code": state["code"],
        "data": data,
        "generation": generation,
    }

def answer(state: State) -> State:
                                
    print("---답변 생성---")                         
    question = state["question"]

    return {"question": question, "generation": llm.invoke(question).content}

def retrieval(state: State) -> State:

    def get_retrieved_text(docs):
        result = "\n".join([doc.page_content for doc in docs])
        return result

    print("---데이터 검색---")                         
    question = state["question"]

    retrieval_chain = db_retriever | get_retrieved_text

    data = retrieval_chain.invoke(question)

    return {"question": question, "data": data}

def answer_with_retrieved_data(state: State) -> State:
                             
    print(
        "---검색된 데이터를 바탕으로 답변 생성---"
    )                         

    question = state["question"]
    data = state["data"]

    messages_with_contexts = [
        ("system", "사용자가 입력하는 정보를 바탕으로 질문에 답하세요."),
        ("human", "정보: {data}.\n{question}."),
    ]
    prompt_with_context = ChatPromptTemplate.from_messages(messages_with_contexts)

    qa_chain = prompt_with_context | llm | StrOutputParser()

    generation = qa_chain.invoke({"data": data, "question": question})
    return {"question": question, "data": data, "generation": generation}

def init_answer(state: State) -> State:
                               
    route_system_message = """당신은 사용자의 질문에 RAG, 엑셀 데이터, 웹 검색 중 어떤 것을 활용할 수 있는지 결정하는 전문가입니다.
인공지능 산업 동향과 관련된 질문이라면 RAG를 활용하세요.
인공지능 데이터 프로필과 관련된 질문이라면 excel_data를 활용하세요.
둘 다 아니지만 추가 정보가 필요하다면 web_search를 활용하세요.
전부 아니라면, plain_answer로 충분합니다.
주어진 질문에 맞춰 `rag`, `excel_data`, `web_search`, `plain_answer`중 하나를 선택하세요.
답변은 `route` key 하나만 있는 JSON으로 답변하고, 다른 텍스트나 설명을 생성하지 마세요."""
    route_user_message = "{question}"
    route_prompt = ChatPromptTemplate.from_messages(
        [("system", route_system_message), ("human", route_user_message)]
    )

    router_chain = route_prompt | route_llm | JsonOutputParser()

    question = state["question"]
    route = router_chain.invoke({"question": question})["route"]
    return {"question": question, "generation": route}

workflow = StateGraph(State)
                        
workflow.add_node("init_answer", init_answer)

workflow.add_node("excel_data", query)
workflow.add_node("rag", retrieval)
workflow.add_node("web_search", retrieve_web)
workflow.add_node("answer_with_data", answer_with_data)
workflow.add_node("plain_answer", answer)

workflow.add_node("answer_with_retrieval", answer_with_retrieved_data)
workflow.add_node("answer_with_web_retrieval", answer_with_retrieved_data)

workflow.set_entry_point("init_answer")

workflow.add_edge("plain_answer", END)
workflow.add_edge("answer_with_data", END)
workflow.add_edge("excel_data", "answer_with_data")

workflow.add_conditional_edges(
    "init_answer",
    lambda state: state["generation"]
    .lower()
    .strip(),                                   
                             
    {
        "excel_data": "excel_data",
        "rag": "rag",
        "plain_answer": "plain_answer",
        "web_search": "web_search",
    },
)

workflow.add_conditional_edges(
    "web_search",
    lambda state: is_data_relevant(state)["relevant"],
    {
        "yes": "answer_with_web_retrieval",
        "no": "plain_answer",
    },
)

workflow.add_conditional_edges(
    "rag",
    lambda state: is_data_relevant(state)["relevant"],
    {
        "yes": "answer_with_retrieval",
        "no": "plain_answer",
    },
)

def judge_answer(state: State) -> str:
    print("---답변 평가---")                             
    try:
        hallucinated = is_hallucinated(state)["answer"]
                                  
        if type(hallucinated) == str:
            hallucinated = hallucinated.lower() == "true"
        print(
            "---주어진 답변은" + " 진실입니다.---"
            if hallucinated == True
            else " 진실이 아닙니다.---"
        )
    except KeyError:
        hallucinated = True
        print("---주어진 답변이 진실인지 알 수 없습니다.---")

    if not hallucinated:
        return "hallucinated"    

    try:
        supportive = is_answer_supportive(state)["answer"]
        print(
            "---주어진 답변은 " + "지원적입니다.---"
            if supportive == "yes"
            else " 지원적이지 않습니다.---"
        )
    except KeyError:
        supportive = "yes"
        print("---주어진 답변이 지원적인지 알 수 없습니다.---")

    try:
        useful = is_answer_useful(state)["useful"]
        print(
            "---주어진 답변은" + " 유용합니다.---"
            if useful == "yes"
            else " 유용하지 않습니다.---"
        )
    except KeyError:
        useful = "yes"
        print("---주어진 답변이 유용한지 알 수 없습니다.---")

    if (supportive == "yes" or useful == "yes") and hallucinated == True:
        return "yes"
    else:
        return "no"

workflow.add_conditional_edges(
    "answer_with_retrieval",
    judge_answer,
    {
        "yes": END,
        "no": "plain_answer",
        "hallucinated": "answer_with_retrieval",
    },
)

workflow.add_conditional_edges(
    "answer_with_web_retrieval",
    judge_answer,
    {
        "yes": END,
        "no": "plain_answer",
        "hallucinated": "web_search",
    },
)

graph = workflow.compile()

while True:
    question = input("질문을 입력해주세요 (종료를 원하시면 '종료'를 입력해주세요.): ")
    if question == "종료":
        break
    else:

        try:
            print(
                "Assistant: ",
                graph.invoke({"question": question})["generation"],
            )
        except Exception as e:
            print("Assistant: 오류가 발생했습니다. 다시 시도해주세요.")
            print(e)
