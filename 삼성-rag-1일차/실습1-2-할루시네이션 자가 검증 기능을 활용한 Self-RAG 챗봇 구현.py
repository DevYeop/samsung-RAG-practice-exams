#!/usr/bin/env python
# coding: utf-8

# # 할루시네이션 자가 검증 기능을 활용한 Self-RAG 챗봇 구현

# ## 실습 목표
# ---
# 할루시네이션을 자가 검증하는 기능을 포함하여 간략화된 Self-RAG 기능을 구현하고, 이를 활용하여 챗봇을 구현합니다.

# ## 실습 목차
# ---
# 
# 1. **Self-RAG 모듈 구현**: Self-RAG를 구현하기 위한 모듈을 구현합니다. 
#    1. **검색 문서 연관성 평가**: 검색 문서의 연관성을 평가하는 로직을 구현합니다.
#    2. **답변 평가**: 답변의 연관성과 지지 정도 (Supportive)를 평가하는 로직을 구현합니다.
#    3. **할루시네이션 평가**: 주어진 정보를 바탕으로, AI의 답변의 할루시네이션 여부를 평가하는 로직을 구현합니다.
# 2. **Self-RAG 챗봇 구현**: 1챕터에서 구현한 모듈과 기타 챗봇 모듈을 결합하여 Self-RAG 챗봇을 구현합니다.

# ## 0. 환경 설정
# 필요한 라이브러리를 불러옵니다.

# In[ ]:


import contextlib
import io
import os
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from langchain.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_community.vectorstores import FAISS
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


# ## 1. Self-RAG 구현
# 
# ### 1.1. 검색 문서 연관성 평가
# ---
# Self-RAG 챗봇을 구현하기 위해서 텍스트와 검색된 정보 간의 연관성을 LLM의 추론 능력을 활용하여 평가하는 모듈을 구현합니다.

# In[ ]:


# 정보 평가
def is_data_relevant(state: State) -> dict:
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
    # 로직 선택용 ChatOllama 객체를 생성합니다. format="json" 인자를 적용하여 출력 양식을 json으로 강제합니다.
    # 같은 질문에 항상 같은 대답을 유도하기 위해 temperature를 0으로 설정합니다.
    router_chain = relevant_judge_prompt | route_llm | JsonOutputParser()

    result = router_chain.invoke({"question": question, "data": data})
    print(result)
    return result


# 모듈이 잘 작동하는지 확인해봅시다.

# In[ ]:


# 파이썬 코드 문서 (이론 과정에서 사용한 예시입니다)
relevant_document = "math.gcd(*integers): 지정된 정수 인자의 최대 공약수를 반환합니다. 인자 중 하나가 0이 아니면, 반환된 값은 모든 인자를 나누는 가장 큰 양의 정수입니다. 모든 인자가 0이면, 반환 값은 0입니다. 인자가 없는 gcd()는 0을 반환합니다."
irrelevant_document = "re.search(pattern, string, flags=0): string을 통해 스캔하여 정규식 pattern이 일치하는 첫 번째 위치를 찾고, 대응하는 일치 객체를 반환합니다."

question = "파이썬에서 공약수를 계산하는 방법을 알려주세요."

relevant_state = State(
    question=question, generation="", data=relevant_document, code=""
)
irrelevant_state = State(
    question=question, generation="", data=irrelevant_document, code=""
)

# 문서와 질문의 연관성을 평가합니다.
is_data_relevant(relevant_state), is_data_relevant(irrelevant_state)


# ### 1.2. 답변 평가
# ---
# 답변의 유용성과 지지 정도 (Supportive)를 평가하는 모듈을 구현합니다.

# In[ ]:


# 정보 평가
def is_answer_supportive(state: State) -> dict:
    # 생성된 텍스트가 질문과 관련이 있는지 확인합니다.
    question = state["question"]
    generation = state["generation"]
    system_message = (
        """당신은 AI의 답변이 사용자의 질문에 대한 해답인지 평가하는 평가자입니다."""
    )
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
    return result


# In[ ]:


# 유용성 평가
def is_answer_useful(state: State) -> dict:
    # 생성된 텍스트가 질문에 대한 해답인지 확인합니다.
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
    # 로직 선택용 ChatOllama 객체를 생성합니다. format="json" 인자를 적용하여 출력 양식을 json으로 강제합니다.
    # 같은 질문에 항상 같은 대답을 유도하기 위해 temperature를 0으로 설정합니다.
    router_chain = useful_judge_prompt | route_llm | JsonOutputParser()

    result = router_chain.invoke({"question": question, "generation": generation})
    return result


# 간단한 질의응답 체인을 만들고, 질문과 관련 있는 문서와 관런 없는 문서를 각각 증강한 질문에 대한 답변을 저장합니다.

# In[ ]:


template = [
    ("system", "사용자가 입력하는 정보를 바탕으로 질문에 답하세요."),
    ("human", "정보: {data}.\n{question}."),
]
temp_prompt = ChatPromptTemplate.from_messages(template)
temp_chain = temp_prompt | llm | StrOutputParser()

relevant_answer = temp_chain.invoke(
    {"data": relevant_document, "question": question}
)
irrelevant_answer = temp_chain.invoke(
    {"data": irrelevant_document, "question": question}
)

print("relevant Answer:", relevant_answer)
print("Irrelevant Answer:", irrelevant_answer)


# 증강한 문서와 무관하게, LLM 모델이 자체적으로 학습한 텍스트를 바탕으로 둘 다 올바른 답변을 반환한 것을 확인할 수 있습니다. 
# 
# 두 텍스트에 더해, 임의로 작성한 정말 연관이 없는 답변까지 활용하여 테스트 해보겠습니다.

# In[ ]:


revelent_state = State(
    question=question, generation=relevant_answer, data=relevant_document, code=""
)
irrelevant_state = State(
    question=question, generation=irrelevant_answer, data=irrelevant_document, code=""
)
true_irrelevant_state = State(
    question=question,
    generation="re.search를 통해 string을 통해 스캔하여 정규식 pattern이 일치하는 첫 번째 위치를 찾고, 대응하는 일치 객체를 반환할 수 있습니다.",
    data="",
    code="",
)

# 답변과 질문의 연관성을 평가합니다.
is_answer_supportive(revelent_state), is_answer_supportive(
    irrelevant_state
), is_answer_supportive(true_irrelevant_state)


# In[ ]:


is_answer_useful(revelent_state), is_answer_useful(
    irrelevant_state
), is_answer_useful(true_irrelevant_state)


# 답변만 놓고 봤을 때 연관이 있는 앞의 두 경우는 `answer`와 `useful` 둘 다 `yes`로 잘 분류된 것을 알 수 있습니다.
# 

# ### 1.3. 검색 문서 기반 할루시네이션 평가
# ---
# 검색된 문서를 바탕으로 생성한 답변의 할루시네이션 정도를 평가하는 모듈을 구현합니다.

# In[ ]:


# 유용성 평가
def is_hallucinated(state: State) -> dict:
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
    return result


# 앞서 살펴본 `relevant_state` 와 `irrelevant_state` 모두 주어진 문서와 무관하게 사용자의 질문에 잘 답한 것을 확인할 수 있었습니다.
# 
# 하지만, `irrelevant_state`는 주어진 근거 문서 (re.search 함수의 정의) 와 관련이 없는 답변을 생성한 것과 같으므로, 일종의 할루시네이션 으로 해석할 수도 있습니다.
# 
# 실제로 그런 결과가 나오는지 확인해봅시다.

# In[ ]:


is_hallucinated(revelent_state), is_hallucinated(irrelevant_state)


# ## 2. Self-RAG 챗봇 구현
# ---
# 방금 구현한 모듈과 기타 챗봇 모듈을 결합하여 Self-RAG 챗봇을 구현합니다.

# 데이터를 불러오고, 유틸리티 함수를 정의합니다.

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


# 시장 조사 문건을 불러옵니다.
vectorstore = Chroma(
    embedding_function=embeddings, persist_directory="./vectorstore/chroma"
)

db_retriever = vectorstore.as_retriever()


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


# 단순 답변, 엑셀 데이터 분석, RAG 기능을 구현합니다.
# - 대화 기억 기능, 데이터 그래프 작성 기능은 이번 실습에서 사용하지 않습니다.

# In[ ]:


def query(state: State) -> State:
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
    # 쿼리한 데이터를 바탕으로 답변을 생성합니다.
    print("---데이터 기반 답변 생성---")  # 현재 상태를 확인하기 위한 Print문
    question = state["question"]
    data = state["data"]

    # 데이터를 바탕으로 질문에 대답하는 코드를 생성합니다.
    reasoning_system_message = "당신은 데이터를 바탕으로 질문에 답하는 데이터 분석가입니다. 사용자가 입력한 데이터를 바탕으로, 질문에 대답하세요."
    reasoning_user_message = "데이터: {data}\n{question}"

    reasoning_with_data = [
        ("system", reasoning_system_message),
        ("human", reasoning_user_message),
    ]
    reasoning_with_data_chain = (
        ChatPromptTemplate.from_messages(reasoning_with_data) | llm | StrOutputParser()
    )

    # 대답 생성
    generation = reasoning_with_data_chain.invoke({"data": data, "question": question})
    return {
        "question": question,
        "code": state["code"],
        "data": data,
        "generation": generation,
    }


def answer(state: State) -> State:
    # 데이터를 쿼리하지 않고 답변을 바로 생성합니다.
    print("---답변 생성---")  # 현재 상태를 확인하기 위한 Print문
    question = state["question"]

    return {"question": question, "generation": llm.invoke(question).content}


def retrieval(state: State) -> State:
    # 문서 검색을 수행합니다.

    def get_retrieved_text(docs):
        result = "\n".join([doc.page_content for doc in docs])
        return result

    print("---데이터 검색---")  # 현재 상태를 확인하기 위한 Print문
    question = state["question"]

    # Retrieval Chain
    retrieval_chain = db_retriever | get_retrieved_text

    data = retrieval_chain.invoke(question)

    return {"question": question, "data": data}


def answer_with_retrieved_data(state: State) -> State:
    # 검색한 문서를 바탕으로 답변을 생성합니다.
    print(
        "---검색된 데이터를 바탕으로 답변 생성---"
    )  # 현재 상태를 확인하기 위한 Print문

    question = state["question"]
    data = state["data"]

    messages_with_contexts = [
        ("system", "사용자가 입력하는 정보를 바탕으로 질문에 답하세요."),
        ("human", "정보: {data}.\n{question}."),
    ]
    prompt_with_context = ChatPromptTemplate.from_messages(messages_with_contexts)

    # 체인 구성
    qa_chain = prompt_with_context | llm | StrOutputParser()

    generation = qa_chain.invoke({"data": data, "question": question})
    return {"question": question, "data": data, "generation": generation}


# In[ ]:


def init_answer(state: State) -> State:
    # 초기 질문의 경로를 결정합니다.        
    route_system_message = """당신은 사용자의 질문에 RAG, 엑셀 데이터 중 어떤 것을 활용할 수 있는지 결정하는 전문가입니다.
인공지능 산업 동향과 관련된 질문이라면 RAG를 활용하세요.
인공지능 데이터 프로필과 관련된 질문이라면 excel_data를 활용하세요.
둘 다 아니라면, plain_answer로 충분합니다.
주어진 질문에 맞춰 `rag`, `excel_data`, `plain_answer`중 하나를 선택하세요.
답변은 `route` key 하나만 있는 JSON으로 답변하고, 다른 텍스트나 설명을 생성하지 마세요."""
    route_user_message = "{question}"
    route_prompt = ChatPromptTemplate.from_messages(
        [("system", route_system_message), ("human", route_user_message)]
    )

    router_chain = route_prompt | route_llm | JsonOutputParser()

    question = state["question"]
    route = router_chain.invoke({"question": question})["route"]
    return {"question": question, "generation": route}


# 앞서 구성한 모듈을 모두 활용하여 그래프를 구성합니다.
# 
# Self-RAG 논문에서는 서로 다른 K개의 문서를 한번에 검색하고, 한번에 평가한 후 가장 적합한 답변을 선별합니다.
# 
# 저희는 이를 간략화하여 한번에 하나씩만 검색 및 생성하고, 적합 여부를 평가하는 로직으로 구현하겠습니다.
# 

# In[ ]:


## 그래프 구성
# 그래프를 구성하기 위해 StateGraph 객체를 생성합니다.
# 생성자의 인자로 State를 전달하여 Node 간에 정보를 전달할 때 State type을 사용함을 명시합니다.
workflow = StateGraph(State)
# 앞서 정의한 Node를 모두 추가합니다.
workflow.add_node("init_answer", init_answer)

workflow.add_node("excel_data", query)
workflow.add_node("rag", retrieval)

workflow.add_node("answer_with_data", answer_with_data)
workflow.add_node("plain_answer", answer)
workflow.add_node("answer_with_retrieval", answer_with_retrieved_data)

# 시작지점을 정의합니다.
workflow.set_entry_point("init_answer")

# 간선을 정의합니다.
# END는 종결 지점을 의미합니다.
workflow.add_edge("plain_answer", END)
workflow.add_edge("answer_with_data", END)
workflow.add_edge("excel_data", "answer_with_data")


# 조건부 간선을 정의합니다.
# 1. 검색 여부를 결정합니다.
workflow.add_conditional_edges(
    "init_answer",
    lambda state: state["generation"]
    .lower()
    .strip(),  # 라우팅 결과를 추출하는 함수를 lambda로 정의합니다.
    # 어떤 노드로 이동할지 mapping합니다.
    {
        "excel_data": "excel_data",
        "rag": "rag",
        "plain_answer": "plain_answer",
    },
)

# 2. 탐색한 문서의 연관성을 평가합니다.
# `no`라면 검색한 데이터를 파기하고 plain_answer로 이동합니다.
workflow.add_conditional_edges(
    "rag",
    lambda state: is_data_relevant(state)["relevant"],
    {
        "yes": "answer_with_retrieval",
        "no": "plain_answer",
    },
)

# 3. 생성된 답변의 지지 여부와 할루시네이션 여부를 평가합니다.
# 이때, 각 함수가 이상한 답변을 생성할 경우 평가를 생략합니다.
def judge_answer(state: State) -> str:
    print("---답변 평가---")  # 현재 상태를 확인하기 위한 Print문    
    try:
        hallucinated = is_hallucinated(state)["answer"]
        # change string to boolean
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


# `yes` 라면 종결 지점으로 이동합니다.
# `no`라면 검색한 데이터를 파기하고 plain_answer로 이동합니다.
# `hallucinated`라면 검색한 데이터를 파기하고 재생성합니다.
workflow.add_conditional_edges(
    "answer_with_retrieval",
    judge_answer,
    {
        "yes": END,
        "no": "plain_answer",
        "hallucinated": "answer_with_retrieval",
    },
)

graph = workflow.compile()


# 모델 구성을 끝마쳤으면, 한번 사용해 봅시다.
# 
# Note. LLM 특성상 의도하지 않은 방식으로 작동하지 않을 가능성이 있습니다.

# In[ ]:


while True:
    question = input("질문을 입력해주세요 (종료를 원하시면 '종료'를 입력해주세요.): ")
    if question == "종료":
        break
    else:
        # graph.invoke 함수를 사용하여 그래프를 실행하고, 최종 결과를 반환합니다.
        # 답변 생성에는 약 1분 정도 소요됩니다.
        try:
            print(
                "Assistant: ",
                graph.invoke({"question": question})["generation"],
            )
        except Exception as e:
            print("Assistant: 오류가 발생했습니다. 다시 시도해주세요.")
            print(e)

