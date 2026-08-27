import contextlib
import io
import os
from pathlib import Path
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

# get_ipython().system('ollama pull llama3.1')

llm = ChatOllama(model="llama3.1", temperature=0)
route_llm = ChatOllama(model="llama3.1", format="json", temperature=0)
# embeddings = OllamaEmbeddings(model="llama3.1")
embeddings = OllamaEmbeddings(model="nomic-embed-text")

class State(TypedDict):
    question: str
    generation: str
    data: str
    code: str

# 검색 문서가 질문과 관련 있는지 {"relevant": "yes"} 또는 no로 판단합니다.
# 즉 유저질문과 유관하다고(유사하다고) 판단되어 추출된 문서들이 추출 되었는지 확인.
# 관련 없으면 RAG 답변을 만들지 않고, 일반 LLM 답변으로 처리하게 됨.
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

relevant_document = "math.gcd(*integers): 지정된 정수 인자의 최대 공약수를 반환합니다. 인자 중 하나가 0이 아니면, 반환된 값은 모든 인자를 나누는 가장 큰 양의 정수입니다. 모든 인자가 0이면, 반환 값은 0입니다. 인자가 없는 gcd()는 0을 반환합니다."
irrelevant_document = "re.search(pattern, string, flags=0): string을 통해 스캔하여 정규식 pattern이 일치하는 첫 번째 위치를 찾고, 대응하는 일치 객체를 반환합니다."

question = "파이썬에서 공약수를 계산하는 방법을 알려주세요."

relevant_state = State(
    question=question, generation="", data=relevant_document, code=""
)
irrelevant_state = State(
    question=question, generation="", data=irrelevant_document, code=""
)


# print(is_data_relevant(relevant_state))
# print(is_data_relevant(irrelevant_state))
# 위의 코드를 아래 코드러 줄여써서,
# is_data_relevant에 대한 결과값을 테스트 및 시연할 목적으로 작성된 코드다
is_data_relevant(relevant_state), is_data_relevant(irrelevant_state)

# “답변 자체”에 대한 평가-1
# supportive: 생성된 답변이 질문에 대한 해답이라고 말할 수 있는가?
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

# “답변 자체”에 대한 평가
# useful: 생성된 답변이 사용자에게 유용하다고 말할 수 있는가?
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

template = [
    ("system", "사용자가 입력하는 정보를 바탕으로 질문에 답하세요."),
    ("human", "정보: {data}.\n{question}."),
]
temp_prompt = ChatPromptTemplate.from_messages(template)
temp_chain = temp_prompt | llm | StrOutputParser()

# 질문과 관련된 문서를 바탕으로 답변 생성
relevant_answer = temp_chain.invoke(
    {"data": relevant_document, "question": question}
)

# 질문과 관련 없는 문서를 바탕으로 답변 생성
irrelevant_answer = temp_chain.invoke(
    {"data": irrelevant_document, "question": question}
)

# 위의 각각의 생성된 답변 내용 출력해서 테스트 해보는 목적
print("relevant Answer:", relevant_answer)
print("Irrelevant Answer:", irrelevant_answer)

# 관련 문서로 생성한 답변을 State에 저장
relevant_state = State(
    question=question, generation=relevant_answer, data=relevant_document, code=""
)

# 관련 없는 문서로 생성한 답변을 State에 저장
irrelevant_state = State(
    question=question, generation=irrelevant_answer, data=irrelevant_document, code=""
)

# 문서와 무관한 하드코딩 된 답변을 직접 넣은 테스트용 State
true_irrelevant_state = State(
    question=question,
    generation="re.search를 통해 string을 통해 스캔하여 정규식 pattern이 일치하는 첫 번째 위치를 찾고, 대응하는 일치 객체를 반환할 수 있습니다.",
    data="",
    code="",
)

# 답변이 질문에 대한 해답인지 평가 테스트 해보기 위해,
# 각각의 State를 is_answer_supportive 함수에 넣어 평가 결과를 출력
print(
    "Supportive 평가 결과:",
    is_answer_supportive(relevant_state),
    is_answer_supportive(irrelevant_state),
    is_answer_supportive(true_irrelevant_state),
)

# 답변이 유용한지 평가 테스트 해보기 위해,
# 각각의 State를 is_answer_useful 함수에 넣어 평가 결과를 출력
print(
    "Useful 평가 결과:",
    is_answer_useful(relevant_state),
    is_answer_useful(irrelevant_state),
    is_answer_useful(true_irrelevant_state),
)

# 근거 문서가 답변을 뒷받침하면 True를 반환합니다.
# True  = 근거 기반이라서 통과,
# False = 근거 불일치/환각 가능성으로 읽어야 합니다.
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

# 답변이 진실인지 평가 테스트 해보기 위해,
# 각각의 State를 is_hallucinated 함수에 넣어 평가 결과를 출력
is_hallucinated(relevant_state), is_hallucinated(irrelevant_state)

#########################################
################## 본격적으로 노드에서 활용할 함수들 정의하기 시작
#########################################

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

# 질문에 답할 판다스 코드를 LLM에게 만들게 한 뒤 실행합니다.
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

# 코드 실행 결과를 자연어 답변으로 바꿉니다.
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

# Chroma 벡터DB에서 문서를 가져와 하나의 문자열로 합칩니다.
def retrieval(state: State) -> State:

    def get_retrieved_text(docs):
        result = "\n".join([doc.page_content for doc in docs])
        return result

    print("---데이터 검색---")                         
    question = state["question"]

    retrieval_chain = db_retriever | get_retrieved_text

    data = retrieval_chain.invoke(question)

    return {"question": question, "data": data}

# 검색 문서를 컨텍스트로 넣어 답변을 생성합니다.
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

# 노드의 시작 지점으로,
# 질문을 받아서 RAG, 엑셀 데이터, 일반 답변 중 어떤 것을 활용할지 결정합니다.
def init_answer(state: State) -> State:
                               
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


## 진짜 본격적으로 노드에서 활용할 함수들 초기화 하기 시작

workflow = StateGraph(State)
                        
workflow.add_node("init_answer", init_answer)

workflow.add_node("excel_data", query)
workflow.add_node("rag", retrieval)

workflow.add_node("answer_with_data", answer_with_data)
workflow.add_node("plain_answer", answer)
workflow.add_node("answer_with_retrieval", answer_with_retrieved_data)

workflow.set_entry_point("init_answer")

workflow.add_edge("plain_answer", END)
workflow.add_edge("answer_with_data", END)
workflow.add_edge("excel_data", "answer_with_data")

# case-1 
# 람다식을 이용하여 바로 조건을 평가하고,
# 그에 따라 다음 노드로 이동하도록 설정
workflow.add_conditional_edges(
    "init_answer",
    lambda state: state["generation"]
    .lower()
    .strip(),                                   
                             
    {
        "excel_data": "excel_data",
        "rag": "rag",
        "plain_answer": "plain_answer",
    },
)

# case-2
# 위의 람다식을 쓰고 싶지 않을 경우 아래와 같이 쓸 수 있다.
# def route_from_init_answer(state: State) -> str:
#     """init_answer가 저장한 route 값을 읽어 다음 노드 이름을 반환한다."""
#     return state["generation"].lower().strip()


# workflow.add_conditional_edges(
#     "init_answer",
#     route_from_init_answer,
#     {
#         "excel_data": "excel_data",
#         "rag": "rag",
#         "plain_answer": "plain_answer",
#     },
# )

workflow.add_conditional_edges(
    "rag",
    lambda state: is_data_relevant(state)["relevant"],
    {
        "yes": "answer_with_retrieval",
        "no": "plain_answer",
    },
)

def judge_answer(state: State) -> str:
    """검색 기반 답변을 검증하고, 다음 분기 이름을 반환한다."""
    print("---답변 평가---")

    #######################################
    # 1. 근거 문서가 답변을 뒷받침하는지 검사
    try:
        hallucinated = is_hallucinated(state)["answer"]

        # LLM이 "true" 문자열을 반환한 경우 bool로 변환
        if type(hallucinated) == str:
            hallucinated = hallucinated.lower() == "true"

        print(
            "---주어진 답변은 진실입니다.---"
            if hallucinated == True
            else "---주어진 답변은 진실이 아닙니다.---"
        )

    # JSON에 "answer" 키가 없으면 KeyError 발생
    except KeyError:
        # 평가 결과를 읽을 수 없으므로, 일단 통과로 처리
        hallucinated = True
        print("---주어진 답변이 진실인지 알 수 없습니다.---")

    # 근거가 답변을 지지하지 않음 → 검색 답변을 다시 생성
    if not hallucinated:
        return "hallucinated"

    #######################################
    # 2. 답변이 질문에 제대로 답하는지 검사
    try:
        supportive = is_answer_supportive(state)["answer"]
        print(
            "---주어진 답변은 지원적입니다.---"
            if supportive == "yes"
            else "---주어진 답변은 지원적이지 않습니다.---"
        )

    # JSON에 "answer" 키가 없으면 통과 처리
    except KeyError:
        supportive = "yes"
        print("---주어진 답변이 지원적인지 알 수 없습니다.---")

    #######################################
    # 3. 답변이 사용자에게 유용한지 검사
    try:
        useful = is_answer_useful(state)["useful"]
        print(
            "---주어진 답변은 유용합니다.---"
            if useful == "yes"
            else "---주어진 답변은 유용하지 않습니다.---"
        )

    # JSON에 "useful" 키가 없으면 통과 처리
    except KeyError:
        useful = "yes"
        print("---주어진 답변이 유용한지 알 수 없습니다.---")

    # 셋 중 근거 검사는 이미 통과했고,
    # 질문 적합성 또는 유용성 중 하나만 yes여도 최종 통과
    if (supportive == "yes" or useful == "yes") and hallucinated == True:
        return "yes"

    # 답변 품질이 부족하면 일반 답변 경로로 이동
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
