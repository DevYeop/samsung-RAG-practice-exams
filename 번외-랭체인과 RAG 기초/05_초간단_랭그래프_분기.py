# =============================================================================
# 05. 초간단 LangGraph 분기 챗봇
# =============================================================================
#
# [이 파일의 학습 목표]
# 질문의 종류에 따라 서로 다른 LLM을 호출하는 가장 단순한 LangGraph를 만듭니다.
#
# - 코딩 질문: OpenAI의 gpt-4o-mini가 답변합니다.
# - 일상 질문: 내 컴퓨터에서 실행되는 Ollama의 llama3.1가 답변합니다.
# - exit 입력: 프로그램을 종료합니다.
#
# [전체 그래프 흐름]
#
#                              +-> OpenAI 코딩 답변 노드 -> END
# START -> 질문 분류 노드 -----+
#                              +-> Ollama 일상 답변 노드 -> END
#
# 질문 분류 노드 다음에 갈 수 있는 길이 두 개입니다.
# `add_conditional_edges()`가 현재 상태의 route 값을 확인하여 둘 중 하나를 고릅니다.
# 이것이 이 예제에서 꼭 알아야 할 LangGraph의 "조건부 분기"입니다.
#
# [최초 1회 설치]
# 터미널에서 다음 명령을 실행합니다.
#
# pip install -U langgraph langchain-openai langchain-ollama
#
# Ollama를 설치한 뒤, 터미널에서 사용할 모델도 미리 내려받습니다.
#
# ollama pull llama3.1
#
# [OpenAI API 키 입력]
# 아래 코드의 "여기에_OpenAI_API_키를_입력하세요" 부분을 실제 API 키로 바꿉니다.
#
# 주의: API 키를 코드에 직접 적으면 파일을 공유하거나 Git에 올릴 때 키가 유출될 수 있습니다.
# 이 방식은 수업용 간단 예제에서만 사용하고, 실습이 끝나면 실제 키를 꼭 지워 주세요.
#
# [실행]
# python 05_초간단_랭그래프_분기.py
# =============================================================================

import os
import sys
from pathlib import Path
from typing import Literal, TypedDict


ROOT_DIR = next(parent for parent in Path(__file__).resolve().parents if (parent / "project_env.py").is_file())
sys.path.insert(0, str(ROOT_DIR))
from project_env import load_project_env, require_env

load_project_env()
require_env("OPENAI_API_KEY")


# -----------------------------------------------------------------------------
# 1. 그래프가 들고 다닐 데이터(State)의 모양 정하기
# -----------------------------------------------------------------------------
# LangGraph에서 State는 각 노드 사이를 이동하는 "공용 메모지"와 같습니다.
#
# 질문 하나를 처리하는 동안 다음 세 값을 저장합니다.
# - question: 사용자가 입력한 원래 질문
# - route: 질문이 coding인지 daily인지 분류한 결과
# - answer: 선택된 LLM이 만든 최종 답변
#
# total=False는 그래프를 처음 실행할 때 세 값을 모두 넣지 않아도 된다는 뜻입니다.
# 첫 입력에는 question만 있고, route와 answer는 노드가 차례대로 채웁니다.
class GraphState(TypedDict, total=False):
    question: str
    route: Literal["coding", "daily"]
    answer: str


# -----------------------------------------------------------------------------
# 2. 질문을 코딩 질문과 일상 질문으로 나누는 아주 단순한 규칙
# -----------------------------------------------------------------------------
# 초간단 예제의 핵심은 "분기 흐름"을 보는 것입니다.
# 그래서 질문 분류를 위해 또 다른 LLM을 호출하지 않고, 단어 포함 여부만 확인합니다.
# 아래 단어 중 하나라도 질문에 들어 있으면 코딩 질문으로 분류합니다.
CODING_KEYWORDS = (
    "코드",
    "코딩",
    "프로그래밍",
    "파이썬",
    "python",
    "자바",
    "java",
    "자바스크립트",
    "javascript",
    "함수",
    "클래스",
    "변수",
    "에러",
    "오류",
    "디버깅",
    "알고리즘",
    "sql",
    "html",
    "css",
)


def choose_route(question: str) -> Literal["coding", "daily"]:
    """질문에 코딩 관련 단어가 있으면 coding, 없으면 daily를 반환합니다."""

    # 영어 키워드도 대소문자 구분 없이 찾을 수 있도록 질문을 소문자로 바꿉니다.
    # 한글은 lower()를 적용해도 그대로 유지됩니다.
    normalized_question = question.lower()

    # any(...)는 키워드 중 하나라도 질문에 포함되면 True가 됩니다.
    if any(keyword in normalized_question for keyword in CODING_KEYWORDS):
        return "coding"

    # 코딩 관련 단어가 하나도 없다면 이 예제에서는 모두 일상 질문으로 봅니다.
    return "daily"


# -----------------------------------------------------------------------------
# 3. LangGraph 만들기
# -----------------------------------------------------------------------------
def build_graph():
    """세 개의 노드와 조건부 엣지를 연결한 뒤 실행 가능한 그래프를 반환합니다."""

    # 필요한 라이브러리를 함수 안에서 불러옵니다.
    # 파일 위쪽의 개념 설명과 choose_route()는 라이브러리 없이도 읽고 시험할 수 있고,
    # 실제 그래프를 만들 때만 LangGraph와 각 LLM 패키지가 필요합니다.
    from langchain_ollama import ChatOllama
    from langchain_openai import ChatOpenAI
    from langgraph.graph import END, START, StateGraph

    # 각 모델 객체는 그래프를 만들 때 한 번만 생성합니다.
    # 질문을 입력할 때마다 객체를 새로 만들지 않고 같은 객체를 계속 재사용합니다.
    openai_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    ollama_llm = ChatOllama(model="llama3.1", temperature=0)

    # -------------------------------------------------------------------------
    # 노드 1: 질문 분류
    # -------------------------------------------------------------------------
    def classify_question_node(state: GraphState) -> dict:
        """질문을 분류하고, 다음 노드가 사용할 route 값을 State에 추가합니다."""

        print("\n[질문 분류 노드] 질문의 종류를 확인합니다.")

        route = choose_route(state["question"])

        if route == "coding":
            print("[질문 분류 결과] 코딩 질문 → OpenAI 노드를 선택합니다.")
        else:
            print("[질문 분류 결과] 일상 질문 → Ollama 노드를 선택합니다.")

        # LangGraph 노드는 State 전체를 다시 반환할 필요가 없습니다.
        # 새로 만들거나 바꿀 값만 딕셔너리로 반환하면 기존 State에 합쳐집니다.
        return {"route": route}

    # -------------------------------------------------------------------------
    # 노드 2: OpenAI로 코딩 질문 답변
    # -------------------------------------------------------------------------
    def openai_coding_answer_node(state: GraphState) -> dict:
        """OpenAI 모델에 코딩 질문을 보내고 답변을 State에 저장합니다."""

        # 어떤 노드가 실제로 실행되었는지 비전공자도 콘솔에서 바로 볼 수 있습니다.
        print("[OpenAI 코딩 답변 노드] gpt-4o-mini에게 질문을 보냅니다.")

        response = openai_llm.invoke(
            [
                ("system", "코딩 초보자도 이해할 수 있도록 쉽고 간결하게 답변하세요."),
                ("human", state["question"]),
            ]
        )

        # invoke()의 결과는 AIMessage 객체입니다.
        # 실제 답변 글은 content 속성에 들어 있습니다.
        return {"answer": response.content}

    # -------------------------------------------------------------------------
    # 노드 3: Ollama로 일상 질문 답변
    # -------------------------------------------------------------------------
    def ollama_daily_answer_node(state: GraphState) -> dict:
        """로컬 Ollama 모델에 일상 질문을 보내고 답변을 State에 저장합니다."""

        print("[Ollama 일상 답변 노드] llama3.2에게 질문을 보냅니다.")

        response = ollama_llm.invoke(
            [
                ("system", "일상적인 질문에 친절하고 자연스럽게 답변하세요."),
                ("human", state["question"]),
            ]
        )

        return {"answer": response.content}

    # -------------------------------------------------------------------------
    # 조건부 엣지가 사용할 길 안내 함수
    # -------------------------------------------------------------------------
    def route_to_answer_node(state: GraphState) -> Literal["coding", "daily"]:
        """분류 노드가 저장한 route 값을 그대로 돌려줍니다."""

        # 이 함수는 답변을 만들지 않습니다.
        # 단지 "coding 길로 갈지, daily 길로 갈지"를 LangGraph에 알려 줍니다.
        return state["route"]

    # StateGraph는 앞으로 노드와 엣지를 담을 그래프 설계도입니다.
    graph_builder = StateGraph(GraphState)

    # 함수 세 개를 각각 이름이 있는 노드로 등록합니다.
    # 첫 번째 인수는 그래프에서 사용할 노드 이름이고,
    # 두 번째 인수는 그 노드에 도착했을 때 실행할 함수입니다.
    graph_builder.add_node("question_classifier", classify_question_node)
    graph_builder.add_node("openai_coding_answer", openai_coding_answer_node)
    graph_builder.add_node("ollama_daily_answer", ollama_daily_answer_node)

    # START는 실제 함수가 아니라 그래프의 시작점을 나타내는 특별한 표시입니다.
    # 모든 질문은 가장 먼저 question_classifier 노드로 이동합니다.
    graph_builder.add_edge(START, "question_classifier")

    # question_classifier 실행 후에는 route_to_answer_node()가 반환한 값에 따라
    # 다음 노드가 달라집니다.
    #
    # 반환값이 "coding"이면 openai_coding_answer 노드로 이동하고,
    # 반환값이 "daily"이면 ollama_daily_answer 노드로 이동합니다.
    graph_builder.add_conditional_edges(
        "question_classifier",
        route_to_answer_node,
        {
            "coding": "openai_coding_answer",
            "daily": "ollama_daily_answer",
        },
    )

    # 두 답변 노드 중 하나가 실행되면 이번 질문의 그래프 실행은 끝납니다.
    graph_builder.add_edge("openai_coding_answer", END)
    graph_builder.add_edge("ollama_daily_answer", END)

    # compile()은 지금까지 작성한 설계도를 실제로 invoke할 수 있는 그래프로 바꿉니다.
    return graph_builder.compile()


# -----------------------------------------------------------------------------
# 4. 질문을 계속 입력받는 반복 실행 부분
# -----------------------------------------------------------------------------
def main() -> None:
    """exit를 입력할 때까지 질문을 받고, 매 질문마다 그래프를 실행합니다."""

    try:
        app = build_graph()
    except ImportError as error:
        print("필요한 라이브러리를 불러오지 못했습니다.")
        print("먼저 다음 명령을 실행하세요:")
        print("pip install -U langgraph langchain-openai langchain-ollama")
        print(f"상세 오류: {error}")
        return

    print("=" * 60)
    print("초간단 LangGraph 분기 챗봇을 시작합니다.")
    print("코딩 질문은 OpenAI, 일상 질문은 Ollama가 답변합니다.")
    print("종료하려면 exit를 입력하세요.")
    print("=" * 60)

    # while True는 break를 만날 때까지 같은 작업을 계속 반복합니다.
    # 따라서 답변이 끝나도 프로그램이 꺼지지 않고 다음 질문을 다시 받습니다.
    while True:
        question = input("\n사용자: ").strip()

        # lower()를 사용하므로 exit, EXIT, Exit를 모두 종료 명령으로 인식합니다.
        if question.lower() == "exit":
            print("챗봇을 종료합니다.")
            break

        # 사용자가 아무 글자도 입력하지 않고 Enter만 누른 경우입니다.
        # continue는 아래 코드를 실행하지 않고 while 문의 처음으로 돌아갑니다.
        if not question:
            print("질문을 입력해 주세요.")
            continue

        try:
            # 그래프의 최초 State에는 사용자의 질문 하나만 넣습니다.
            # 그래프가 실행되면서 분류 노드가 route를 추가하고,
            # 선택된 답변 노드가 answer를 추가합니다.
            final_state = app.invoke({"question": question})

            # 모든 노드 실행이 끝난 최종 State에서 answer만 꺼내 보여 줍니다.
            print(f"\nAI: {final_state['answer']}")
        except Exception as error:
            # API 키 오류, Ollama 미실행, 모델 미설치 같은 문제가 생겨도
            # 프로그램 전체를 종료하지 않고 오류를 보여 준 뒤 다음 질문을 받습니다.
            print(f"\n[오류] 답변을 만들지 못했습니다: {error}")


# 이 조건 덕분에 이 파일을 직접 실행할 때만 main()이 호출됩니다.
# 다른 파일이나 테스트에서 import할 때는 채팅 반복문이 자동으로 시작되지 않습니다.
if __name__ == "__main__":
    main()
