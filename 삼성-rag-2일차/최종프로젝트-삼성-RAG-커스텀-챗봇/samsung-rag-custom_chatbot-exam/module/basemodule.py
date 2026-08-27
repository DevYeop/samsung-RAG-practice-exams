from typing_extensions import TypedDict


class State(TypedDict):
    # 그래프 상태의 속성을 정의합니다.
    # 질문, LLM이 생성한 텍스트, 데이터, 코드를 저장합니다.
    question: str
    generation: str
    data: str
    code: str


# LangGraph 기반 RAG 시스템을 구성하기 위해, 각 모듈이 가져야 할 메서드를 정의합니다.
class BaseRAGModule:
    # 각 모듈은 LangGraph Workflow의 노드 역할을 합니다.
    node_id: str

    # 각 모듈은 다음 노드로 전달할 데이터를 반환하는 `run` 메서드를 가져야 합니다.
    def run(self, state: State) -> State:
        raise NotImplementedError

    def __call__(self, state: State) -> State:
        return self.run(state)


# 조건부 간선을 구현하기 위한 클래스를 정의합니다.
class BaseConditionalEdgeModule:
    edge_id: str

    def run(self, state: State) -> str:
        raise NotImplementedError
