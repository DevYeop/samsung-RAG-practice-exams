from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from module import BaseConditionalEdgeModule, State


class SelfReflectionRAGModule(BaseConditionalEdgeModule):
    edge_id = "self_reflection"

    def __init__(self, llm, route_llm):
        self.llm = llm
        self.route_llm = route_llm

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

        if (supportive or useful) and hallucinated is False:
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
        router_chain = relevant_judge_prompt | self.route_llm | JsonOutputParser()

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
        router_chain = useful_judge_prompt | self.route_llm | JsonOutputParser()

        result = router_chain.invoke({"question": question, "generation": generation})
        if result["useful"] == "yes":
            return True
        return False

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
        router_chain = hallucination_judge_prompt | self.route_llm | JsonOutputParser()

        result = router_chain.invoke({"generation": generation, "docs": docs})
        print(result)
        if result["answer"] == "True":
            return True
        return False
