from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from module import BaseRAGModule, State


class LLMAnswerRAGModule(BaseRAGModule):
    node_id = "llm_answer"

    def __init__(self, llm):
        self.llm = llm

    def run(self, state: State) -> State:
        question = state["question"]
        data = state.get("data", "")

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
                | self.llm
                | StrOutputParser()
            )
            generation = chain_data.invoke({"data": data, "question": question})

        else:
            # 데이터가 없는 경우 단순히 질문만으로 답변 생성 (plain answer)
            print("---답변 생성 (데이터 없음)---")
            generation = self.llm.invoke(question).content

        return {
            "question": question,
            "code": state.get("code", ""),
            "data": data,
            "generation": generation,
        }
