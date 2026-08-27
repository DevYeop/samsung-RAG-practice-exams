# 챗봇의 각 기능을 모듈로 나누어 구현합니다.
from langchain_community.tools.tavily_search import TavilySearchResults
from project_env import require_env
from module import BaseRAGModule, State


class WebRetrieverRAGModule(BaseRAGModule):
    node_id = "web_retriever"

    def run(self, state: State) -> State:
        print("---웹 검색---")
        require_env("TAVILY_API_KEY")
        tavily_search_tool = TavilySearchResults(max_results=1)
        results = tavily_search_tool.invoke({"query": state["question"]})
        contents_list = [r["content"] for r in results]
        state["data"] = contents_list
        return state
