# 챗봇의 각 기능을 모듈로 나누어 구현합니다.
from typing import List

from langchain_core.documents import Document
from module import BaseRAGModule, State


# 챗봇의 각 기능을 모듈로 나누어 구현합니다.
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
