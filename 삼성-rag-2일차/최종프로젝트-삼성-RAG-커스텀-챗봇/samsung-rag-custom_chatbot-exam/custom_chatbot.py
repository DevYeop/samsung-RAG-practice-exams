import os

from langchain.document_loaders import PyPDFLoader
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_chroma import Chroma
from langchain_community.chat_models import ChatOllama
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, StateGraph
from module import (
    LLMAnswerRAGModule,
    SelfReflectionRAGModule,
    State,
    VectorDBRetrieverRAGModule,
    WebRetrieverRAGModule,
)


class ModularRAG:
    def __init__(
        self, documents_dir, documents_description, force_reload: bool = False
    ):
        # [지시사항 1] 챗봇에서 사용할 LLM 객체와 Embedding 객체를 초기화 하세요.
        # Hint. 각각 ChatOllama, OllamaEmbeddings 클래스를 사용합니다.
        # Hint2. route_llm은 답변을 json format으로 반환해야 합니다.
        #################################################
        self.llm = "code here"
        self.route_llm = "code here"
        self.embeddings = "code here"
        #################################################

        self.top_n_reranked_docs = 3
        self.cross_encoder = HuggingFaceCrossEncoder(
            model_name="BAAI/bge-reranker-v2-m3"
        )
        self.reranker = CrossEncoderReranker(
            model=self.cross_encoder, top_n=self.top_n_reranked_docs
        )

        if force_reload or not os.path.exists(documents_dir + "/vectorstore"):
            print("Building vectorstore...")
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500, chunk_overlap=10
            )

            doc_list = []
            for root, dirs, files in os.walk(documents_dir):
                for file in files:
                    if file.endswith(".pdf"):
                        loader = PyPDFLoader(file_path=os.path.join(root, file))
                        doc = loader.load()
                        doc = text_splitter.split_documents(doc)
                        doc_list.extend(doc)

            self.vectorstore = Chroma.from_documents(
                doc_list,
                embedding=self.embeddings,
                persist_directory=documents_dir + "/vectorstore",
            )
        else:
            self.vectorstore = Chroma(
                embedding_function=self.embeddings,
                persist_directory=documents_dir + "/vectorstore",
            )

        self.db_retriever = self.vectorstore.as_retriever()
        self.documents_description = documents_description

        # [지시사항 2] module 폴더 안의 코드를 참고하여, 아래 각 field에 알맞은 Object를 초기화 하세요.
        #################################################
        self.vector_db_retriever = "code here"
        self.web_retriever = "code here"
        self.llm_answer = "code here"
        self.self_reflection = "code here"
        #################################################

        # Build workflow graph
        self.workflow = StateGraph(State)
        # Add nodes: our modules and a lambda node for plain answer generation
        self.workflow.add_node("web_retriever", self.web_retriever)
        self.workflow.add_node("vector_db_retriever", self.vector_db_retriever)
        self.workflow.add_node("llm_answer", self.llm_answer)
        self.workflow.add_node(
            "plain_answer",
            lambda state: {
                **state,
                "generation": self.llm.invoke(state["question"]).content,
            },
        )
        # Set entry point to start with data retrieval
        self.workflow.set_entry_point("vector_db_retriever")

        # Conditional edge after vector DB retrieval: if data is relevant then go to llm_answer, else try web retrieval
        self.workflow.add_conditional_edges(
            "vector_db_retriever",
            lambda state: "yes" if self._is_data_relevant(state) else "no",
            {"yes": "llm_answer", "no": "web_retriever"},
        )
        # Conditional edge after web retrieval: if relevance is confirmed, go to llm_answer; otherwise use plain answer.
        self.workflow.add_conditional_edges(
            "web_retriever",
            lambda state: "yes" if self._is_data_relevant(state) else "no",
            {"yes": "llm_answer", "no": "plain_answer"},
        )
        # Both answer nodes lead to termination.
        self.workflow.add_edge("llm_answer", END)
        self.workflow.add_edge("plain_answer", END)

        self.workflow = self.workflow.compile()

    # 정보 평가 (reuse legacy logic)
    def _is_data_relevant(self, state: State) -> bool:
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
        router_chain = relevant_judge_prompt | self.route_llm | JsonOutputParser()
        result = router_chain.invoke({"question": question, "data": data})
        print(result)
        return result.get("relevant", "no") == "yes"

    def run(self, question: str) -> str:
        # Create the initial state and invoke the workflow
        init_state = {"question": question, "data": "", "code": "", "generation": ""}
        final_state = self.workflow.invoke(init_state)
        # Evaluate answer quality using the SelfReflection module
        quality = self.self_reflection.judge_answer(final_state)
        if quality != "yes":
            print(
                "---Answer did not pass self-reflection. Falling back to plain answer.---"
            )
            final_state["generation"] = self.llm.invoke(question).content
        return final_state["generation"]
