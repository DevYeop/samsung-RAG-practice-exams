from langchain.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from pathlib import Path

import os

BASE_DIR = Path(__file__).resolve().parent

# get_ipython().system("ollama pull llama3.1")
# os.system("ollama pull llama3.1")
# os.system("ollama pull nomic-embed-text")

llm = ChatOllama(model="llama3.1")
# embeddings = OllamaEmbeddings(model="llama3.1")
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# doc_path = "키오스크(무인정보단말기) 이용실태 조사.pdf"
doc_path = BASE_DIR / "키오스크(무인정보단말기) 이용실태 조사.pdf"

loader = PyPDFLoader(doc_path)
docs = loader.load()

def profile_docs(docs):
    doc_len = [len(doc.page_content) for doc in docs]
    print(f"Page 수: {len(docs)}")
    print(f"각 Document 별 길이: {doc_len}")

profile_docs(docs)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

splited_docs = text_splitter.split_documents(docs)

profile_docs(splited_docs)

vectorstore = Chroma.from_documents(
    documents=splited_docs,
    embedding=embeddings
)

db_retriever = vectorstore.as_retriever()

def get_retrieved_text(docs):
    return "\n".join([doc.page_content for doc in docs])

def init_chain():
    messages_with_contexts = [
        ("system", "당신은 마케터를 위한 친절한 지원 챗봇입니다. 사용자가 입력하는 정보를 바탕으로 질문에 답하세요."),
        ("human", "정보: {context}.\n{question}."),
    ]

    prompt_with_context = ChatPromptTemplate.from_messages(
        messages_with_contexts
    )

    qa_chain = (
        {
            "context": db_retriever | get_retrieved_text,
            "question": RunnablePassthrough(),
        }
        | prompt_with_context
        | llm
        | StrOutputParser()
    )

    return qa_chain

qa_chain = init_chain()

messages_with_variables = [
    ("system", "당신은 마케터를 위한 친절한 지원 챗봇입니다."),
    ("human", "{question}."),
]

prompt = ChatPromptTemplate.from_messages(messages_with_variables)
parser = StrOutputParser()
chain = prompt | llm | parser

print(chain.invoke("키오스크 관련 설문조사 결과를 알려줘"))
print(qa_chain.invoke("키오스크 관련 설문조사 결과를 알려줘"))

qa_chain = init_chain()

while True:
    question = input("질문을 입력해주세요 (종료를 원하시면 '종료'를 입력해주세요.): ")

    if question == "종료":
        break

    result = qa_chain.invoke(question)
    print(result)