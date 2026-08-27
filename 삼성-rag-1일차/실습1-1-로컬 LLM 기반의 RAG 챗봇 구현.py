#!/usr/bin/env python
# coding: utf-8

# # 로컬 LLM 기반의 RAG 챗봇 구현

# ## 실습 목표
# ---
# 로컬에서 동작하는 오픈 소스 기반 LLM을 활용하여 문서를 바탕으로 질의 응답을 할 수 있는 챗봇을 구현합니다.

# ## 실습 목차
# ---
# 
# 1. **OllamaEmbeddings를 활용한 문서 벡터화:** 로컬 LLM을 통해 문서를 Vector로 변환하기 위한 OllamaEmbeddings를 생성합니다. 생성한 임베딩을 통해 주어진 문서를 벡터화하여 Chroma DB에 저장합니다.
# 
# 2. **Retriever Chain 구성:** 사용자의 입력과 가장 유사한 벡터화된 문서를 불러오는 Chain을 구성합니다.
# 
# 3. **RAG Chain 구성:** RAG 기반 챗봇의 일부 기능을 구현한 미니 RAG Chain을 구성해봅시다.

# ## 0. 환경 설정
# - 필요한 라이브러리를 불러옵니다.

# In[ ]:


from langchain.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


# - Ollama를 통해 llama 3.1 8B 모델을 불러옵니다.

# In[ ]:


get_ipython().system('ollama pull llama3.1')


# ## 1. 시장조사 문서 벡터화
# - RAG 챗봇에서 활용하기 위해 시장조사 파일을 읽어서 벡터화하는 과정을 실습합니다.

# 먼저, llama 3.1 모델을 사용하는 ChatOllama 객체와 OllamaEmbeddings 객체를 생성합니다.

# In[ ]:


# 현업에서 ollama를 통해 로컬 LLM을 활용하기 위해서는 ollama serve 명령어를 통해 ollama instance를 실행해야 합니다.
# 현재 실습 환경에서는 nohup ollama serve & 명령어를 통해 백그라운드에 ollama instance를 실행한 상태입니다. 
llm = ChatOllama(model="llama3.1")
embeddings = OllamaEmbeddings(model="llama3.1")


# 다음으로, 시장조사 PDF 문서를 불러와서 벡터화 해보겠습니다.
# - 한국소비자원의 2022년 키오스크(무인정보단말기) 이용 실태조사 보고서를 활용했습니다
#   - https://www.kca.go.kr/smartconsumer/sub.do?menukey=7301&mode=view&no=1003409523&page=2&cate=00000057
# - 이 실태조사 보고서는 2022년 키오스크의 사용자 경험, 접근성, 후속 조치에 대해 논의하는 보고서입니다. 
# - 이를 활용해서 키오스크를 어떻게 세일즈 할 수 있을지 아이디어를 제공하는 챗봇을 만들어야 하는 상황이라고 가정해 봅시다.

# 먼저, LangChain의 `PyPDFLoader`를 활용해서 시장조사 보고서의 텍스트를 추출하고, 페이지 별로 `Document`를 생성하여 저장합니다.

# In[ ]:


doc_path = "키오스크(무인정보단말기) 이용실태 조사.pdf"
loader = PyPDFLoader(doc_path)
docs = loader.load()


# 생성된 Document의 수와 각 Document 별 길이를 확인하는 함수를 정의하고, 불러온 보고서의 크기를 확인해 봅시다.

# In[ ]:


def profile_docs(docs):
    doc_len = [len(doc.page_content) for doc in docs]
    print(f"Page 수: {len(docs)}")
    print(f"각 Document 별 길이: {doc_len}")


# In[ ]:


profile_docs(docs)


# 1천자 미만의 문서도 있지만, 6천자가 넘는 문서도 있는 것을 확인할 수 있습니다. 이대로 그냥 사용할 경우, Context가 너무 길어져 오히려 성능이 낮아질 수도 있습니다.
# 
# 각 페이지를 1천자 단위로 나눠봅시다.

# In[ ]:


text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splited_docs = text_splitter.split_documents(docs)


# In[ ]:


profile_docs(splited_docs)


# Page 수가 많이 늘어난 대신, 1천자를 넘는 문서가 없는 것을 확인할 수 있습니다.

# ## 2. RAG 체인 구성
# RAG 체인을 구성하기 위해 `Document`를 `OllamaEmbeddings`를 활용해 벡터로 변환하고, Chroma를 활용해 Vector DB로 변환하여 저장합니다.
# - 변환 및 저장 과정은 약 2분 정도 소요됩니다.

# In[ ]:


get_ipython().run_cell_magic('time', '', 'vectorstore = Chroma.from_documents(documents=splited_docs, embedding=embeddings)\n')


# In[ ]:


db_retriever = vectorstore.as_retriever()


# Vector DB와 Retriever를 활용하는 RAG Chain을 구성합니다.
# - `RunnablePassthrough()`는 Chain의 이전 구성 요소에서 전달된 값을 그대로 전달하는 역할을 수행합니다.

# In[ ]:


def get_retrieved_text(docs):
    # Retrieve된 Document에서 page_content 정보만 추출해서 반환하는 함수입니다.
    result = "\n".join([doc.page_content for doc in docs])
    return result

def init_chain():
    messages_with_contexts = [
        ("system", "당신은 마케터를 위한 친절한 지원 챗봇입니다. 사용자가 입력하는 정보를 바탕으로 질문에 답하세요."),
        ("human", "정보: {context}.\n{question}."),
    ]

    prompt_with_context = ChatPromptTemplate.from_messages(messages_with_contexts)

    # 체인 구성
    # context에는 질문과 가장 비슷한 문서를 반환하는 db_retriever에 get_retrieved_text 함수를 적용한 chain의 결과값이 전달됩니다.
    qa_chain = (
        {"context": db_retriever | get_retrieved_text, "question": RunnablePassthrough()}
        | prompt_with_context
        | llm
        | StrOutputParser()
    )
    
    return qa_chain


# In[ ]:


qa_chain = init_chain()


# Chain 구성이 완료되었습니다.

# ## 3. 챗봇 구현 및 사용
# - 구성한 RAG 체인을 활용해서 시장조사 문서 기반 챗봇을 구현하고 사용해봅니다.

# 방금 구현한 RAG Chain을 사용해서 시장조사 문서 기반 챗봇을 구현해볼 것입니다. 
# 
# 그 전에, 별도로 RAG 기능을 추가하지 않은 LLM과 답변의 퀄리티를 비교해 봅시다.

# In[ ]:


messages_with_variables = [
    ("system", "당신은 마케터를 위한 친절한 지원 챗봇입니다."),
    ("human", "{question}."),
]
prompt = ChatPromptTemplate.from_messages(messages_with_variables)
parser = StrOutputParser()
chain = prompt | llm | parser


# In[ ]:


print(chain.invoke("키오스크 관련 설문조사 결과를 알려줘"))


# 별다른 출처를 추가하지 않은 챗봇은 알 수 없는 출처의 답변을 생성했습니다. 이제 RAG 챗봇에 동일한 질문을 입력해 봅시다.

# In[ ]:


print(qa_chain.invoke("키오스크 관련 설문조사 결과를 알려줘"))


# 일반 체인은 아무런 출처가 없는 답변을 생성한 반면, RAG 기능을 추가한 챗봇은 데이터를 기반으로 상대적으로 정확한 답변을 하는 것을 확인할 수 있습니다.
# 
# 이제 챗봇을 한번 사용해 봅시다.

# In[ ]:


qa_chain = init_chain()
while True:
    question = input("질문을 입력해주세요 (종료를 원하시면 '종료'를 입력해주세요.): ")
    if question == "종료":
        break
    else:
        result = qa_chain.invoke(question)
        print(result)

