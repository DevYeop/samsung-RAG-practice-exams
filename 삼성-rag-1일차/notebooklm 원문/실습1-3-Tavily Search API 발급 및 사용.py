#!/usr/bin/env python
# coding: utf-8

# # Tavily Search API 발급 및 사용

# ## 실습 목표
# ---
# Adaptive RAG를 구현하기 위해 Tavily Search API를 발급하고 LangChain을 통해 사용해 볼 것입니다.
# 
# 24년 10월 기준 API의 무료 사용 가능 횟수는 월 1천회 로 제한되므로, 프로젝트에 사용할 분량은 남겨주세요.

# ## 실습 목차
# ---
# 
# 1. **Tavily Search API 발급:** Tavily AI에 가입하고, API Key를 발급 받습니다.
# 
# 2. **웹 검색 체인 구성:** 발급 받은 Tavily AI를 활용해 웹 데이터를 검색하고, 그 결과를 바탕으로 답변하는 체인을 구성합니다.

# ## 0. 환경 설정
# - 필요한 라이브러리를 불러옵니다.

# In[ ]:


import os
import sys
from pathlib import Path

ROOT_DIR = next(parent for parent in Path(__file__).resolve().parents if (parent / "project_env.py").is_file())
sys.path.insert(0, str(ROOT_DIR))
from project_env import load_project_env, require_env

load_project_env()
require_env("TAVILY_API_KEY")

from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough


# - Ollama를 통해 Llama3.1 모델을 불러옵니다.

# In[ ]:


get_ipython().system('ollama pull llama3.1')


# ## 1. Tavily Search API 발급
# 
# - Tavily Search API는 LLM과 RAG에 최적화된 웹 검색 API로써, LangChain 공식 튜토리얼에서도 활용하는 검색 API입니다.

# 1. 먼저, 아래 링크에 접속한 후 'Sign-in' 버튼을 눌러 로그인 화면으로 이동합니다.
#    - https://app.tavily.com/sign-in
# 2. 그 다음, 여러분의 Google, GitHub 계정과 연동되는 계정을 만들거나 아래의 'Sign up' 버튼을 눌러 새로운 계정을 생성합니다.
# 3. 계정을 생성한 후, 아래 링크에 접속하여 default API Key를 복사하여 아래 코드에 적용합니다.
#    - https://app.tavily.com/home

# In[ ]:


# Tavily API key는 tvly- 로 시작하는 문자열입니다.
# API Key를 입력했다면, 이 셀을 실행해서 API Key를 환경 변수에 등록합니다.


# API Key를 성공적으로 등록했다면, 아래 코드를 실행해서 Tavily Search API가 잘 작동하는 지 확인합니다.

# In[ ]:


from langchain_community.tools.tavily_search import TavilySearchResults

tavily_search_tool = TavilySearchResults(max_results=5)


# In[ ]:


tavily_search_tool.invoke({"query": "LangChain과 LlamaIndex의 특징과 차이점은 무엇인가요?"})


# 아래 예시와 비슷한 구조로 출력된다면 정상적으로 API를 적용한 것입니다.
# ```python
# [{'url': 'https://stack-queue.tistory.com/1684',
#   'content': 'Langchain은 LlamaIndex보다도 더 유연하며, 사용자가 응용프로그램의 동작을 자유롭게 사용자 정의할 수 있습니다. LlamaIndex는 명확한 검색 및 검색 응용프로그램을 구축하는 데 특별히 설계되었습니다. LLM에 대한 질의 및 관련 문서 검색을 위한 간단한 인터페이스를'},
#  {'url': 'https://datasciencedojo.com/blog/llamaindex-vs-langchain/',
#   'content': 'In conclusion, by recognizing the unique features and differences between LlamaIndex and LangChain, developers can more effectively align their needs with the capabilities of these tools, resulting in the construction of more efficient, powerful, and accurate search and retrieval applications powered by large language models\nOver 95,000 individuals trust our LinkedIn newsletter for the latest insights in data science, generative AI, and large language models.\n Reviews\nConsulting\nCommunity\nIndex:\nLlamaIndex vs LangChain: Understand the key differences\nMarch 3\nIn the debate of LlamaIndex vs LangChain, developers can align their needs with the capabilities of both tools, resulting in an efficient application.\n Read this blog on LlamaIndex to learn more in detail\nFeatures of LlamaIndex:\nLlamaIndex is an innovative tool designed to enhance the utilization of large language models (LLMs) by seamlessly connecting your data with the powerful computational capabilities of these models. Below are detailed use cases for LlamaIndex, specifically centered around semantic search, and case studies that highlight its indexing capabilities:\nSemantic Search with LlamaIndex:\nCase studies showcasing indexing capabilities:\nDue to its granular control and adaptability, LangChain’s framework is specifically designed to build complex applications, including context-aware query engines. As developers and innovators seek tools to expand the reach of LLMs, delving into the offerings of LLamaIndex and LangChain can guide them toward creating standout applications that resonate with efficiency, accuracy, and creativity.\n'},
#  {'url': 'https://www.vellum.ai/blog/llamaindex-vs-langchain-comparison',
#   'content': "LlamaIndex doesn't have a sandbox for testing prompts. It offers ready-to-use prompt templates, which are good for basic testing but might be limiting for more complex needs, as some users have noted. LangChain, like Llamaindex, offers simple prompt templates for various uses. It's handy for inspiration and starting out."},
# ```

# API 적용이 잘 되었다면, 이제 이 검색 결과를 활용해 답변을 생성하는 간단한 체인을 구성해 봅시다.

# ## 2. 웹 검색 체인 구성
# 
# 웹 검색 기반 답변 체인은 RAG 기반 답변 체인과 비슷한 구조를 사용해 구현할 수 있습니다.
# 
# 4챕터 실습3에서 저희는 Vector DB에서 사용자의 질문과 관련이 있는 Document를 Retrieve 했고, 그 텍스트를 이어 붙여서 프롬프트에 증강하였습니다.
# 
# 이번 실습에서 구현할 웹 체인도 이와 비슷하게, 웹 검색 결과를 이어 붙여서 프롬프트에 증강합니다.
# 

# llama3.1 모델을 사용하는 ChatOllama 객체와 OllamaEmbeddings 객체를 생성합니다.

# In[ ]:


llm = ChatOllama(model="llama3.1")
embeddings = OllamaEmbeddings(model="llama3.1")


# 다음으로, 사용자의 질문에 대해 Tavily Search API를 통해 여러 문서를 수집하고, 그 결과를 이어 붙이는 함수를 정의합니다.

# In[ ]:


def tavily_search_and_concat(query: str) -> str:
    results = tavily_search_tool.invoke({"query": query})
    return "\n".join([result["content"] for result in results])


# `init_chain()` 함수를 정의합니다.
# - 체인에서 `tavily_search_and_concat` 함수를 사용하는 것을 확인할 수 있습니다.

# In[ ]:


def init_chain():
    messages_with_contexts = [
        ("system", "웹 검색을 통해 수집한 정보를 바탕으로 질문에 답하세요."),
        ("human", "정보: {context}.\n{question}."),
    ]

    prompt_with_context = ChatPromptTemplate.from_messages(messages_with_contexts)

    # 체인 구성
    # Context로 Tavily Serach API 결과를 이어 붙이는 함수를 사용합니다.
    qa_chain = (
        {"context": tavily_search_and_concat, "question": RunnablePassthrough()}
        | prompt_with_context
        | llm
        | StrOutputParser()
    )
    
    return qa_chain


# In[ ]:


qa_chain = init_chain()


# Chain 구성이 완료되었으므로, 사용해봅시다.

# In[ ]:


question = "LangChain과 LlamaIndex의 특징과 차이점은 무엇인가요?"

print(qa_chain.invoke(question))


# Tavily Search API를 활용해서 저희는 사용자의 질문에 대해 웹에서 정보를 수집해서 이를 바탕으로 답변하는 체인을 구성했습니다. 
# 
# Adaptive RAG를 구현하면서 이 체인을 활용해서 웹 검색 기능을 추가해 볼 것입니다.
