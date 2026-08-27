#!/usr/bin/env python
# coding: utf-8

# # 로컬 LLM 기반의 RAG 챗봇 구현 - LlamaIndex

# ## 실습 목표
# ---
# 로컬에서 동작하는 오픈 소스 기반 LLM을 활용하여 문서를 바탕으로 질의 응답을 할 수 있는 챗봇을 구현합니다.

# ## 실습 목차
# ---
# 
# 1. **LlamaIndex 사용해보기**: 본격적으로 RAG 챗봇을 구현하기에 앞서 LlamaIndex를 사용해 문서를 불러오고 Vector DB로 저장하고, 간단한 RAG 챗봇을 구성합니다.

# ## 0. 환경 설정
# - 필요한 라이브러리를 불러옵니다.

# In[ ]:


from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import Settings
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core import VectorStoreIndex, get_response_synthesizer
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor


# - Ollama를 통해 llama 3.1 8B 모델을 불러옵니다.

# In[ ]:


get_ipython().system('ollama pull llama3.1')


# ## 1. LlamaIndex 사용해보기
# - RAG 챗봇에서 활용하기 위해 시장조사 파일을 읽어서 벡터화하는 과정을 LlamaIndex로 실습합니다.

# LlamaIndex의 글로벌 세팅에 LLM을 할당하여 LlamaIndex 시스템에서 불러올 수 있도록 설정합니다.

# In[ ]:


Settings.llm = Ollama(model="llama3.1")
Settings.embedding = OllamaEmbedding(model_name="llama3.1")


# 잘 작동하는지 확인해 봅시다.

# In[ ]:


response = Settings.llm.complete("Tell me about a apple")
print(f"Response: {response.text}")


# 다음으로, RAG 챗봇을 위한 데이터를 불러옵니다. LlamaIndex는 LangChain과 비슷하게 다양한 데이터를 처리할 수 있는 Reader를 지원하며, 그 외에도 [LlamaHub](https://llamahub.ai/) 를 통해 다양한 개발자들이 제공하는 Reader를 활용할 수 있습니다.
# 
# 이번 실습에서는 가장 간단한 Reader인 `SimpleDirectoryReader`를 통해 PDF 문서와 엑셀 데이터를 한번에 불러오겠습니다.

# In[ ]:


documents = SimpleDirectoryReader("./data").load_data()


# 문서와 데이터를 불러온 후, 종류와 출처를 살펴봅시다.

# In[ ]:


for doc in documents:
    print("Type:", doc.metadata['file_type'], " Source: ", doc.metadata['file_path'])


# 불러온 데이터들의 메타데이터를 살펴보면, PDF 파일은 페이지 수와 같은 수 만큼 생성되었고, CSV 파일은 하나 생성된 것을 확인할 수 있습니다.

# 불러온 데이터를 Vector 형태로 변환하고, DB에 저장합니다.
# - LlamaIndex에서 제공하는 `VectorStoreIndex`를 그대로 사용해봅시다. 
# - 벡터 변환 및 저장 과정은 약 30초 정도 소요됩니다.

# In[ ]:


get_ipython().run_cell_magic('time', '', '\n# embed_model에 저희가 생성한 OllamaEmbedding을 넣어줍니다.\nvector_index = VectorStoreIndex.from_documents(documents, embed_model=Settings.embedding)\n')


# DB에 저장한 벡터 데이터를 검색하는 Query Engine을 구성합니다.
# - LangChain의 Retriever를 포함하는 개념으로, Query Engine 그 자체로도 간단한 RAG 챗봇이라고 볼 수 있습니다. 
# - 사용자의 질문을 입력 받고, Retriever를 통해 적합한 문서를 탐색하고, 이를 바탕으로 답변을 생성하고 후처리한 후 반환합니다.

# In[ ]:


query_engine = vector_index.as_query_engine()


# 생성한 Query Engine에 문서와 관련 있는 질문을 입력하고, 검색 결과와 생성된 답변을 확인해봅시다.

# In[ ]:


query_result = query_engine.query("인공지능 연구 동향을 알려줘.")
print("Answer:", query_result)


# 문서가 있는 그대로 나오는 것이 아니라 RAG 챗봇 처럼 답변이 나오는 것을 확인할 수 있습니다. 검색한 문서의 이름도 확인해봅시다.

# In[ ]:


for source_node in query_result.source_nodes:
    print("Source Node:", source_node.metadata['file_path'], " Score:", source_node.score)
    if 'page_label' in source_node.metadata:
        print("Page Label:", source_node.metadata['page_label'])


# 원본 문서 중 두 페이지를 발췌하여 답변을 생성했음을 알 수 있습니다.

# 
# 이외에도 Retrieve 알고리즘, 파라미터, 답변 생성 로직, 후처리 방법 등 데이터를 불러오고 처리하는 과정을 디테일하게 조정할 수 있습니다. <br>
# 이번 실습에서는 유사도 점수 기반 필터링 후처리, Retriever의 파라미터 변경 이렇게 두 가지를 추가해 보겠습니다.
# 
# 자세한 정보는 LlamaIndex API 문서를 참고해주세요. [LlamaIndex Retriever Docs](https://docs.llamaindex.ai/en/stable/api_reference/retrievers/)

# In[ ]:


# `as_query_engine` 메서드를 통해 한번에 Query Engine을 제작하지 않고, 세부사항을 조절하기 위해 여러 단계로 나누어서 진행합니다.

# 1. Retriever를 생성하면서 상황에 맞게 조정합니다.
# Top k 인자를 5로 설정해서 한번에 유사도가 높은 최대 5개의 문서를 검색하도록 설정합니다.
retriever = VectorIndexRetriever(index=vector_index, similarity_top_k=5)

# 2. 답변을 생성하는 객체를 생성합니다.
response_synthesizer = get_response_synthesizer()

# 3. 유사도 점수 기준 후처리 객체를 생성합니다.
# 유사도 0.5 이상인 문서만을 반환하도록 설정합니다.
postprocessor = SimilarityPostprocessor(similarity_cutoff=0.5)

# 3. Retriever와 답변 생성 객체를 활용해 Query Engine을 구성합니다.
query_engine_tuned = RetrieverQueryEngine(
    retriever=retriever, response_synthesizer=response_synthesizer, node_postprocessors=[postprocessor]
)


# In[ ]:


query_result = query_engine_tuned.query("인공지능 연구 동향을 알려줘.")
print("Answer:", query_result)


# 검색한 문서를 확인해봅시다.

# In[ ]:


for source_node in query_result.source_nodes:
    print(type(source_node))
    print("Source Node:", source_node.metadata['file_path'], " Score:", source_node.score)
    if 'page_label' in source_node.metadata:
        print("Page Label:", source_node.metadata['page_label'])


# 최대 5개의 문서를 반환했으며, 모두 점수가 0.5 이상임을 알 수 있습니다.

# In[ ]:


while True:
    question = input("질문을 입력해주세요 (종료를 원하시면 '종료'를 입력해주세요.): ")
    if question == "종료":
        break
    else:
        result = query_engine_tuned.query(question)
        print(result)


# "오늘 저녁 뭐 먹을까?" 같이 Document와 무관한 주제의 질문을 하면 대부분은 `Empty Response` 가 반환될 것입니다. 이는 앞서 추가한 `SimilarityPostprocessor` 후처리로 인해 질문과 무관한 문서가 모두 필터 되었고, 이에 따라 답변을 생성하지 않았기 때문입니다.
# 
# LangGraph의 Graph를 구성한 것 처럼 Workflow를 활용하여 이런 다양한 시나리오에 맞춘 행동과 흐름을 설계할 수 있습니다.
