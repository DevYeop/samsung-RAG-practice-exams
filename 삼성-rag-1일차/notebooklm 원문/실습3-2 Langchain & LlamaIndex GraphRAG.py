#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# 필요한 라이브러리 설치
# !pip install -qU \
# sentence-transformers langchain langchain-openai langchain-community langchain-experimental \
# langchain-text-splitters tiktoken faiss-cpu openai pypdf requests pyvis\
# llama_index neo4j llama-index-graph-stores-neo4j llama-index-llms-langchain llama-index-embeddings-langchain \


# # LangChain/LlamaIndex를 활용한 Graph RAG 구현 실습
# 
# ## 목표
# 비정형 텍스트에서 **지식 그래프(노드, 관계)** 를 자동으로 추출하고, 이를 **VectorDB와 결합한 Graph RAG**를 구현하여 엔티티(개체) 간의 관계를 묻는 질문에 답하는 능력을 확보합니다.
# 

# In[ ]:


# OpenAI API KEY 환경변수 등록
import sys
from pathlib import Path

ROOT_DIR = next(parent for parent in Path(__file__).resolve().parents if (parent / "project_env.py").is_file())
sys.path.insert(0, str(ROOT_DIR))
from project_env import load_project_env, optional_env, require_env

load_project_env()
require_env("OPENAI_API_KEY")


# In[ ]:


from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# LLM 및 Embedding 모델 초기화
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")


# ## 샘플 문서 로드
# 비교를 위해 의미 있는 내용이 충분히 담긴 문서를 로드합니다.  
# 여기서는 **한국지능정보사회진흥원**의 AI국가전략 보고서 중 "전국민 AI 일상화 실행 계획"을 사용하겠습니다.
#  - https://www.nia.or.kr/site/nia_kor/ex/bbs/View.do?cbIdx=99952&bcIdx=27378&parentSeq=27378

# In[ ]:


import requests

# PDF 다운로드 URL
pdf_url = "https://www.nia.or.kr/common/board/Download.do?bcIdx=27378&cbIdx=99952&fileNo=1"
pdf_filename = "National_AI_Plan.pdf"

# 파일 저장
with open(pdf_filename, "wb") as f:
    f.write(requests.get(pdf_url).content)

print("PDF 파일 저장 완료:", pdf_filename)


# In[ ]:


from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("National_AI_Plan.pdf")
documents = loader.load()
print(documents[15].page_content[:500])


# In[ ]:


from llama_index.core import Document

# langchain 문서 → llama_index Document 변환
llama_documents = [Document(text=doc.page_content) for doc in documents]


# ## 지식 그래프 자동 추출 (Neo4j 연동)
# 
# LLM을 활용하여 문서에서 **(주체, 관계, 대상)** 형태의 트리플(Triplet)을 자동으로 추출하고, 이를 기반으로 지식 그래프를 생성합니다.   
# 이번 섹션에서는 Neo4j 그래프 데이터베이스에 지식 그래프를 저장하도록 설정합니다.

# ***Neo4j AuraDB:***
# 클라우드 기반의 Neo4j 서비스인 AuraDB를 사용합니다. 무료 티어가 제공되며, 연결 정보를 쉽게 얻을 수 있습니다.  
# 
# https://neo4j.com/product/auradb/
# 
# 1. 해당 링크를 통해 회원 가입
# 2. free instance 만들기 (몇 분 정도 소요됩니다.)
# 3. password 꼭 따로 저장하기 - 해당 instance에 대한 password는 한번만 보여주므로, 따로 저장해서 잘 보관해야합니다.
# 
# 아래 코드의 `NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD`에 Neo4j 연결 정보를 입력하세요.
#   
# Neo4j 사용이 불가하다면 Llamaindex에서 제공하는 인메모리 그래프 스토어를 사용합니다.

# In[ ]:


# Neo4j AuraDB 연결 정보 설정
NEO4J_URI = optional_env("NEO4J_URI")
NEO4J_USERNAME = optional_env("NEO4J_USERNAME")
NEO4J_PASSWORD = optional_env("NEO4J_PASSWORD")


# In[ ]:


from neo4j import GraphDatabase
from llama_index.core import ServiceContext, StorageContext
from llama_index.core.indices.knowledge_graph import KnowledgeGraphIndex
from llama_index.graph_stores.neo4j import Neo4jGraphStore, SimpleGraphStore  # Neo4jGraphStore 임포트
from llama_index.core.graph_stores.simple import SimpleGraphStore # Neo4j 사용 불가 시 인메모리 스토어 사용
from llama_index.core import Settings

# Neo4j 드라이버 연결 테스트
if NEO4J_URI and NEO4J_USERNAME and NEO4J_PASSWORD:
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
        driver.verify_connectivity()
        print("Neo4j 데이터베이스에 성공적으로 연결되었습니다.")
        neo4j_connected = True
    except Exception as e:
        print(f"Neo4j 데이터베이스 연결 실패: {e}")
        print("Neo4j가 실행 중이고, 루트 .env 연결 정보가 올바른지 확인하세요.")
        print("연결에 실패하면 인메모리 그래프 스토어를 사용합니다.")
        neo4j_connected = False
else:
    print("루트 .env에 Neo4j 연결 정보가 없어 인메모리 그래프 스토어를 사용합니다.")
    neo4j_connected = False

# LlamaIndex 전체 설정 반영
Settings.llm = llm
Settings.embed_model = embeddings
Settings.chunk_size = 512

# 그래프 스토어 초기화
if neo4j_connected:
    # Neo4jGraphStore 사용
    graph_store = Neo4jGraphStore(
        url=NEO4J_URI,
        username=NEO4J_USERNAME,
        password=NEO4J_PASSWORD,
        database="neo4j" # 또는 사용할 데이터베이스 이름
    )
    print("Neo4jGraphStore를 사용하도록 설정되었습니다.")
else:
    # Neo4j 연결 실패 시 인메모리 SimpleGraphStore 사용 (대체)
    graph_store = SimpleGraphStore()
    print("Neo4j 연결 실패로 인해 SimpleGraphStore (인메모리)를 사용합니다.")

storage_context = StorageContext.from_defaults(graph_store=graph_store)

if documents: # 문서가 성공적으로 로드된 경우에만 지식 그래프 생성
    print("지식 그래프 추출 및 인덱싱을 시작합니다. (시간이 약 10분 정도 소요될 수 있습니다.)")

    # KnowledgeGraphIndex 생성
    # LLM이 문서를 읽고 트리플(주체, 관계, 대상)을 추출하여 그래프를 구축합니다.
    kg_index = KnowledgeGraphIndex.from_documents(
        llama_documents,
        storage_context=storage_context,
        include_text=True, # 노드에 원본 텍스트 포함
        max_triplets_per_chunk=10, # 청크당 최대 10개의 트리플 추출
        kg_extract_k=5, # 지식 그래프에서 검색할 상위 5개 노드 (쿼리 시 사용)
        show_progress=True # 진행 상황 표시
    )

    print("\n지식 그래프 추출 및 인덱싱 완료.")
    if neo4j_connected:
        print("추출된 지식 그래프가 Neo4j 데이터베이스에 저장되었습니다.")
else:
    print("문서가 로드되지 않아 지식 그래프를 생성할 수 없습니다. PDF 로드 오류를 해결하세요.")
    kg_index = None # 지식 그래프 인덱스를 None으로 설정하여 다음 단계에서 오류 방지


# ## 지식 그래프 시각화
# 
# Neo4j에 저장된 데이터에서 쿼리로 노드와 관계를 직접 조회해 간단한 네트워크 그래프로 시각화하여, 데이터가 어떻게 구조화되었는지 눈으로 확인합니다.  

# In[ ]:


from neo4j import GraphDatabase
import pandas as pd
from pyvis.network import Network
import webbrowser

nodes_df = pd.DataFrame()
edges_df = pd.DataFrame()

# Step 1: 사용 중인 그래프 스토어에 따라 데이터 추출 방식 변경
if neo4j_connected:
    print("Neo4j 데이터베이스에서 데이터를 추출합니다.")
    # Neo4j 드라이버 초기화
    from neo4j import GraphDatabase
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

    def get_nodes_and_relationships(tx):
        nodes = {}
        for record in tx.run("MATCH (n) RETURN id(n) as id, labels(n) as labels, properties(n) as props"):
            nodes[record["id"]] = {
                "label": record["props"].get("name") or record["props"].get("id") or str(record["id"]),
            }
        relationships = []
        for record in tx.run("MATCH (s)-[r]->(o) RETURN id(s) as sid, type(r) as type, id(o) as oid"):
            relationships.append({
                "source": record["sid"],
                "target": record["oid"],
                "label": record["type"]
            })
        return nodes, relationships

    with driver.session() as session:
        nodes, relationships = session.read_transaction(get_nodes_and_relationships)

    nodes_df = pd.DataFrame([{"id": k, "label": v["label"]} for k, v in nodes.items()])
    edges_df = pd.DataFrame(relationships)

else:
    print("인메모리 SimpleGraphStore에서 데이터를 추출합니다.")
    # SimpleGraphStore에서 모든 트리플(주체, 관계, 객체)을 가져옵니다.
    # LlamaIndex v0.10.1+ 기준
    try:
        # SimpleGraphStore의 내부 딕셔너리에서 직접 데이터를 재구성합니다.
        graph_dict = graph_store._data.graph_dict
        
        nodes = set()
        relationships = []
        
        # graph_dict를 순회하여 (주체, 관계, 객체) 트리플을 만듭니다.
        for subj, rel_obj_list in graph_dict.items():
            nodes.add(subj)
            for rel, obj in rel_obj_list:
                nodes.add(obj)
                relationships.append({
                    "source": subj,
                    "target": obj,
                    "label": rel
                })
        
        # pyvis에서 사용할 수 있도록 DataFrame으로 변환합니다.
        if nodes:
             nodes_df = pd.DataFrame([{"id": node, "label": node} for node in nodes])
        if relationships:
            edges_df = pd.DataFrame(relationships)
        
        print(f"총 {len(nodes_df)}개의 노드와 {len(edges_df)}개의 엣지를 추출했습니다.")
        
    except AttributeError:
         print("'_data.graph_dict' 속성을 찾을 수 없습니다. graph_store 객체를 확인해주세요.")
    except Exception as e:
        print(f"SimpleGraphStore에서 데이터를 가져오는 중 오류 발생: {e}")
        print("그래프 데이터가 비어있을 수 있습니다.")


# Step 2: 추출된 데이터로 시각화
if not nodes_df.empty and not edges_df.empty:
    net = Network(notebook=True, height="750px", width="100%", bgcolor="#222222", font_color="white", cdn_resources='remote')

    # 노드 추가
    for _, node in nodes_df.iterrows():
        net.add_node(node['id'], label=node['label'], title=node['label'], color="#ADD8E6")

    # 엣지 추가
    for _, edge in edges_df.iterrows():
        net.add_edge(edge['source'], edge['target'], title=edge['label'], label=edge['label'], color="#FFD700")

    # 그래프 레이아웃 설정
    net.set_options("""
    var options = {
      "physics": {
        "enabled": true,
        "barnesHut": {
          "gravitationalConstant": -2000,
          "centralGravity": 0.3,
          "springLength": 95,
          "springConstant": 0.04,
          "damping": 0.09,
          "avoidOverlap": 0.5
        },
        "solver": "barnesHut"
      }
    }
    """)

    graph_html_path = "kg_visualization.html"
    net.save_graph(graph_html_path)
    print(f"\n지식 그래프 시각화가 '{graph_html_path}' 파일로 저장되었습니다.")

    # Jupyter 환경에서 바로 보기
    try:
        from IPython.display import IFrame, display
        display(IFrame(src=graph_html_path, width='100%', height='750'))
        print("Jupyter에서 그래프가 보이지 않는다면, 아래 경로를 직접 열어 확인해주세요.")
    except ImportError:
        print("IPython 환경이 아니므로 IFrame을 표시할 수 없습니다.")
    except Exception as e:
        print(f"IFrame 렌더링 중 오류 발생: {e}")

    # 브라우저로 직접 열기
    file_path = os.path.abspath(graph_html_path)
    print(f"브라우저에서 그래프 시각화를 보려면 다음 경로를 주소창에 붙여넣으세요:\nfile://{file_path}")
    try:
        webbrowser.open(f"file://{file_path}")
    except Exception as e:
        print(f"웹 브라우저를 자동으로 여는 데 실패했습니다: {e}")

else:
    print("시각화할 노드 또는 엣지 데이터가 없습니다.")


# ## Graph RAG 쿼리 엔진 구축 및 성능 테스트
# 
# 추출된 지식 그래프를 기반으로 Graph RAG 쿼리 엔진을 생성하고, 엔티티 간의 명시적인 관계를 묻는 질문에 대해 정확한 답변을 얻는지 확인합니다.

# In[ ]:


if kg_index:  # 지식 그래프 인덱스가 성공적으로 생성된 경우에만 쿼리 엔진 생성
    # KnowledgeGraphIndex에서 쿼리 엔진 생성
    graph_query_engine = kg_index.as_query_engine(
        retriever_mode="keyword",
        with_hybrid=False,
        llm=llm,
    )


# In[ ]:


# 성능 테스트
questions_graph_rag = [
    "AI 기반 복지 서비스란?",
    "의료인을 대상으로 한 AI 강의가 있나요?",
    "AI 안전 관리 플랫폼에 대한 정보가 있나요?"
]

for i, q in enumerate(questions_graph_rag):
    print(f"\n[질문 {i+1}] {q}")
    response = graph_query_engine.query(q)
    print(f"  [답변] {response.response}")
    print(f"  [소스 노드] {response.source_nodes}")  # 어떤 노드를 참조했는지 확인 가능


# ## Vector + Graph 하이브리드 검색
# 
# Vector RAG와 Graph RAG를 결합하여, 일반적인 질문은 VectorDB에서 답을 찾고, 관계에 대한 질문은 지식 그래프에서 답을 찾도록 하는 하이브리드 쿼리 엔진을 구성합니다.

# In[ ]:


from llama_index.core import VectorStoreIndex

if documents and kg_index:  # 문서와 지식 그래프 인덱스가 모두 성공적으로 생성된 경우에만 하이브리드 검색
    # VectorStoreIndex 생성 (원본 문서에 대한 벡터 인덱스)
    vector_index = VectorStoreIndex.from_documents(
        llama_documents,
    )

    # KnowledgeGraphIndex 기반 쿼리 엔진 생성, 하이브리드 모드 활성화 및 벡터 인덱스 연결
    hybrid_query_engine = kg_index.as_query_engine(
        retriever_mode="keyword",
        with_hybrid=True,                 # 하이브리드 모드 활성화
        vector_store_index=vector_index, # 벡터 인덱스 연결
        alpha=0.5,                       # 벡터 검색과 그래프 검색의 가중치 (0.5는 동일 가중치)
        llm=llm,
    )


# In[ ]:


# 하이브리드 검색 성능 테스트 질문
questions_hybrid_rag = [
    "전국민 AI 일상화를 위한 핵심 추진 과제는 무엇이며, 각 과제와 관련된 주요 서비스나 사업은 무엇인가요?",  # 일반 정보 + 관계 확인 (하이브리드형)
    "보고서에 언급된 '사회적 약자 지원'과 관련된 AI 서비스들은 서로 어떻게 연관되어 있으며, 어떤 부처들이 협력하여 추진하나요?",  # 복합 관계 추론
    "'디지털플랫폼정부위원회'는 AI 일상화 추진 과정에서 어떤 역할을 담당하며, 어느 과제와 가장 밀접하게 관련되어 있나요?",  # 특정 주체와 역할
    "보고서에서 말하는 '생성형 AI 선도국가 도약'의 의미는 무엇이며, 이를 위한 핵심 전략은 무엇인가요?" # 개념 및 전략 요약
]

for i, q in enumerate(questions_hybrid_rag):
    print(f"\n[질문 {i+1}] {q}")
    response = hybrid_query_engine.query(q)
    print(f"  [답변] {response.response}")
    print(f"  [소스 노드] {response.source_nodes}")  # 참조 노드 확인 가능


# ## 마무리
# 
# 이 실습을 통해 비정형 텍스트에서 지식 그래프를 추출하고, 이를 활용하여 관계 기반 질문에 답변하는 Graph RAG를 구현하는 방법을 익혔습니다.  
# 또한, Vector RAG와 Graph RAG를 결합한 하이브리드 검색의 가능성과 함께, **Neo4j 그래프 데이터베이스 연동**에 대해서도 살펴보았습니다.  
