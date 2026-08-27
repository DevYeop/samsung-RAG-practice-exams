from pathlib import Path
from env_config import load_environment, optional_env, require_env
from download_utils import download_if_missing

load_environment()
require_env("OPENAI_API_KEY")

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

pdf_url = "https://www.nia.or.kr/common/board/Download.do?bcIdx=27378&cbIdx=99952&fileNo=1"
# 기존 방식: pdf_filename = "National_AI_Plan.pdf"
BASE_DIR = Path(__file__).resolve().parent
pdf_filename = BASE_DIR / "National_AI_Plan.pdf"

download_if_missing(pdf_url, pdf_filename)

print("PDF 파일 저장 완료:", pdf_filename)

from langchain_community.document_loaders import PyPDFLoader

# 기존 방식: loader = PyPDFLoader("National_AI_Plan.pdf")
loader = PyPDFLoader(str(pdf_filename))
documents = loader.load()
print(documents[15].page_content[:500])

from llama_index.core import Document

llama_documents = [Document(text=doc.page_content) for doc in documents]

NEO4J_URI = optional_env("NEO4J_URI")
NEO4J_USERNAME = optional_env("NEO4J_USERNAME")
NEO4J_PASSWORD = optional_env("NEO4J_PASSWORD")

from neo4j import GraphDatabase
from llama_index.core import ServiceContext, StorageContext
from llama_index.core.indices.knowledge_graph import KnowledgeGraphIndex
# from llama_index.graph_stores.neo4j import Neo4jGraphStore, SimpleGraphStore # 기존 코드: SimpleGraphStore는 neo4j 패키지에 없으므로 ImportError가 발생합니다.
from llama_index.graph_stores.neo4j import Neo4jGraphStore # 수정 코드: Neo4j 저장소와 인메모리 저장소는 각각의 올바른 모듈에서 가져옵니다.
from llama_index.core.graph_stores.simple import SimpleGraphStore                            
from llama_index.core import Settings

if NEO4J_URI and NEO4J_USERNAME and NEO4J_PASSWORD:
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
        driver.verify_connectivity()
        print("Neo4j 데이터베이스에 성공적으로 연결되었습니다.")
        neo4j_connected = True
    except Exception as e:
        print(f"Neo4j 데이터베이스 연결 실패: {e}")
        print("Neo4j가 실행 중이고, .env의 연결 정보가 올바른지 확인하세요.")
        print("연결에 실패하면 인메모리 그래프 스토어를 사용합니다.")
        neo4j_connected = False
else:
    print(".env에 Neo4j 연결 정보가 없어 인메모리 그래프 스토어를 사용합니다.")
    neo4j_connected = False

Settings.llm = llm
Settings.embed_model = embeddings
Settings.chunk_size = 512

if neo4j_connected:
                        
    graph_store = Neo4jGraphStore(
        url=NEO4J_URI,
        username=NEO4J_USERNAME,
        password=NEO4J_PASSWORD,
        database="neo4j"                   
    )
    print("Neo4jGraphStore를 사용하도록 설정되었습니다.")
else:
                                                 
    graph_store = SimpleGraphStore()
    print("Neo4j 연결 실패로 인해 SimpleGraphStore (인메모리)를 사용합니다.")

storage_context = StorageContext.from_defaults(graph_store=graph_store)

if documents:                               
    print("지식 그래프 추출 및 인덱싱을 시작합니다. (시간이 약 10분 정도 소요될 수 있습니다.)")

    kg_index = KnowledgeGraphIndex.from_documents(
        llama_documents,
        storage_context=storage_context,
        include_text=True,                
        max_triplets_per_chunk=10,                     
        kg_extract_k=5,                                  
        show_progress=True           
    )

    print("\n지식 그래프 추출 및 인덱싱 완료.")
    if neo4j_connected:
        print("추출된 지식 그래프가 Neo4j 데이터베이스에 저장되었습니다.")
else:
    print("문서가 로드되지 않아 지식 그래프를 생성할 수 없습니다. PDF 로드 오류를 해결하세요.")
    kg_index = None                                        

from neo4j import GraphDatabase
import pandas as pd
from pyvis.network import Network
import webbrowser

nodes_df = pd.DataFrame()
edges_df = pd.DataFrame()

if neo4j_connected:
    print("Neo4j 데이터베이스에서 데이터를 추출합니다.")
                    
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

    try:
                                                     
        graph_dict = graph_store._data.graph_dict
        
        nodes = set()
        relationships = []

        for subj, rel_obj_list in graph_dict.items():
            nodes.add(subj)
            for rel, obj in rel_obj_list:
                nodes.add(obj)
                relationships.append({
                    "source": subj,
                    "target": obj,
                    "label": rel
                })

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

if not nodes_df.empty and not edges_df.empty:
    net = Network(notebook=True, height="750px", width="100%", bgcolor="#222222", font_color="white", cdn_resources='remote')

    for _, node in nodes_df.iterrows():
        net.add_node(node['id'], label=node['label'], title=node['label'], color="#ADD8E6")

    for _, edge in edges_df.iterrows():
        net.add_edge(edge['source'], edge['target'], title=edge['label'], label=edge['label'], color="#FFD700")

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

    try:
        from IPython.display import IFrame, display
        display(IFrame(src=graph_html_path, width='100%', height='750'))
        print("Jupyter에서 그래프가 보이지 않는다면, 아래 경로를 직접 열어 확인해주세요.")
    except ImportError:
        print("IPython 환경이 아니므로 IFrame을 표시할 수 없습니다.")
    except Exception as e:
        print(f"IFrame 렌더링 중 오류 발생: {e}")

    file_path = os.path.abspath(graph_html_path)
    print(f"브라우저에서 그래프 시각화를 보려면 다음 경로를 주소창에 붙여넣으세요:\nfile://{file_path}")
    try:
        webbrowser.open(f"file://{file_path}")
    except Exception as e:
        print(f"웹 브라우저를 자동으로 여는 데 실패했습니다: {e}")

else:
    print("시각화할 노드 또는 엣지 데이터가 없습니다.")

if kg_index:                                       
                                    
    graph_query_engine = kg_index.as_query_engine(
        retriever_mode="keyword",
        with_hybrid=False,
        llm=llm,
    )

questions_graph_rag = [
    "AI 기반 복지 서비스란?",
    "의료인을 대상으로 한 AI 강의가 있나요?",
    "AI 안전 관리 플랫폼에 대한 정보가 있나요?"
]

for i, q in enumerate(questions_graph_rag):
    print(f"\n[질문 {i+1}] {q}")
    response = graph_query_engine.query(q)
    print(f"  [답변] {response.response}")
    print(f"  [소스 노드] {response.source_nodes}")                      

from llama_index.core import VectorStoreIndex

if documents and kg_index:                                              
                                            
    vector_index = VectorStoreIndex.from_documents(
        llama_documents,
    )

    hybrid_query_engine = kg_index.as_query_engine(
        retriever_mode="keyword",
        with_hybrid=True,                               
        vector_store_index=vector_index,            
        alpha=0.5,                                                         
        llm=llm,
    )

questions_hybrid_rag = [
    "전국민 AI 일상화를 위한 핵심 추진 과제는 무엇이며, 각 과제와 관련된 주요 서비스나 사업은 무엇인가요?",                          
    "보고서에 언급된 '사회적 약자 지원'과 관련된 AI 서비스들은 서로 어떻게 연관되어 있으며, 어떤 부처들이 협력하여 추진하나요?",            
    "'디지털플랫폼정부위원회'는 AI 일상화 추진 과정에서 어떤 역할을 담당하며, 어느 과제와 가장 밀접하게 관련되어 있나요?",             
    "보고서에서 말하는 '생성형 AI 선도국가 도약'의 의미는 무엇이며, 이를 위한 핵심 전략은 무엇인가요?"             
]

for i, q in enumerate(questions_hybrid_rag):
    print(f"\n[질문 {i+1}] {q}")
    response = hybrid_query_engine.query(q)
    print(f"  [답변] {response.response}")
    print(f"  [소스 노드] {response.source_nodes}")               
