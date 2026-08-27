from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from env_config import load_environment, require_env

# get_ipython().system('ollama pull llama3.1')

load_environment()
require_env("TAVILY_API_KEY")

from langchain_community.tools.tavily_search import TavilySearchResults

tavily_search_tool = TavilySearchResults(max_results=5)

tavily_search_tool.invoke({"query": "LangChain과 LlamaIndex의 특징과 차이점은 무엇인가요?"})

llm = ChatOllama(model="llama3.1")
# embeddings = OllamaEmbeddings(model="llama3.1")
embeddings = OllamaEmbeddings(model="nomic-embed-text")

def tavily_search_and_concat(query: str) -> str:
    results = tavily_search_tool.invoke({"query": query})
    return "\n".join([result["content"] for result in results])

def init_chain():
    messages_with_contexts = [
        ("system", "웹 검색을 통해 수집한 정보를 바탕으로 질문에 답하세요."),
        ("human", "정보: {context}.\n{question}."),
    ]

    prompt_with_context = ChatPromptTemplate.from_messages(messages_with_contexts)

    qa_chain = (
        {"context": tavily_search_and_concat, "question": RunnablePassthrough()}
        | prompt_with_context
        | llm
        | StrOutputParser()
    )
    
    return qa_chain

qa_chain = init_chain()

question = "LangChain과 LlamaIndex의 특징과 차이점은 무엇인가요?"

print(qa_chain.invoke(question))
