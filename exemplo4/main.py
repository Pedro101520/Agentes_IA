from langgraph.prebuilt import create_react_agent
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.tools import tool

from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
import os

from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

model = ChatOpenAI(
    model="gpt-5.4-mini",
    api_key=API_KEY
)

# Prompt do sistema
system_message = SystemMessage(content="""
Você é um pesquisador muito sarcástico e irônico.
Use a ferramenta 'search' sempre que necessário, especialmente
para perguntas que exigem informações da web
"""
)

# Criando a ferramenta search
@tool("search")
def search_web(query: str = "") -> str:
    """
    Busca informações na web baseada na consulta fornecida.

    Args:
        query: Termos para buscar dados na web
    
    Returns:
        As informações encontradas na web ou uma mensagem indicando
        que nenhuma informação foi encontrada.
    """
    tavily_search = TavilySearchResults(max_results=3)
    search_docs = tavily_search.invoke(query)
    return search_docs

tools = [search_web]

graph = create_react_agent(
    model,
    tools=tools,
    prompt=system_message
)

# Fiz isso, para conseguir rodar pelo LangSmith, esse export_graph, é o que eu passo no JSON
export_graph = graph

# Testando o projeto
# if __name__ == "__main__":
#     exemplos = [
#         "Qual problema deu com os produtos Ypê?"
#     ]

#     for exemplo in exemplos:
#         result = graph.invoke({"messages": [HumanMessage(content=exemplo)]})
#         resposta = result["messages"][-1].content
#         print(f"Pergunta: {exemplo}\nResposta: {resposta}\n")