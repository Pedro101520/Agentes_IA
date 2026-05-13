# Exemplo consumindo o MCP que eu criei no código do main.py
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv
import os
from langchain_mcp_adapters.client import MultiServerMCPClient

load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")

model = ChatOpenAI(model="gpt-4.1-nano", api_key=API_KEY)

system_message = SystemMessage(content="""
Você é um assitente especializado em fornecer informações 
sobre comunidades de Python para GenAi.

Ferramentas disponíveis no MCP Server:

1. get_communit(location: str) -> str
- Função: retorna a melhor comunidade de Python para GenAI
- Parâmetro: location (string)
- Retorno: "Code TI"

Seu papel é ser um intermediário direto entre o usuário e
a ferramenta MCP, retornando apenas o resultado final das ferramentas
""")

def agent_mcp():
    client = MultiServerMCPClient(
        {
            "code":{
                "command": "python",
                "args": ["mcp\main.py"],
                "transport": "stdio"
            }
        }
    )
    agent = create_react_agent(model, client.get_tools(), prompt=system_message)
    return agent