from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from dotenv import load_dotenv
import os

load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")

llm_model = ChatOpenAI(model="gpt-4.1-nano", api_key=API_KEY)

system_message = SystemMessage(content="""
    Você é um assistente. Se o usuário pedir contas, use
    a ferramenta 'soma'. Caso contrário, apenas responda normalmente
""")

# Criando uma tool
@tool("soma")
def somar(valores):
    """Soma dois números separados por vírgula"""
    try:
        a,b = map(float, valores.split(","))
        return str(a+b)
    except Exception as e:
        return f"Erro ao somar: {str(e)}"

# Criando uma lista de ferramentas
tools = [somar]

# Criando o agente de IA
graph = create_react_agent(
    model=llm_model,
    tools=tools,
    prompt=system_message
)
export_graph = graph


# Extraindo a resposta final
def extrair_resposta_final(result):
    ai_messages = [m for m in result["messages"] if isinstance(m, AIMessage) and m.content]
    if ai_messages:
        return ai_messages[-1].content
    else:
        return "Nenhuma resposta final encontrada"
    
if __name__ == "__main__":
    entrada1 = HumanMessage(content="Quanto é 9 + 10?")
    result1 = export_graph.invoke({"messages":[entrada1]})
    for m in result1["messages"]:
        print(m)
    resposta_texto_1 = extrair_resposta_final(result1)
    print("Resposta 1: ", resposta_texto_1)

    entrada2 = HumanMessage(content="Quem foi Pedro Alvares Cabral")
    result2 = export_graph.invoke({"messages":[entrada2]})
    for m in result2["messages"]:
        print(m)
    resposta_texto_2 = extrair_resposta_final(result2)
    print("Resposta 1: ", resposta_texto_2)

