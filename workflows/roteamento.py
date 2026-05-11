# Roteamento, seria passar um determinado processo para um LLM em especifico
# como por exemplo, em perguntas e respostas, onde pode-se ter dois modelos, um para perguntas tecnicas e outro
# para perguntas convencionais, ai é decidio entre os dois, qual deverá responder a pergunta recebida

from typing_extensions import TypedDict
from langgraph.graph import START, END, StateGraph
from langchain_core.messages import SystemMessage, HumanMessage
from models import models

# System Messages
SYSTEM_MESSAGE_ASSISTENTE = SystemMessage(content="""
Você é um assistente virtual especializado em ajudar 
com diferentes tipos de consulta.
Seja educado e prestativo em suas respostas.
""")

SYSTEM_MESSAGE_TECNICO = SystemMessage(content="""
Você é um especialista técnico que fornece
respostas detalhadas e precisas sobre tecnologia.
Use linguagem técnica apropriada e forneça exemplos
práticos quando possivel.
""")

SYSTEM_MESSAGE_SAUDE = SystemMessage(content="""
Você é um consultor de saúde que fornece informações gerais
sobre bem estar e saúde.
Lembre-se de sempre enfatizar que suas respostas são
apenas informativas e não substituem consultas médicas.
""")

# Estados
class State(TypedDict):
    query: str
    category: str
    answer: str

def router(state: State):
    """Roteia a consulta para diferentes categorias baseado no conteúdo"""
    query = state["query"].lower()
    palavras_tecnologia = ["python", "programação", "código", "desenvolvimento", "software", "tecnologia"]
    palavras_saude = ["saúde", "exercicio", "alimentação", "bem-estar", "medicina", "dieta"]
    if any(palavra in query for palavra in palavras_tecnologia):
        return {"category": "tecnico"}
    elif any(palavra in query for palavra in palavras_saude):
        return {"category": "saude"}
    else:
        return {"category": "assistente"}
    
def assistente(state: State):
    """Processa consultas gerais"""
    messages = [
        SYSTEM_MESSAGE_ASSISTENTE,
        HumanMessage(content=state["query"])
    ]
    response = models["gpt_4.1_nano"].invoke(messages)
    return {"answer": response.content}

def tecnico(state: State):
    """Responde perguntas tecnicas sobre tecnologia"""
    messages = [
        SYSTEM_MESSAGE_TECNICO,
        HumanMessage(content=state["query"])
    ]
    response = models["gpt_4.1_nano"].invoke(messages)
    return {"answer": response.content}

def saude(state: State):
    """Responde perguntas gerais sobre a saúde"""
    messages = [
        SYSTEM_MESSAGE_SAUDE,
        HumanMessage(content=state["query"])
    ]
    response = models["gpt_4.1_nano"].invoke(messages)
    return {"answer": response.content}

# Construindo o workflow
workflow_builder = StateGraph(State)
workflow_builder.add_node("router", router)
workflow_builder.add_node("assistente", assistente)
workflow_builder.add_node("tecnico", tecnico)
workflow_builder.add_node("saude", saude)

workflow_builder.set_entry_point("router")

workflow_builder.add_conditional_edges("router",
                                       lambda state: state["category"],{
                                           "assistente": "assistente",
                                           "tecnico": "tecnico",
                                           "saude": "saude"
                                       })

workflow_builder = workflow_builder.compile()



# Testando o código
def test_workflow():
    resultado_tecnico = workflow_builder.invoke({
        "query": "Como posso aprender Python?"
    })
    print("\n=== Consulta Técnica")
    print(f"Pergunta: Como posso aprender Python?")
    print(f"Resposta: {resultado_tecnico["answer"]}")

    resultado_saude = workflow_builder.invoke({
        "query": "Como é uma boa alimentação?"
    })
    print("\n=== Consulta Saúde")
    print(f"Pergunta: Como é uma boa alimentação?")
    print(f"Resposta: {resultado_saude["answer"]}")

    resultado = workflow_builder.invoke({
        "query": "Quem foi Dom Pedro III?"
    })
    print("\n=== Consulta")
    print(f"Pergunta: Quem foi Dom Pedro III?")
    print(f"Resposta: {resultado["answer"]}")

test_workflow()