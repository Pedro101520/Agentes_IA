from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

# Aqui o sistema é multi-agente. Pode-se pensar que um Agente seria uma pessoa com uma tarefa em especifico
# dentro do código, e que é passado dentro da estrutura do grafo, e é isso que diferencia uma função normal para um agente aqui no LangGraph.
# Caso eu quiser passar as infos de agente para agente, onde cada um recebe o retorno do outro, basta pensar em uma estrutura em cadeia

# Estado
class GraphState(BaseModel):
    input: str
    output: str = ""
    tipo: str = ""

# Agente 1 - Matemático
def agente_matematico(state: GraphState) -> GraphState:
    return state.copy(update={"output": "Resultado matemático: 42"})

# Agente 2 - Escritor
def agente_escritor(state: GraphState) -> GraphState:
    return state.copy(update={"output": "Era uma vez um número chamado 42..."})

# Agente 3 - Tradutor
def agente_tradutor(state: GraphState) -> GraphState:
    return state.copy(update={"output": "Translation: The answer is 42"})

# Orquestrador - decide qual agente chamar
def orquestrador(state: GraphState) -> GraphState:
    entrada = state.input.lower()

    if any(op in entrada for op in ["+", "-", "*", "/", "calcule", "quanto"]):
        tipo = "matematico"
    elif "traduza" in entrada or "translate" in entrada:
        tipo = "tradutor"
    else:
        tipo = "escritor"

    return state.copy(update={"tipo": tipo})

# Roteador - define para qual nó ir baseado no tipo
def roteador(state: GraphState) -> str:
    return state.tipo

# Montando o grafo
graph = StateGraph(GraphState)

graph.add_node("orquestrador", orquestrador)
graph.add_node("matematico", agente_matematico)
graph.add_node("escritor", agente_escritor)
graph.add_node("tradutor", agente_tradutor)

graph.set_entry_point("orquestrador")

# Roteamento condicional
graph.add_conditional_edges(
    "orquestrador",
    roteador,
    {
        "matematico": "matematico",
        "escritor": "escritor",
        "tradutor": "tradutor"
    }
)

# Todos os agentes terminam no END
graph.add_edge("matematico", END)
graph.add_edge("escritor", END)
graph.add_edge("tradutor", END)

app = graph.compile()

# Testando
resultado = app.invoke(GraphState(input="Calcule 10 + 5"))
print(resultado.output)  # Resultado matemático: 42

resultado = app.invoke(GraphState(input="Traduza olá mundo"))
print(resultado.output)  # Translation: The answer is 42

resultado = app.invoke(GraphState(input="Me conte uma história"))
print(resultado.output)  # Era uma vez um número chamado 42...