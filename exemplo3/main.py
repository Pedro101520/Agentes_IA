from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph
from pydantic import BaseModel
from dotenv import load_dotenv
import os

load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")

llm_model = ChatOpenAI(
    model="gpt-4.1-nano", 
    api_key=API_KEY
)

# Definição do estado do graph
class GraphState(BaseModel):
    input: str
    output: str
    # Esse pelo que eu entendi, serve para recebr o tipo de ação que o agente vai seguir
    tipo: str = None


# Essas função é algo mais para entender como se trabalha com condições no LangGraph, por isso as respostas são ficticias
# Função de realizar cálculo
def realizar_calculo(state: GraphState) -> GraphState:
    print(state.tipo)
    return GraphState(input=state.input, output="Resposta de cálculo fictício: 42")

def responder_curiosidades(state: GraphState) -> GraphState:
    response = llm_model.invoke([HumanMessage(content=state.input)])
    return GraphState(input=state.input, output=response.content)

def responder_erro(state: GraphState) -> GraphState:
    return GraphState(input=state.input, output="Desculpe, não entendi sua pergunta")

# Indicando qual tipo seria por chamada de função
def classificar(state: GraphState) -> GraphState:
    pergunta = state.input.lower()
    if any(palavra in pergunta for palavra in ["soma", "quanto é", "+", "calcular"]):
        tipo = "calculo"
    elif any(palavra in pergunta for palavra in ["quem", "onde", "quando", "por que", "qual"]):
        tipo = "curiosidade"
    else:
        tipo = "desconhecido"
    
    # O input=state.input só passa a pergunta pra frente, que no caso iria para as tools, ai na tool, a unica coisa que muda é o output
    # Quando esse return acontece, o LangGraph consegue passar o estado para o próximo no fluxo
    return GraphState(input=state.input,
                      output="",
                      tipo=tipo)

graph = StateGraph(GraphState)
graph.add_node("classificar", classificar)
graph.add_node("realizar_calculo", realizar_calculo)
graph.add_node("responder_curiosidade", responder_curiosidades)
graph.add_node("responder_erro", responder_erro)

# Esse nó que chama os outros nós
# Adicionando condicionais
graph.add_conditional_edges(
    # Aqui, ele consegue acessar o state.tipo, pois eu coloquei para rodar primeiro a função classificar, e o LangGraph consegue perssistir as informações retornadas
    "classificar",
    lambda state: {
        "calculo": "realizar_calculo",
        "curiosidade": "responder_curiosidade",
        "desconhecido": "responder_erro"
    }[state.tipo]
)

# Definindo entrada e saída e compilação
graph.set_entry_point("classificar")
graph.set_finish_point(["realizar_calculo", "responder_curiosidade", "responder_erro"])
export_graph = graph.compile()


# Testando o projeto
if __name__ == "__main__":
    exemplos = [
        "Quanto é 10 + 5?",
        "Quem inventou a lâmpada?",
        "Me diga um comando especial"
    ]

    for exemplo in exemplos:
        result = export_graph.invoke(GraphState(input=exemplo, output=""))
        print(f"Pergunta: {exemplo}\nResposta: {result['output']}\n")