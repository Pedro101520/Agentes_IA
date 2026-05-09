from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph
from pydantic import BaseModel
from dotenv import load_dotenv
import os

load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")

llm_model = ChatOpenAI(model="gpt-4.1-nano", api_key=API_KEY)

class GraphState(BaseModel):
    input: str
    output: str

def responder(state):
    input_message = state.input
    response = llm_model.invoke([HumanMessage(content=input_message)])
    # Os parametros da classe serão a entrada do usuário e a saída do LLM
    return GraphState(input=state.input, 
                      output=response.content
                      )

# Criando o Graph
graph = StateGraph(GraphState)
graph.add_node("responder", responder)
graph.set_entry_point("responder")
graph.set_finish_point("responder")

# Compilando o Grafo
export_graph = graph.compile()

# Testando o modelo
if __name__ == "__main__":
    result = export_graph.invoke(GraphState(input="Batata frita é bom?", output=""))
    print(result)

    # Visualizar o grafo
    print(export_graph.get_graph().draw_mermaid())