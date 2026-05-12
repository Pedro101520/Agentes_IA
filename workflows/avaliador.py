# Aqui é uma forma de validador, o próprio LLM da um feedback para o resultado gerado, ai dependendo da resposta
# ele continua ou não com a geração

from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import START, END, StateGraph
from langchain_core.messages import HumanMessage
from langgraph.types import interrupt
from models import models
import logging

# Estados
class StoryState(TypedDict):
    prompt: str
    story: str
    feedback: str
    act: int

# Nós
def generate_story(state: StoryState):
    """Gera um novo trecho da história baseado no feedback"""
    feedback = state.get("feedback", "")
    story = state.get("story", "")
    act = state.get("act", 1)

    act_descriptions = {
        1: "primeiro ato, onde apresentamos os personagens e o conflito inicial",
        2: "segundo ato, onde o conflito se intensifica",
        3: "terceiro ato, onde chegamos ao climax da história",
        4: "quarto ato, onde resolvemos o conflito e concluimos a historia"
    }

    msg = f"""Você é um escritor criativo especializado em criar histórias envolventes.
    Sua tarefas é continuar a história no {act_descriptions[act]}

    **Instruções:**
    - Crie um trecho da história que se conecte naturalmente
    com o que já foi escrito.
    - Mantenha consistência com os personagens e eventos anteriores.
    - Seja criativo e envolvente.

    Prompt Inicial:
    {state["prompt"]}

    História até agora:
    {story}

    Feedback anterior:
    {feedback}

    Escreva apenas o novo trevho da história,
    sem explicações adicionais.
    """

    messages = [HumanMessage(content=msg)]
    response = models["gpt_4.1_nano"].invoke(messages)

    new_story = story + "\n\n" + response.content if story else response.content
    return {"story": new_story}

# Aqui é como um nó no grafo, que para a execução e espera uma aprovação, tem como fazer isso, para uma esperada do usuário
# E também tem como fazer isso, para esperar o retorno do LLM
def get_feedback(state: StoryState):
    """Solicita feedback"""
    feedback = interrupt(
        {
            "prompt": state["prompt"],
            "story": state["story"],
            "act": state["act"]
        }
    )
    logging.info(feedback)
    # Retorna o que foi indicado pelo usuário ou pelo LLM
    return {"feedback": feedback}

def router_story(state: StoryState):
    """Roteia para o próximo ato ou finaliza a história"""
    f = state.get("feedback")
    act = state.get("act", 1)

    if f == 'aprovado':
        if act < 4:
            return "next_act"
        return "final"
    return "revise"

def next_act(state: StoryState):
    """Avança para o próximo ato"""
    current_act = state.get("act", 1)
    return {"act": current_act + 1}

def final_story(state: StoryState):
    """Finaliza o workflow e retorna a história completa"""
    return {"story": state["story"]}

builder = StateGraph(StoryState)

builder.add_node("generate_story", generate_story)
builder.add_node("get_feedback", get_feedback)
builder.add_node("next_act", next_act)
builder.add_node("final_story", final_story)

builder.add_edge(START, "generate_story")
builder.add_edge("generate_story", "get_feedback")
builder.add_conditional_edges(
    "get_feedback",
    router_story,{
        "revise": "generate_story",
        "next_act": "next_act",
        "final": "final_story"
    }
)

builder.add_edge("next_act", "generate_story")
builder.add_edge("final_story", END)

story_workflow = builder.compile()

# Testando o workflow
def test():
    result = story_workflow.invoke({
        "prompt": "Um detetive que descobre que ele mesmo é o assassino",
        "story": "",
        "feedback": "",
        "act": 1
    })

    print(result["act"])

if __name__ == "__main__":
    test()