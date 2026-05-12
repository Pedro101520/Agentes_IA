# Aqui a parte do workflow funciona em cadeia, ou seja, um nó depende da saída de um para conseguir continnuar com as respostas

from typing import Annotated
from typing_extensions import TypedDict
from operator import or_
import random

from langgraph.graph import START, END, StateGraph
from langchain_core.messages import HumanMessage
from models import models

# Estados
class State(TypedDict):
    detective: str
    crime: str
    location: str
    clue: str
    story: Annotated[dict[str, str], or_]

# Nós
def first_act(state: State):
    """Escreve o Primeiro Ato"""
    detective = random.choice([
        "Um detetive excêntrico com memória fotográfica",
        "Um investigador que resolve casos através de análises de padrões",
        "Um detetive aposentado que não consegue resistir a um bom mistério"
    ])
    crime = random.choice([
        "O desaparecimento de um objeto valioso",
        "Uma mensagem codificada deixada na cena do crime",
        "Um assassinato aparentemente impossivel"
    ])
    location = random.choice([
        "Uma mansão vitoriana isolada",
        "Um museu de antiguidades",
        "Um clube exclusivo de alta sociedade"
    ])
    clue = random.choice([
        "Uma marca de cigarro rara",
        "Um relógio parado em um horário especifico",
        "Uma carta de baralho manchada"
    ])

    msg = f"""Você é um escritor de mistério experiente
    encarregado de enscrever o Primeiro Ato de uma história de detetive.

        **Instruções:**

        Baseado nas seguintes informações iniciais:
        *   **Detetive:** {detective}
        *   **Crime/Mistério:** {crime}
        *   **Local:** {location}
        *   **Pista Inicial:** {clue}

        Escreva o **Primeiro Ato** desta história. Este ato deve:
        1. ** Apresentar o Detetive:** Mostre quem é o detetive e suas
        caracteristicas únicas.
        2. **Estabelecer o Contexto:** Descreva o local e a situação
        antes do crime ser descoberto
        3. ** Introduzir o Mistério:** O momento em que o detetive é
        chamado para investigar {crime}.
        4. ** Terminar com o Primeiro Ponto de Virada:** O detetive encontra
        {clue} e decide aceitar o caso.
    """

    messages = [HumanMessage(content=msg)]
    response = models["gpt_4.1_nano"].invoke(messages)

    return{
        "story": {"act_1":response.content},
        "detective": detective,
        "crime": crime,
        "location": location,
        "clue":clue
    }