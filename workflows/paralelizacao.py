# Aqui, basicamente são tarefas que rodam em simultaneo em dois modelos diferentes (A mesma tarefa para cada)
# E no fim, um terceiro modelo analisa o retorno de cada um, e indica qual teve o melhor resultado

from typing_extensions import TypedDict
from langgraph.graph import START, END, StateGraph
from langchain_core.messages import SystemMessage, HumanMessage
from models import models

# SystemMessage
SYSTEM_MESSAGE_LLMS = SystemMessage(content="""
Você é um especialista em análise de código e boas
práticas de programação.
Sua tarefa é analisar o código fornecido e sugerir
melhorias em termos de:
1. Performance e otimização
2. Boas práticas e padrões de código
3. Segurança e tratamento de erros
4. Legibilidade e manutenibilidade
                                    
Forneça suas sugestões de forma estruturada e clara,
com exemplos práticos de como implementar as melhorias sugeridas
Seja especifico e detalhado em suas recomendações.
""")


# Definição do Estado
class State(TypedDict):
    query: str # Código a ser analisado
    llm1: str # Análise do Gemini
    llm2: str # Análise do o4 mini
    best_llm: str # Melhor análise escolhida

# Nós
def call_llm1(state: State):
    """Recebe o código e retorna a análise do modelo Gemini"""
    messages = [
        SystemMessage(content=SYSTEM_MESSAGE_LLMS.content),
        HumanMessage(content=f"Analise o seguinte código e forneça sugestões de melhorias:\n\n{state['query']}")
    ]

    # Aqui chamando o models ele já roda aquele for que atribui valores a ele, então significa que ele já vem com os dados preenchidos
    response = models["gemini_2.5_flash"].invoke(messages)
    return {"llm1": response.content}

def call_llm2(state: State):
    """Recebe o código e retorna a análise do modelo gpt 4.1 nano"""
    messages = [
        SystemMessage(content=SYSTEM_MESSAGE_LLMS.content),
        HumanMessage(content=f"Analise o seguinte código e forneça sugestões de melhorias:\n\n{state['query']}")
    ]

    # Aqui chamando o models ele já roda aquele for que atribui valores a ele, então significa que ele já vem com os dados preenchidos
    response = models["gpt_4.1_nano"].invoke(messages)
    return {"llm2": response.content}

# Nó que escolhe o melhor resultado procecssado entre os outros dois LLMs
def judge(state: State):
    """Avalia qual análise foi mais completa e útil"""

    msg = f"""
    Aja como revisor técnico sênior e avalie a quantidade das
    análises de código fornecidas por dois especialistas.

    Sua tarefa é escolher a análise que:
    1. Identifica mais problemas potenciais
    2. Fornece sugestões mais práticas e implementáveis
    3. Considera aspectos do código, como performance, segurança, legibilidade, etc.
    4. Explica melhor o raciocínio por trás das sugestões

    [Código Analisado]
    {state['query']}

    [Análise do Especialista A]
    {state['llm1']}

    [Análise do Especialista B]
    {state['llm2']}

    Forneça sua avaliação comparativa e conclua com seu veredido
    final usando exatamente um destes formatos:
    '[[A]] se a análise A for melhor'
    '[[B]] se a análise B for melhor'
    '[[C]] em caso de empate'
    """

    messages = [SystemMessage(content=msg)]
    response = models["gpt_5.4_mini"].invoke(messages)
    return {"best_llm": response.content}

# Construindo o WorkFlow
code_analysis_builder = StateGraph(State)

# Adiciona os nós
code_analysis_builder.add_node("call_llm_1", call_llm1)
code_analysis_builder.add_node("call_llm_2", call_llm2)
code_analysis_builder.add_node("judge", judge)

# Adicionando arestas

# Aqui um ponto interessante, é que como estou lidando com paralelismo neste workflow, eu vou indicar dois nós
# como START, pois assim, vai da pra rodar os dois processos de uma vez
code_analysis_builder.add_edge(START, "call_llm_1")
code_analysis_builder.add_edge(START, "call_llm_2")

# Adicionando relacionamento entre nós usando as arestas
code_analysis_builder.add_edge("call_llm_1", "judge")
code_analysis_builder.add_edge("call_llm_2", "judge")

# Finalizando o grafo
code_analysis_builder.add_edge("judge", END)

code_analysis_builder = code_analysis_builder.compile()


# ==================================================================================================================

# Testando o fluxo
codigo_teste = """
def media():
    valores = [1,2,3,4,5]

    soma = 0
    for i in valores:
        print(i)
        soma += i
    
    media = soma/len(valores)
    print(media)

# Testando a função
media()
"""

resultado = code_analysis_builder.invoke({
    "query": codigo_teste
})

# Resultados
print("===== gemini_2.5_flash =====")
print(resultado["llm1"])

print("\n===== gpt_4.1_nano =====")
print(resultado["llm2"])

print("\n===== Fim =====")
print(resultado["best_llm"])
