from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools import Tool
from langchain import hub
from ddgs import DDGS
from dotenv import load_dotenv
import os
# Pedi para o claude gerar uma versão mais moderna de Agente de IA
# Consegui entender que workflows de IA são melhores quando se quer algo mais especifico e trabalhado
# Mas quando se quer algo mais amplo e dinamico, agentes de IA são melhores

# Entretanto, tem como usar Agentes de IA dentro de um workflow de IA, onde por exemplo, tem como fazer isso em um nó, e passar no fluxo e perssistir as informações trabalhadas (Isso é bom para quando se quer que o agente analise algo essencial para o fluxo, mas não tem como saber de forma conssistente, então o agente de IA entra pois se sai melhor com tarefas mais dinamicas)

load_dotenv()

llm = ChatOpenAI(model="gpt-4.1-nano", temperature=0)

# Aqui a criação de Tools, ficou diferente
def buscar_vagas(query: str) -> str:
    with DDGS() as ddgs:
        results = list(ddgs.text(f"vagas de emprego {query} site:linkedin.com OR site:gupy.io", max_results=5))
        if not results:
            return "Nenhuma vaga encontrada"
        return "\n".join([r["body"] for r in results])

def buscar_empresa(empresa: str) -> str:
    with DDGS() as ddgs:
        results = list(ddgs.text(f"{empresa} empresa cultura salário tecnologia stack", max_results=5))
        if not results:
            return "Nenhuma informação encontrada"
        return "\n".join([r["body"] for r in results])

def buscar_salario(cargo: str) -> str:
    with DDGS() as ddgs:
        results = list(ddgs.text(f"salário médio {cargo} Brasil glassdoor", max_results=5))
        if not results:
            return "Nenhuma informação salarial encontrada"
        return "\n".join([r["body"] for r in results])

# Gostei mais dessa abordagem para a declaração das Tools
tools = [
    Tool(
        name="buscar_vagas",
        func=buscar_vagas,
        description="Use para buscar vagas de emprego. Input: cargo e/ou tecnologia desejada. Exemplo: 'Python júnior São Paulo'"
    ),
    Tool(
        name="buscar_empresa",
        func=buscar_empresa,
        description="Use para pesquisar informações sobre uma empresa específica. Input: nome da empresa"
    ),
    Tool(
        name="buscar_salario",
        func=buscar_salario,
        description="Use para pesquisar faixa salarial de um cargo. Input: nome do cargo"
    )
]

# Criando o Agente
prompt = hub.pull("hwchase17/react")

agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=6,
    max_execution_time=45,
    early_stopping_method="generate" 
)


perguntas = [
    "A empresa Nubank é boa para trabalhar? Quais tecnologias eles usam? Qual o salário de um dev java júnior"
]

for pergunta in perguntas:
    print(f"\n{'='*50}")
    print(f"Pergunta: {pergunta}")
    print('='*50)
    resultado = agent_executor.invoke({"input": pergunta})
    print(f"\nResposta final: {resultado['output']}")