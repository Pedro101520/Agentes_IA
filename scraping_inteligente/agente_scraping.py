# Estava dando erro de compatibilidade nas bibliotecas, então criei um ambiente virtual para resolver isso
# Para ativar fiz o seguinte:
# python -m venv .venv
# .venv\Scripts\activate

from dotenv import load_dotenv
import os

from langchain_community.agent_toolkits import PlayWrightBrowserToolkit
from langchain_community.tools.playwright.utils import create_sync_playwright_browser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentType, initialize_agent

load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")

if __name__ == '__main__':
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0
    )

    browser = create_sync_playwright_browser()
    toolkit = PlayWrightBrowserToolkit.from_browser(sync_browser=browser)
    tools = toolkit.get_tools()

    agent_chain = initialize_agent(
        tools,
        llm,
        agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )

    result = agent_chain.invoke(
        input='''Quero que analise a seguinte página: https://venhasersafra.gupy.io/, e me retorne o seletor dos botões que mudam a página de exibição das vagas'''
    )

    print(result.get('output'))