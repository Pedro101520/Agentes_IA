from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import AsyncChromiumLoader
from langchain_community.document_transformers import BeautifulSoupTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from pydantic import BaseModel
from typing import List
import os

load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0
)

# Pelo o que eu entendi, aqui é a forma que o modelo vai retornar as infos extraidas dos sites em um json junto do tipo da variavel
class TimeClassificacao(BaseModel):
    posicao: int
    time: str
    jogos: int
    vitorias: int
    empates: int
    derrotas: int
    gols_pro: int
    gols_contra: int
    saldo_gols: int
    pontos: int

class Classificacao(BaseModel):
    times: List[TimeClassificacao]

structured_llm = llm.with_structured_output(Classificacao)

# Aqui o LLM faz o acesso das info e traz os dados listados no schema dado o HTML
def extract(content):
    result = structured_llm.invoke(
        f"Extraia a tabela de classificação do seguinte HTML:\n\n{content}"
    )
    return result.times

# O Langchain já tem imbutido o playwright e o beautifulsoup
def scrape_with_playwright(urls):
    loader = AsyncChromiumLoader(urls)
    docs = loader.load()

    # Serve para pegar o HTML
    bs_transformed = BeautifulSoupTransformer()

    # Do HTML inteiro passado, pega apenas as informações de código que estão dentro da TAG table
    docs_transformed = bs_transformed.transform_documents(
        documents=docs,
        tags_to_extract=['table']
    )

    # Os modelos tem limite de token por requisição, e como o código HTML que vai ser passado é muito grande, deve-se
    # "quebrar" o código HTML em pedaços menores, em pedaços com o número de tokens pré definidos, além de indicar o
    # parametro chunk_overlap com 0, que significa que não vai ter a opção de repetir o final do chunk anterior no inicio do outro
    # isso só seria necessário, em casos de RAG, em que é esperado não se perder contexto das info, mas como aqui a ideia
    # é passar o códido HTML (que vai ser muita coisa), então repetir o fim do outro no inicio, seria gasto de token desnecessário
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=2000,
        chunk_overlap=0,
    )
    # Aqui vai armazenar os pedaços do HTML
    splits = splitter.split_documents(
        documents=docs_transformed
    )

    extracted_content = []

    # Como vão ser pedaços, então vai ser necessário iterar sobre eles, para passar para depois passar para o LLM
    for split in splits:
        extracted_content.extend(
            extract(
                content=split.page_content
            )
        )
    
    # Aqui extracted_content já vai ser uma lista com os dados tratados e organizados retornados pela LLM
    return extracted_content

print(scrape_with_playwright([
    "https://ge.globo.com/futebol/brasileirao-serie-a/"
]))