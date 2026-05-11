# essa arquivo é para facilitar o uso dos modelos nos demais scripts

from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

#API_KEY dos modelos que estão adicionados aqui para uso
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Passa o provedor, que no caso é o nome da empresa, junto da função do import ao respectivo modelo a ser usado
_PROVIDER_MAP = {
    "openai": ChatOpenAI,
    "google": ChatGoogleGenerativeAI
}

# Se quiser adicionar mais algum modelo, vai ser aqui
MODEL_CONFIGS = [
    {
        "key_name": "gemini_2.5_flash",
        "provider": "google",
        "model_name": "gemini-2.5-flash",
        "temperature": 1.0
    },
    {
        "key_name": "gpt_5.4_mini",
        "provider": "openai",
        "model_name": "gpt-5.4-mini"
    },
    {
        "key_name": "gpt_4.1_nano",
        "provider": "openai",
        "model_name": "gpt-4.1-nano"
    }
]

def _create_chat_model(model_name: str, provider: str, temperature: float | None = None):
    # Se o provedor informado, não estiver dentro dos disponiveis, eu lanço um erro
    if provider not in _PROVIDER_MAP:
        raise ValueError(f"Provedor não suportado: {provider}. Provedores suportados são: {list(_PROVIDER_MAP.keys())}")

    model_class = _PROVIDER_MAP[provider]
    params = {"model": model_name}
    if temperature is not None:
        params["temperature"] = temperature

    # Aqui por conta do **, ele retorna um código pronto, ele pega o valor do dicionario _PROVIDER_MAP e engloba dentro dele os parametros do dicionario params
    return model_class(**params)

models = {}

# itera por bloco no dicionario, e armazena como valor no models
for config in MODEL_CONFIGS:
    # Ele pega o nome de um item do dicionario, e atribui a uma chave do models, depois, ele recebe o retorno da função _create_chat_model
    # Que basicamente, engloba como função própria para uso de IA junto dos parametros params, que são identificados ao passar aqui os parametros
    models[config["key_name"]] = _create_chat_model(
        model_name=config["model_name"],
        provider=config["provider"],
        temperature=config.get("temperature")
    )