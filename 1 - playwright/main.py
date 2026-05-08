from playwright.sync_api import sync_playwright
# O playwright não precisa do time, mas aqui eu coloquei, só para conseguir entender melhor o processo de execução
import time

# O playwright consegue abrir uma página e indicar ao clique a localização de um elemento
# para isso, tem que escrever no terminal: python -m playwright codegen <site>, ai vai na opção pick locator e clica no elemento

# Como eu criei o contexto, eu posso pedir ações de forma independente para cada uma
with sync_playwright() as pw:
    navegador = pw.chromium.launch(headless=False)

    # Criando contexto, para o caso de ter que rodar mais de uma página por vez
    # pagina = navegador.new_page()
    contexto = navegador.new_context()
    pagina = contexto.new_page()

    pagina.goto("https://www.hashtagtreinamentos.com/")

    # Eu poderia ter colocado o click no fim da linha 15 tbm
    botao = pagina.get_by_role("link", name="Quero aprender").first
    # A linha acima foi usada para exemplo, que vai ser demonstrado abaixo
    # Aqui eu vou armazenar a página que foi clicada em uma variavel, e depois vou conseguir manipular ela normalmente
    with contexto.expect_page() as pagina2_info:
        botao.click()

    # Então depois do contexto ser criado, eu posso vir e começar a acessar as info resultantes da pagina 2
    pagina2 = pagina2_info.value
    
    # A página primeiro vai ser direcionada para o resultado do clique do botão que o contexto recebeu na parte acima
    pagina2.goto("https://www.hashtagtreinamentos.com/curso-python?src=lespy-site&utm_source=site&utm_medium=header&utm_content=link-header-cursos&utm_campaign=programacao")

    # Preenchendo formulário
    # Se for na página do codegen e apenas ir clicando nos campos e digitando os valores, ele já gera o código completo, vai ficar na parte de cima do nageador, junto de todos os códigos
    pagina2.get_by_role("textbox", name="Seu primeiro nome").fill("Pedro")

    time.sleep(4)
    navegador.close()