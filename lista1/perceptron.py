import random


def funcao_ativacao_perceptron(x, w):
    ativacao = 0
    for i in range(len(x)):
        ativacao += w[i] * x[i]
    return 1.0 if ativacao > 0.0 else -1.0


def iteracao_treinamento(dados_treino, taxa_aprendizado, desejado, w_n):
    soma_erro = 0.0
    for indice, x_n in enumerate(dados_treino):
        y_n = funcao_ativacao_perceptron(x_n, w_n)
        d_n = desejado[indice]
        erro = d_n - y_n
        soma_erro += erro**2
        for i in range(len(x_n)):
            w_n[i] = w_n[i] + taxa_aprendizado * erro * x_n[i]
    return soma_erro


def treinamento_perceptron(dados_treino, taxa_aprendizado, desejado, repeticoes=0):
    # Iniciação de pesos randômicos
    w_n = [random.uniform(0.0, 1.0) for _ in range(len(dados_treino[0]))]
    # EQM = Erro Quadrático Médio
    eqm = []
    eqm_atual = 1

    print('Pesos iniciais do Perceptron:')
    print(w_n)

    # Caso o critério de parada seja erro igual a 0
    if repeticoes == 0:
        while eqm_atual > 0:
            soma_erro = iteracao_treinamento(
                dados_treino, taxa_aprendizado, desejado, w_n)

            eqm_atual = soma_erro/len(dados_treino)
            eqm.append(eqm_atual)
    # Caso seja por número de repetições
    else:
        for _ in range(repeticoes):
            soma_erro = iteracao_treinamento(
                dados_treino, taxa_aprendizado, desejado, w_n)

            eqm_atual = soma_erro/len(dados_treino)
            eqm.append(eqm_atual)

    return w_n, eqm
