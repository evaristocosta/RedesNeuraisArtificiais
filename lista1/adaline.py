import random


def funcaoAtivacaoAdaline(x, w):
    ativacao = 0
    for i in range(len(x)):
        ativacao += w[i] * x[i]
    return 1.0 if ativacao > 0.0 else -1.0


def soma(x, w):
    soma = 0
    for i in range(len(x)):
        soma += w[i] * x[i]
    return soma


def treinamentoAdaline(dados_treino, taxaAprendizado, repeticoes, desejado):
    #w_n = [0.0 for i in range(len(dados_treino[0]))]
    eqm = []
    w_n = [random.uniform(0.0, 1.0) for i in range(len(dados_treino[0]))]
    for i in range(repeticoes):
        soma_erro = 0.0
        for indice, x_n in enumerate(dados_treino):
            y_n = soma(x_n, w_n)
            d_n = desejado[indice]
            erro = d_n - y_n
            soma_erro += erro**2
            for i in range(len(x_n)):
                w_n[i] = w_n[i] + taxaAprendizado * erro * x_n[i]

        eqm.append(soma_erro/len(dados_treino))
        """ print('> iteracao=%d, erro=%.3f' %
              (iteracao, eqm[-1])) """
    return w_n, eqm


def treinamentoAdalinePrecisao(dados_treino, taxaAprendizado, precisao, desejado):
    eqm = []
    eqmAtual = 1
    w_n = [random.uniform(0.0, 1.0) for i in range(len(dados_treino[0]))]

    print('Pesos iniciais do Adaline:')
    print(w_n)

    while eqmAtual > precisao:
        soma_erro = 0.0
        for indice, x_n in enumerate(dados_treino):
            y_n = soma(x_n, w_n)
            d_n = desejado[indice]
            erro = d_n - y_n
            soma_erro += erro**2
            for i in range(len(x_n)):
                w_n[i] = w_n[i] + taxaAprendizado * erro * x_n[i]

        eqmAtual = soma_erro/len(dados_treino)
        eqm.append(eqmAtual)

    return w_n, eqm
