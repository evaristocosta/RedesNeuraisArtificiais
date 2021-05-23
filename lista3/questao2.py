import os
import numpy as np
from sklearn.datasets import load_iris, load_wine
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from lista3.elm import ELM


def treino():
    print("Escolha a base de dados:\n"
          "1. IRIS\n"
          "2. WINE\n")
    tipo = input("Digite o número: ")
    inputsPossiveis = ['IRIS', 'WINE']

    # hiperparametros
    neuroniosTeste = [4, 8, 16, 32, 64]
    numFolds = 30
    porcentagem = 0.20

    # cria pastas (se não existirem)
    pastaResultados = 'resultados_q2/'
    if not os.path.isdir(pastaResultados):
        os.mkdir(pastaResultados)

    # carrega dados de treino (400 dados)
    if tipo == '1':
        (entrada, saida) = load_iris(return_X_y=True)
    elif tipo == '2':
        (entrada, saida) = load_wine(return_X_y=True)
    else:
        print("Entrada não reconhecida, terminando...")
        exit()

    saida = saida.reshape(-1, 1)

    # normaliza os dados
    normalizador = MinMaxScaler(feature_range=(-1, 1))
    encoder = OneHotEncoder(sparse=False)

    entradaNormalizada = normalizador.fit_transform(entrada)
    # https://gist.github.com/NiharG15/cd8272c9639941cf8f481a7c4478d525
    saidaNormalizada = encoder.fit_transform(saida)

    entradaTreino, entradaTeste, saidaTreino, saidaTeste = train_test_split(
        entradaNormalizada, saidaNormalizada, test_size=porcentagem, shuffle=True)

    for neuronios in neuroniosTeste:
        acertos = []
        erros = []

        for _ in range(numFolds):
            modelo = ELM(
                entradaNormalizada.shape[1], saidaNormalizada.shape[1], neuronios)
            modelo.train(entradaTreino, saidaTreino)

            predicao = modelo.predict(entradaTeste)

            # transforma em array
            predicao = np.squeeze(np.asarray(predicao))
            predicao = np.argmax(predicao, axis=1)
            saidaCategorica = np.argmax(saidaTeste, axis=1)

            acerto = accuracy_score(saidaCategorica, predicao) * 100
            mse = mean_squared_error(saidaCategorica, predicao)

            acertos.append(acerto)
            erros.append(mse)

        print(
            f'Médias para {neuronios} neurônios:\n Acerto: {np.mean(acertos)}\n MSE: {np.mean(erros)}')
        
        base = inputsPossiveis[int(tipo)-1]
        f = open(f'{pastaResultados}{base}.txt', 'a')
        f.write('------------------\n')
        f.write(f'ELM de {neuronios} neurônios para base {base}\n')
        f.write(f'Médias:\n Acerto: {np.mean(acertos)} (+-{np.std(acertos)})\n MSE: {np.mean(erros)} (+-{np.std(erros)})')
        f.write(f'\n\nTodos acertos: {acertos}')
        f.write(f'\nTodos erros: {erros}')
        f.write('\n------------------\n')
        f.close()



if __name__ == '__main__':
    treino()
