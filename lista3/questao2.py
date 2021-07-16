import numpy as np
from sklearn.datasets import load_iris, load_wine
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from elm import ELM


def treino():
    print("Escolha a base de dados:\n"
          "1. IRIS\n"
          "2. WINE\n")
    tipo = input("Digite o número: ")

    # hiperparametros
    neuronios_teste = [4, 8, 16, 32, 64]
    num_folds = 30
    porcentagem = 0.20

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

    entrada_normalizada = normalizador.fit_transform(entrada)
    # https://gist.github.com/NiharG15/cd8272c9639941cf8f481a7c4478d525
    saida_normalizada = encoder.fit_transform(saida)

    entrada_treino, entrada_teste, saida_treino, saida_teste = train_test_split(
        entrada_normalizada, saida_normalizada, test_size=porcentagem, shuffle=True)

    for neuronios in neuronios_teste:
        acertos = []
        erros = []

        for _ in range(num_folds):
            modelo = ELM(
                entrada_normalizada.shape[1], saida_normalizada.shape[1], neuronios)
            modelo.train(entrada_treino, saida_treino)

            predicao = modelo.predict(entrada_teste)

            # transforma em array
            predicao = np.squeeze(np.asarray(predicao))
            predicao = np.argmax(predicao, axis=1)
            saida_categorica = np.argmax(saida_teste, axis=1)

            acerto = accuracy_score(saida_categorica, predicao) * 100
            mse = mean_squared_error(saida_categorica, predicao)

            acertos.append(acerto)
            erros.append(mse)

        print(
            f'Médias para {neuronios} neurônios:\n Acerto: {np.mean(acertos)}\n MSE: {np.mean(erros)}')


if __name__ == '__main__':
    treino()
