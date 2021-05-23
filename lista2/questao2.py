import os
import numpy as np
import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import SGD
from keras import utils
from sklearn.model_selection import KFold
from sklearn.datasets import load_wine, load_iris
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

from lista2.rede import modelo_mlp


def resultados(historico, neuronios, base, mse):
    print("Plotando historio de erro")
    menorErro = np.argmin(historico.history['val_loss'])
    # Plot valores de erro de treino e teste
    plt.plot(historico.history['loss'])
    plt.plot(historico.history['val_loss'])
    plt.axvline(x=menorErro, color='red', linestyle='dashed')
    plt.title('Erro do Modelo')
    plt.ylabel('MSE')
    plt.xlabel('Épocas')
    plt.legend(['Treino', 'Validação', 'Menor erro'], loc='upper right')
    plt.savefig('resultados_q2/erro_mlp_neuronios%s_base%s.svg' %
                (neuronios, base))
    print("Erro plotado")
    plt.close()

    print("Plotando boxplot")
    plt.boxplot(mse, patch_artist=True, showfliers=False)
    plt.title('MSE por rodada')
    plt.ylabel('MSE')
    plt.xlabel('Rodada')
    plt.savefig('resultados_q2/boxplot_mlp_neuronios%s_base%s.svg' %
                (neuronios, base))
    plt.close()



def treino():
    print("Escolha a base de dados:\n"
          "1. IRIS\n"
          "2. WINE\n")
    tipo = input("Digite o número: ")
    inputsPossiveis = ['IRIS', 'WINE']

    # hiperparametros
    quantidadeEpocas = 500
    tamanhoLote = 1
    neuronios = 32
    numFolds = 10
    paciencia = 50
    taxaAprendizado = 0.05

    # cria pastas (se não existirem)
    pastaResultados = 'resultados_q2/'
    if not os.path.isdir(pastaResultados):
        os.mkdir(pastaResultados)

    pastaPesos = 'pesos_modelo_q2/'
    if not os.path.isdir(pastaPesos):
        os.mkdir(pastaPesos)

    pastaModelo = 'json_modelo_q2/'
    if not os.path.isdir(pastaModelo):
        os.mkdir(pastaModelo)

    # carrega dados de treino (400 dados)
    if tipo == '1':
        (entrada, saida) = load_iris(return_X_y=True)
    elif tipo == '2':
        (entrada, saida) = load_wine(return_X_y=True)
    else:
        print("Entrada não reconhecida, terminando...")
        exit()

    # normaliza os dados
    normalizador = MinMaxScaler(feature_range=(-1, 1))
    entradaNormalizada = normalizador.fit_transform(entrada)
    # https://gist.github.com/NiharG15/cd8272c9639941cf8f481a7c4478d525
    saida = saida.reshape(-1, 1)
    saidaNormalizada = utils.to_categorical(saida)

    # treino usando KFold
    if numFolds > 1:
        kfold = KFold(n_splits=numFolds, shuffle=True)

        foldAtual = 1
        precisaoPorFold = []
        erroPorFold = []
        historicos = []

        for treino, teste in kfold.split(entradaNormalizada, saidaNormalizada):
            modelo = modelo_mlp(entradaNormalizada.shape[1], saidaNormalizada.shape[1], neuronios)

            sgd = SGD(learning_rate=taxaAprendizado)

            # compilacao do modelo
            modelo.compile(loss='mean_squared_error',
                           optimizer=sgd, metrics=['accuracy'])

            checkpoint = ModelCheckpoint(f'{pastaPesos}mlp_fold{foldAtual}_neuronios{neuronios}_base{inputsPossiveis[int(tipo)-1]}.h5',
                                         monitor='val_loss', verbose=1,
                                         save_best_only=True, mode='min')

            earlyStop = EarlyStopping(monitor='val_loss',
                                      mode='min', verbose=1, patience=paciencia)

            listaCallbacks = [checkpoint, earlyStop]

            historico = modelo.fit(entradaNormalizada[treino],
                                   saidaNormalizada[treino],
                                   batch_size=tamanhoLote,
                                   epochs=quantidadeEpocas,
                                   validation_data=(
                                       entradaNormalizada[teste], saidaNormalizada[teste]),
                                   callbacks=listaCallbacks)
            historicos.append(historico)

            # LOAD BEST MODEL to evaluate the performance of the model
            modelo.load_weights(
                f'{pastaPesos}mlp_fold{foldAtual}_neuronios{neuronios}_base{inputsPossiveis[int(tipo)-1]}.h5')

            metricas = modelo.evaluate(
                entradaNormalizada[teste], saidaNormalizada[teste], verbose=0)
            print(
                f'Resultados para fold {foldAtual}: {modelo.metrics_names[0]} de {metricas[0]}; {modelo.metrics_names[1]} de {metricas[1]*100}%')
            precisaoPorFold.append(metricas[1] * 100)
            erroPorFold.append(metricas[0])

            # salva o modelo
            modeloJson = modelo.to_json()
            caminhoModelo = f'{pastaModelo}mlp_fold{foldAtual}_neuronios{neuronios}_base{inputsPossiveis[int(tipo)-1]}.json'
            open(caminhoModelo, 'w').write(modeloJson)

            tf.keras.backend.clear_session()

            foldAtual = foldAtual + 1

        # médias
        melhorFold = erroPorFold.index(min(erroPorFold))
        print('\n--------------------------\n')

        print('Acerto médio: %.4f (+-%.4f)' %
              (np.mean(precisaoPorFold), np.std(precisaoPorFold)))
        print('Erro médio: %.4f (+-%.4f)' %
              (np.mean(erroPorFold), np.std(erroPorFold)))
        print('Acertos: ', precisaoPorFold)
        print('Erros: ', erroPorFold)

        epocas = []
        mse = []
        for hist in range(numFolds):
            epocas.append(len(historicos[hist].history['val_loss']))
            mse.append(historicos[hist].history['val_loss'])

        print('Quantidade de épocas média: %.4f (+-%.4f)' %
              (np.mean(epocas), np.std(epocas)))
        print("Maior acerto: %.4f" % (max(precisaoPorFold)))
        print("Menor erro: %.4f" % (min(erroPorFold)))
        print("Melhor fold: %i" % (melhorFold+1))
        print("Acerto do melhor fold: %.4f" % (precisaoPorFold[melhorFold]))
        print("Erro do melhor fold: %.4f" % (erroPorFold[melhorFold]))
        print("Quantidade de épocas do melhor fold: %i" % (np.argmin(
            historicos[melhorFold].history['val_loss'])+1))

        print('\n--------------------------\n')

        resultados(historicos[melhorFold], neuronios,
                   inputsPossiveis[int(tipo)-1], mse)

    else:
        modelo = modelo_mlp(
            entradaNormalizada.shape[1], saidaNormalizada.shape[1], neuronios)

        sgd = SGD(learning_rate=taxaAprendizado)

        modelo.compile(loss='mean_squared_error',
                       optimizer=sgd, metrics=['accuracy'])

        checkpoint = ModelCheckpoint(f'{pastaPesos}mlp_neuronios{neuronios}_base{inputsPossiveis[int(tipo)-1]}.h5',
                                     monitor='val_loss', verbose=1,
                                     save_best_only=True, mode='min')

        earlyStop = EarlyStopping(monitor='val_loss',
                                  mode='min', verbose=1, patience=paciencia)

        listaCallbacks = [checkpoint, earlyStop]

        historico = modelo.fit(entradaNormalizada,
                               saidaNormalizada,
                               batch_size=tamanhoLote,
                               epochs=quantidadeEpocas,
                               validation_split=0.3,
                               callbacks=listaCallbacks)

        # LOAD BEST MODEL to evaluate the performance of the model
        modelo.load_weights(
            f'{pastaPesos}mlp_neuronios{neuronios}_base{inputsPossiveis[int(tipo)-1]}.h5')

        metricas = modelo.evaluate(
            entradaNormalizada, saidaNormalizada, verbose=0)
        print(
            f'Resultados: {modelo.metrics_names[0]} de {metricas[0]}; {modelo.metrics_names[1]} de {metricas[1]*100}%')

        # salva o modelo
        modeloJson = modelo.to_json()
        caminhoModelo = f'{pastaModelo}mlp_neuronios{neuronios}_base{inputsPossiveis[int(tipo)-1]}.json'
        open(caminhoModelo, 'w').write(modeloJson)

        resultados(historico, neuronios,
                   inputsPossiveis[int(tipo)-1], historico.history['val_loss'])

    print("Fim do treino!")


if __name__ == '__main__':
    treino()
