import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import KFold
from keras import backend as K
from keras.layers import Input, Dense
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import SGD
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions

from lista2.entradas import generate_and, generate_or, generate_xor


def modelo_mlp(dimensaoEntrada, dimensaoSaida, neuronios):
    entrada_camada = Input(
        shape=(dimensaoEntrada), dtype='float32')

    meio = Dense(neuronios, activation='tanh')(entrada_camada)

    saida_camada = Dense(dimensaoSaida, activation='softmax')(meio)

    modelo = Model(inputs=entrada_camada, outputs=saida_camada)

    return modelo


def resultados(historico, neuronios, porta):
    # Resultados do treino
    print("Plotando historio de acerto")
    # Plot valores de acerto de treino e teste
    plt.plot(historico.history['accuracy'])
    plt.plot(historico.history['val_accuracy'])
    plt.title('Acerto do Modelo')
    plt.ylabel('Acerto')
    plt.xlabel('Épocas')
    plt.legend(['Treino', 'Validação'], loc='upper left')
    plt.savefig('resultados/acerto_mpl_neuronios%s_porta%s.png' %
                (neuronios, porta))
    print("Acerto plotado")
    plt.close()

    print("Plotando historio de erro")
    # Plot valores de erro de treino e teste
    plt.plot(historico.history['loss'])
    plt.plot(historico.history['val_loss'])
    plt.title('Erro do Modelo')
    plt.ylabel('MSE')
    plt.xlabel('Épocas')
    plt.legend(['Treino', 'Validação'], loc='upper left')
    plt.savefig('resultados/erro_mlp_neuronios%s_porta%s.png' %
                (neuronios, porta))
    print("Erro plotado")
    plt.close()


def treino():
    print("Escolha a porta:\n"
          "1. AND\n"
          "2. OR\n"
          "3. XOR\n")
    tipo = input("Digite o número: ")
    inputsPossiveis = ['AND', 'OR', 'XOR']

    # hiperparametros
    quantidadeEpocas = 500
    tamanhoLote = 1
    neuronios = 200
    numFolds = 10
    taxaAprendizado = 0.01

    # cria pastas (se não existirem)
    pastaResultados = 'resultados/'
    if not os.path.isdir(pastaResultados):
        os.mkdir(pastaResultados)

    pastaPesos = 'pesos_modelo/'
    if not os.path.isdir(pastaPesos):
        os.mkdir(pastaPesos)

    pastaModelo = 'json_modelo/'
    if not os.path.isdir(pastaModelo):
        os.mkdir(pastaModelo)

    # carrega dados de treino (400 dados)
    if tipo == '1':
        entrada, saida = generate_and(100)
    elif tipo == '2':
        entrada, saida = generate_or(100)
    elif tipo == '3':
        entrada, saida = generate_xor(100)
    else:
        print("Entrada não reconhecida, terminando...")
        exit()

    entrada = np.array(entrada)
    saida = np.array(saida)

    # treino usando KFold
    if numFolds > 1:
        kfold = KFold(n_splits=numFolds, shuffle=True)

        foldAtual = 1
        precisaoPorFold = []
        erroPorFold = []
        historicos = []

        for treino, teste in kfold.split(entrada, saida):
            modelo = modelo_mlp(2, 1, neuronios)

            sgd = SGD(learning_rate=taxaAprendizado)

            # compilacao do modelo
            modelo.compile(loss='mean_squared_error',
                           optimizer=sgd, metrics=['accuracy'])

            checkpoint = ModelCheckpoint(f'{pastaPesos}mlp_fold{foldAtual}_neuronios{neuronios}_porta{inputsPossiveis[int(tipo)-1]}.h5',
                                         monitor='val_loss', verbose=1,
                                         save_best_only=True, mode='min')

            earlyStop = EarlyStopping(monitor='val_loss',
                                      mode='min', verbose=1, patience=10)

            listaCallbacks = [checkpoint, earlyStop]

            historico = modelo.fit(entrada[treino],
                                   saida[treino],
                                   batch_size=tamanhoLote,
                                   epochs=quantidadeEpocas,
                                   validation_data=(
                                       entrada[teste], saida[teste]),
                                   callbacks=listaCallbacks)
            historicos.append(historico)

            # LOAD BEST MODEL to evaluate the performance of the model
            modelo.load_weights(
                f'{pastaPesos}mlp_fold{foldAtual}_neuronios{neuronios}_porta{inputsPossiveis[int(tipo)-1]}.h5')

            metricas = modelo.evaluate(
                entrada[teste], saida[teste], verbose=0)
            print(
                f'Resultados para fold {foldAtual}: {modelo.metrics_names[0]} de {metricas[0]}; {modelo.metrics_names[1]} de {metricas[1]*100}%')
            precisaoPorFold.append(metricas[1] * 100)
            erroPorFold.append(metricas[0])

            # salva o modelo
            modeloJson = modelo.to_json()
            caminhoModelo = f'{pastaModelo}mlp_fold{foldAtual}_neuronios{neuronios}_porta{inputsPossiveis[int(tipo)-1]}.json'
            open(caminhoModelo, 'w').write(modeloJson)

            # fronteira de decisao
            plot_decision_regions(entrada[teste], np.reshape(
                saida[teste], (saida[teste].shape[0])), clf=modelo, legend=2)
            plt.xlim(-0.1, 1.1)
            plt.ylim(-0.1, 1.1)
            plt.title('Fronteira de decisão')
            plt.ylabel('X1')
            plt.xlabel('X2')
            plt.savefig(
                f'{pastaResultados}/fronteira_mlp_fold{foldAtual}_neuronios{neuronios}_porta{inputsPossiveis[int(tipo)-1]}.png')
            plt.close()

            tf.keras.backend.clear_session()

            foldAtual = foldAtual + 1

        # médias
        melhorFold = erroPorFold.index(min(erroPorFold))
        print('\n--------------------------\n')

        print('Acerto médio: %.4f (+-%.4f)' %
              (np.mean(precisaoPorFold), np.std(precisaoPorFold)))
        print('Erro médio: %.4f (+-%.4f)' %
              (np.mean(erroPorFold), np.std(erroPorFold)))

        epocas = []
        for hist in range(numFolds):
            epocas.append(len(historicos[hist].history['val_loss']))

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
                   inputsPossiveis[int(tipo)-1])

    else:
        modelo = modelo_mlp(2, 1, neuronios)

        sgd = SGD(learning_rate=taxaAprendizado)

        modelo.compile(loss='mean_squared_error',
                       optimizer=sgd, metrics=['accuracy'])

        checkpoint = ModelCheckpoint(f'{pastaPesos}mlp_neuronios{neuronios}_porta{inputsPossiveis[int(tipo)-1]}.h5',
                                     monitor='val_loss', verbose=1,
                                     save_best_only=True, mode='min')

        earlyStop = EarlyStopping(monitor='val_loss',
                                  mode='min', verbose=1, patience=10)

        listaCallbacks = [checkpoint, earlyStop]

        historico = modelo.fit(entrada,
                               saida,
                               batch_size=tamanhoLote,
                               epochs=quantidadeEpocas,
                               validation_split=0.25,
                               callbacks=listaCallbacks)

        # LOAD BEST MODEL to evaluate the performance of the model
        modelo.load_weights(
            f'{pastaPesos}mlp_neuronios{neuronios}_porta{inputsPossiveis[int(tipo)-1]}.h5')

        metricas = modelo.evaluate(
            entrada, saida, verbose=0)
        print(
            f'Resultados: {modelo.metrics_names[0]} de {metricas[0]}; {modelo.metrics_names[1]} de {metricas[1]*100}%')

        # salva o modelo
        modeloJson = modelo.to_json()
        caminhoModelo = f'{pastaModelo}mlp_neuronios{neuronios}_porta{inputsPossiveis[int(tipo)-1]}.json'
        open(caminhoModelo, 'w').write(modeloJson)

        resultados(historico, neuronios, inputsPossiveis[int(tipo)-1])

        plot_decision_regions(entrada, np.reshape(
            saida, (saida.shape[0])), clf=modelo, legend=2)
        plt.xlim(-0.1, 1.1)
        plt.ylim(-0.1, 1.1)
        plt.title('Fronteira de decisão')
        plt.ylabel('X1')
        plt.xlabel('X2')
        plt.savefig(
            f'{pastaResultados}/fronteira_mlp_neuronios{neuronios}_porta{inputsPossiveis[int(tipo)-1]}.png')
        plt.close()

    print("Fim do treino!")


if __name__ == '__main__':
    treino()
