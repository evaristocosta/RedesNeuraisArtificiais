from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model


def modelo_mlp(dimensao_entrada, dimensao_saida, neuronios, ativacao):
    ativacoes = ['sigmoid', 'softmax']

    entrada_camada = Input(
        shape=(dimensao_entrada), dtype='float32')

    meio = Dense(neuronios, activation='tanh')(entrada_camada)

    saida_camada = Dense(dimensao_saida, activation=ativacao)(meio)

    modelo = Model(inputs=entrada_camada, outputs=saida_camada)

    return modelo
