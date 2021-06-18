import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import KFold
from entradas import generate_and, generate_or, generate_xor
from rede import modelo_mlp
from resultados import resultado_acerto, resultado_erro, resultado_fronteira


print("Escolha a porta:\n"
      "1. AND\n"
      "2. OR\n"
      "3. XOR\n")
tipo = input("Digite o número: ")
inputs_possiveis = ['AND', 'OR', 'XOR']
nome_arquivo = f'porta{inputs_possiveis[int(tipo)-1]}'

# hiperparametros
quantidade_epocas = 500
tamanho_lote = 1
neuronios = 20
num_folds = 3
paciencia = 10
taxa_aprendizado = 0.001

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
kfold = KFold(n_splits=num_folds, shuffle=True)

fold_atual = 1
precisao_por_fold = []
erro_por_fold = []
entradas = []
saidas = []
historicos = []

for treino, teste in kfold.split(entrada, saida):
    entradas.append(entrada[teste])
    saidas.append(saida[teste])

    modelo = modelo_mlp(2, 1, neuronios, 'sigmoid')

    sgd = SGD(learning_rate=taxa_aprendizado)

    # compilacao do modelo
    modelo.compile(loss='mean_squared_error',
                   optimizer=sgd, metrics=['accuracy'])

    checkpoint = ModelCheckpoint(f'mlp_pesos.h5',
                                 monitor='val_loss', verbose=1,
                                 save_best_only=True, mode='min')

    early_stop = EarlyStopping(monitor='val_loss',
                               mode='min', verbose=1, patience=paciencia)

    lista_callbacks = [checkpoint, early_stop]

    historico = modelo.fit(entrada[treino],
                           saida[treino],
                           batch_size=tamanho_lote,
                           epochs=quantidade_epocas,
                           validation_data=(entrada[teste], saida[teste]),
                           callbacks=lista_callbacks)
    historicos.append(historico)

    # LOAD BEST MODEL to evaluate the performance of the model
    modelo.load_weights(
        f'mlp_pesos.h5')

    metricas = modelo.evaluate(
        entrada[teste], saida[teste], verbose=0)

    print(
        f'Resultados para fold {fold_atual}: {modelo.metrics_names[0]} de {metricas[0]}; {modelo.metrics_names[1]} de {metricas[1]*100}%')

    precisao_por_fold.append(metricas[1] * 100)
    erro_por_fold.append(metricas[0])

    tf.keras.backend.clear_session()

    fold_atual = fold_atual + 1


# salva o modelo
modelo_json = modelo.to_json()
caminho_modelo = f'mlp_neuronios{neuronios}_{nome_arquivo}.json'
open(caminho_modelo, 'w').write(modelo_json)

# médias
melhor_fold = erro_por_fold.index(min(erro_por_fold))

print('\n--------------------------\n')

print('Acerto médio: %.4f (+-%.4f)' %
      (np.mean(precisao_por_fold), np.std(precisao_por_fold)))
print('Erro médio: %.4f (+-%.4f)' %
      (np.mean(erro_por_fold), np.std(erro_por_fold)))

epocas = []
for hist in range(num_folds):
    epocas.append(len(historicos[hist].history['val_loss']))

print('Quantidade de épocas média: %.4f (+-%.4f)' %
      (np.mean(epocas), np.std(epocas)))
print("Maior acerto: %.4f" % (max(precisao_por_fold)))
print("Menor erro: %.4f" % (min(erro_por_fold)))
print("Melhor fold: %i" % (melhor_fold+1))
print("Acerto do melhor fold: %.4f" % (precisao_por_fold[melhor_fold]))
print("Erro do melhor fold: %.4f" % (erro_por_fold[melhor_fold]))
print("Quantidade de épocas do melhor fold: %i" % (np.argmin(
    historicos[melhor_fold].history['val_loss'])+1))

print('\n--------------------------\n')

resultado_fronteira(
    entradas[melhor_fold], saidas[melhor_fold], modelo, neuronios, nome_arquivo, 1)

resultado_acerto(historicos[melhor_fold], neuronios,
                 nome_arquivo, 1)

resultado_erro(historicos[melhor_fold], neuronios,
               nome_arquivo, 1)

print("Fim do treino!")
