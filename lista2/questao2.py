import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import utils
from sklearn.model_selection import KFold
from sklearn.datasets import load_wine, load_iris
from sklearn.preprocessing import MinMaxScaler
from rede import modelo_mlp
from resultados import resultado_erro, resultado_boxplot


# Hiperparâmetros da rede e de treino
quantidade_epocas = 500
neuronios = 32
tamanho_lote = 1
taxa_aprendizado = 0.05
paciencia = 50
num_folds = 10

# Escolha de porta lógica
print("Escolha a base de dados:\n"
      "1. IRIS\n"
      "2. WINE\n")
tipo = input("Digite o número: ")
inputs_possiveis = ['IRIS', 'WINE']
nome_arquivo = f'base{inputs_possiveis[int(tipo)-1]}'


# Carrega dados de treino
if tipo == '1':
    (entrada, saida) = load_iris(return_X_y=True)
elif tipo == '2':
    (entrada, saida) = load_wine(return_X_y=True)
else:
    print("Entrada não reconhecida, terminando...")
    exit()

# Normaliza/processa os dados
normalizador = MinMaxScaler(feature_range=(-1, 1))
entrada_normalizada = normalizador.fit_transform(entrada)
# https://gist.github.com/NiharG15/cd8272c9639941cf8f481a7c4478d525
saida = saida.reshape(-1, 1)
saida_normalizada = utils.to_categorical(saida)

# Treino usando KFold
kfold = KFold(n_splits=num_folds, shuffle=True)

# Variáveis de controle de iteração
fold_atual = 1
precisao_por_fold = []
erro_por_fold = []
historicos = []

for treino, teste in kfold.split(entrada_normalizada, saida_normalizada):
    modelo = modelo_mlp(
        entrada_normalizada.shape[1], saida_normalizada.shape[1], neuronios, 'softmax')
    sgd = SGD(learning_rate=taxa_aprendizado)

    # Compilação do modelo
    # Não é recomendado utilizar essa opção de loss, entretanto era requisito da questão #
    modelo.compile(loss='mean_squared_error',
                   optimizer=sgd, metrics=['accuracy'])

    checkpoint = ModelCheckpoint(f'mlp_pesos.h5',
                                 monitor='val_loss', verbose=1,
                                 save_best_only=True, mode='min')
    criterio_parada = EarlyStopping(
        monitor='val_loss', mode='min', verbose=1, patience=paciencia)

    lista_callbacks = [checkpoint, criterio_parada]

    # Treinamento
    historico = modelo.fit(entrada_normalizada[treino],
                           saida_normalizada[treino],
                           batch_size=tamanho_lote,
                           epochs=quantidade_epocas,
                           validation_data=(
                               entrada_normalizada[teste], saida_normalizada[teste]),
                           callbacks=lista_callbacks)
    historicos.append(historico)

    # Carrega o melhor modelo para avaliação
    modelo.load_weights(f'mlp_pesos.h5')
    metricas = modelo.evaluate(
        entrada_normalizada[teste], saida_normalizada[teste], verbose=0)
    print(
        f'Resultados para fold {fold_atual}: {modelo.metrics_names[0]} de {metricas[0]}; {modelo.metrics_names[1]} de {metricas[1]*100}%')

    # Salva os resultados da iteração
    precisao_por_fold.append(metricas[1] * 100)
    erro_por_fold.append(metricas[0])

    tf.keras.backend.clear_session()
    fold_atual = fold_atual + 1


# Salva o modelo (são todos iguais independente da iteração)
modelo_json = modelo.to_json()
caminho_modelo = f'mlp_neuronios{neuronios}_{nome_arquivo}.json'
open(caminho_modelo, 'w').write(modelo_json)

# Impressão de resultados
melhor_fold = erro_por_fold.index(min(erro_por_fold))
epocas = []
mse = []
for hist in range(num_folds):
    epocas.append(len(historicos[hist].history['val_loss']))
    mse.append(historicos[hist].history['val_loss'])

print('\n--------------------------\n')
print('Acerto médio: %.4f (+-%.4f)' %
      (np.mean(precisao_por_fold), np.std(precisao_por_fold)))
print('Erro médio: %.4f (+-%.4f)' %
      (np.mean(erro_por_fold), np.std(erro_por_fold)))
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

# Resultados gráficos
resultado_erro(historicos[melhor_fold], neuronios, nome_arquivo, 1)
resultado_boxplot(mse, neuronios, nome_arquivo, 1)

print("Fim da execução")
