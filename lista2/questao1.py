import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import KFold
from entradas import generate_and, generate_or, generate_xor
from rede import modelo_mlp
from resultados import resultado_acerto, resultado_erro, resultado_fronteira


# Hiperparâmetros da rede e de treino
quantidade_epocas = 500
neuronios = 20
tamanho_lote = 1
taxa_aprendizado = 0.001
paciencia = 10
num_folds = 3

# Escolha de porta lógica
print("Escolha a porta:\n"
      "1. AND\n"
      "2. OR\n"
      "3. XOR\n")
tipo = input("Digite o número: ")
inputs_possiveis = ['AND', 'OR', 'XOR']
nome_arquivo = f'porta{inputs_possiveis[int(tipo)-1]}'

# Carrega dados de treino (400 dados)
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

# Treino usando KFold
kfold = KFold(n_splits=num_folds, shuffle=True)

# Variáveis de controle de iteração
fold_atual = 1
precisao_por_fold = []
erro_por_fold = []
entradas = []
saidas = []
historicos = []

for treino, teste in kfold.split(entrada, saida):
    # Para posterior impressão da fronteira de decisão
    entradas.append(entrada[teste])
    saidas.append(saida[teste])

    modelo = modelo_mlp(2, 1, neuronios, 'sigmoid')
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
    historico = modelo.fit(entrada[treino],
                           saida[treino],
                           batch_size=tamanho_lote,
                           epochs=quantidade_epocas,
                           validation_data=(entrada[teste], saida[teste]),
                           callbacks=lista_callbacks)
    historicos.append(historico)

    # Carrega o melhor modelo para avaliação
    modelo.load_weights(f'mlp_pesos.h5')
    metricas = modelo.evaluate(entrada[teste], saida[teste], verbose=0)
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
for hist in range(num_folds):
    epocas.append(len(historicos[hist].history['val_loss']))

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
resultado_fronteira(
    entradas[melhor_fold], saidas[melhor_fold], modelo, neuronios, nome_arquivo, 1)
resultado_acerto(historicos[melhor_fold], neuronios, nome_arquivo, 1)
resultado_erro(historicos[melhor_fold], neuronios, nome_arquivo, 1)

print("Fim da execução")
