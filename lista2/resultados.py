import numpy as np
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions

""" 
    Funções para impressão de resultados gráficos
"""

# Gráfico de acerto (accuracy) por época (epoch)
def resultado_acerto(historico, neuronios, nome_arquivo, mostrar):
    # Plot valores de acerto de treino e teste
    plt.plot(historico.history['accuracy'])
    plt.plot(historico.history['val_accuracy'])
    # Critério de parada
    menor_erro = np.argmin(historico.history['val_loss'])
    plt.axvline(x=menor_erro, color='red', linestyle='dashed')
    # Metadados
    plt.title('Acerto do Modelo')
    plt.ylabel('Acerto')
    plt.xlabel('Épocas')
    plt.legend(['Treino', 'Validação', 'Menor Erro'], loc='upper left')

    # Salva ou mostra a figura
    if mostrar == 1:
        plt.show()
    else:
        arquivo = f'acerto_mlp_{neuronios}neuronios_{nome_arquivo}.png'
        plt.savefig(arquivo)

    plt.close()


# Grafíco de erro (nesse caso, MSE) por época
def resultado_erro(historico, neuronios, nome_arquivo, mostrar):
    # Plot valores de erro de treino e teste
    plt.plot(historico.history['loss'])
    plt.plot(historico.history['val_loss'])
    # Critério de parada
    menor_erro = np.argmin(historico.history['val_loss'])
    plt.axvline(x=menor_erro, color='red', linestyle='dashed')
    # Metadados
    plt.title('Erro do Modelo')
    plt.ylabel('MSE')
    plt.xlabel('Épocas')
    plt.legend(['Treino', 'Validação', 'Menor Erro'], loc='upper right')

    # Salva ou mostra a figura
    if mostrar == 1:
        plt.show()
    else:
        arquivo = f'erro_mlp_{neuronios}neuronios_{nome_arquivo}.png'
        plt.savefig(arquivo)

    plt.close()


# Boxplot de variação de MSE para cada rodada
def resultado_boxplot(mse, neuronios, nome_arquivo, mostrar):
    plt.boxplot(mse, patch_artist=True, showfliers=False)
    # Metadados
    plt.title('MSE por rodada')
    plt.ylabel('MSE')
    plt.xlabel('Rodada')

    # Salva ou mostra a figura
    if mostrar == 1:
        plt.show()
    else:
        arquivo = f'boxplot_mlp_{neuronios}neuronios_{nome_arquivo}.png'
        plt.savefig(arquivo)

    plt.close()


# Gráfico da fronteira de decisão
def resultado_fronteira(entradas, saidas, modelo, neuronios, nome_arquivo, mostrar):
    plot_decision_regions(entradas, np.reshape(
        saidas, (saidas.shape[0])), clf=modelo, legend=2)
    plt.xlim(-0.1, 1.1)
    plt.ylim(-0.1, 1.1)
    # Metadados
    plt.title('Fronteira de decisão')
    plt.ylabel('X1')
    plt.xlabel('X2')

    # Salva ou mostra a figura
    if mostrar == 1:
        plt.show()
    else:
        arquivo = f'fronteira_mlp_neuronios{neuronios}_{nome_arquivo}.png'
        plt.savefig(arquivo)

    plt.close()
