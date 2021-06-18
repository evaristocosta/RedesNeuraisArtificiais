import numpy as np
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions


def resultado_acerto(historico, neuronios, nome_arquivo, mostrar):
    menor_erro = np.argmin(historico.history['val_loss'])

    # Plot valores de acerto de treino e teste
    plt.plot(historico.history['accuracy'])
    plt.plot(historico.history['val_accuracy'])
    # Melhor resultado e critério de parada
    plt.axvline(x=menor_erro, color='red', linestyle='dashed')

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


def resultado_erro(historico, neuronios, nome_arquivo, mostrar):
    menor_erro = np.argmin(historico.history['val_loss'])

    # Plot valores de erro de treino e teste
    plt.plot(historico.history['loss'])
    plt.plot(historico.history['val_loss'])
    # Melhor resultado e critério de parada
    plt.axvline(x=menor_erro, color='red', linestyle='dashed')

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


def resultado_boxplot(mse, neuronios, nome_arquivo, mostrar):
    # Plot de boxplot de MSE
    plt.boxplot(mse, patch_artist=True, showfliers=False)

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


def resultado_fronteira(entradas, saidas, modelo, neuronios, nome_arquivo, mostrar):
    # fronteira de decisao
    plot_decision_regions(entradas, np.reshape(
        saidas, (saidas.shape[0])), clf=modelo, legend=2)
    plt.xlim(-0.1, 1.1)
    plt.ylim(-0.1, 1.1)
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
