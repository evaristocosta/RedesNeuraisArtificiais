import numpy as np
import matplotlib.pyplot as plt
from hopfield_python import hopfield, differences
from carrega_figuras import livros, numeros, letras


def plot_figuras(linhas, colunas, padroes, hp):
    estilo = 'cividis'
    memorias = len(padroes)
    _, (axs1, axs2, axs3) = plt.subplots(nrows=3, ncols=memorias, sharex=True)

    for indice, padrao in enumerate(padroes):
        axs1[indice].imshow(padrao.reshape(linhas, colunas), cmap=estilo)

        axs2[indice].imshow(hp.noised_img.iloc[indice, :].values.reshape(
            linhas, colunas), cmap=estilo)

        axs3[indice].imshow(hp.outputs.iloc[indice, :].values.reshape(
            linhas, colunas), cmap=estilo)

        axs1[indice].set_xticks([0, (colunas-1)])
        axs1[indice].set_xticklabels([1, (colunas)])
        axs2[indice].set_xticks([0, (colunas-1)])
        axs2[indice].set_xticklabels([1, (colunas)])
        axs3[indice].set_xticks([0, (colunas-1)])
        axs3[indice].set_xticklabels([1, (colunas)])
        axs1[indice].set_yticks([0, (linhas-1)])
        axs1[indice].set_yticklabels([1, (linhas)])
        axs2[indice].set_yticks([0, (linhas-1)])
        axs2[indice].set_yticklabels([1, (linhas)])
        axs3[indice].set_yticks([0, (linhas-1)])
        axs3[indice].set_yticklabels([1, (linhas)])

    plt.show()


def figuras_hp():
    print("Qual conjunto quer verificar?\n"
          "1 - livro.dat\n"
          "2 - numeros.dat\n"
          "3 - letras.dat")
    conjunto = input("Digite o valor: ")

    ruidos = [0.08, 0.1, 0.25, 0.5]

    if conjunto == '1':
        padroes, linhas, colunas = livros()
    elif conjunto == '2':
        padroes, linhas, colunas = numeros()
    elif conjunto == '3':
        padroes, linhas, colunas = letras()

    for ruido in ruidos:
        hp = hopfield(patterns=padroes, noise_percentage=ruido,
                      pattern_n_row=linhas, pattern_n_column=colunas, ib=0, epochs=1000)

        hp.run()

        acertos = []
        erros = []
        for indice, padrao in enumerate(padroes):
            a = padrao
            b = hp.outputs.iloc[indice, :].values

            diferenca = differences(a, b)
            acerto = (len(a) - diferenca)/len(a)
            pixels_distintos = diferenca/len(a)

            acertos.append(acerto)
            erros.append(pixels_distintos)

        print(f'Para ruído de {ruido}:')
        print(f'Acerto médio: {np.mean(acertos)*100}%')
        print(f'Erro médio: {np.mean(erros)*100}%')
        print('---------------------\n')

        plot_figuras(linhas, colunas, padroes, hp)


if __name__ == "__main__":
    figuras_hp()
