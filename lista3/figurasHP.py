import os
import numpy as np
import matplotlib.pyplot as plt
from lista3.Hopfield_python import hopfield, differences
from lista3.carrega_figuras import livros, numeros, letras


def plotFiguras(linhas, colunas, padroes, hp, ruido, nomeArquivo, pastaResultados):
    estilo = 'cividis'
    memorias = len(padroes)
    fig1, axs1 = plt.subplots(nrows=1, ncols=memorias,
                              sharey=True, tight_layout=True, figsize=(memorias-1, 1.5))
    fig2, axs2 = plt.subplots(nrows=1, ncols=memorias,
                              sharey=True, tight_layout=True, figsize=(memorias-1, 1.5))
    fig3, axs3 = plt.subplots(nrows=1, ncols=memorias,
                              sharey=True, tight_layout=True, figsize=(memorias-1, 1.5))


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

    fig1.savefig(f'{pastaResultados}{nomeArquivo}_{ruido}_padrao.png')
    fig2.savefig(f'{pastaResultados}{nomeArquivo}_{ruido}_ruido.png')
    fig3.savefig(f'{pastaResultados}{nomeArquivo}_{ruido}_hp.png')


def figurasHP():
    pastaResultados = 'resultados_q1/'
    if not os.path.isdir(pastaResultados):
        os.mkdir(pastaResultados)

    print("Qual conjunto quer verificar?\n"
          "1 - livro.dat\n"
          "2 - numeros.dat\n"
          "3 - letras.dat")
    conjunto = input("Digite o valor: ")
    conjuntosPossiveis = ['livro', 'numeros', 'letras']

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
            pixelsDistintos = diferenca/len(a)

            acertos.append(acerto)
            erros.append(pixelsDistintos)

        nomeArquivo = conjuntosPossiveis[int(conjunto)-1]
        f = open(f'{pastaResultados}{nomeArquivo}.txt', 'a')
        f.write(f'Para ruído de {ruido}:\n')
        f.write(f'Todos acertos: {np.array(acertos)*100}\n')
        f.write(f'Acerto médio: {np.mean(acertos)*100}%\n')
        f.write(f'Erro médio: {np.mean(erros)*100}%\n')
        f.write('---------------------\n')
        f.close()

        plotFiguras(linhas, colunas, padroes, hp,
                    ruido, nomeArquivo, pastaResultados)

        print(f'Para ruído de {ruido}:')
        print(f'Todos acertos: {np.array(acertos)*100}')
        print(f'Acerto médio: {np.mean(acertos)*100}%')
        print(f'Erro médio: {np.mean(erros)*100}%')
        print('---------------------\n')


if __name__ == "__main__":
    figurasHP()
