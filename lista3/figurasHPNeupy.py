import numpy as np
import matplotlib.pyplot as plt
from neupy.algorithms import DiscreteHopfieldNetwork
from lista3.Hopfield_python import differences
from lista3.carrega_figuras import livros, numeros, letras

def plotFiguras(linhas, colunas, padroes, padroes_ruido, predicoes):
    memorias = len(padroes)
    fig, axs = plt.subplots(nrows=3, ncols=memorias)

    for indice, (padrao, ruido, predicao) in enumerate(zip(padroes, padroes_ruido, predicoes)):
        axs[0][indice].set_title(f'Padrão {indice+1}')
        axs[0][indice].imshow(padrao.reshape(linhas, colunas))

        axs[1][indice].set_title(f'Padrão com ruído {indice+1}')
        #axs[1][indice].imshow(
        #    hp.noised_img.iloc[indice, :].values.reshape(linhas, colunas))
        axs[1][indice].imshow(ruido.reshape(linhas, colunas))

        axs[2][indice].set_title(f'Padrão HP {indice+1}')
        #axs[2][indice].imshow(
        #    hp.outputs.iloc[indice, :].values.reshape(linhas, colunas))
        axs[2][indice].imshow(predicao.reshape(linhas, colunas))

    plt.show()


def noise_attribution(pattern, nrow, ncol, noise):
    dim = len(pattern)
    randM = np.random.rand(nrow, ncol)
    auxA = noise > randM
    auxB = noise < randM
    randM[auxA] = 1
    randM[auxB] = 0
    new_patter = pattern.reshape(nrow, ncol)*randM
    return new_patter.reshape(dim, 1)

def figurasHPNeupy():
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
        padroes[padroes==-1] = 0

        hp = DiscreteHopfieldNetwork(mode='sync', check_limit=False)
        hp.train(padroes)

        # adiciona ruido
        padroes_ruido = []
        for padrao in padroes:
            padroes_ruido.append(noise_attribution(padrao, linhas, colunas, 1-ruido).T[0])
        
        padroes_ruido = np.array(padroes_ruido)
        
        predicoes = hp.predict(padroes_ruido)

        plotFiguras(linhas, colunas, padroes, padroes_ruido, predicoes)

        acertos = []
        erros = []
        print('\n---------------------')
        print(f'Para ruído de {ruido}:')
        for indice, (padrao, predicao) in enumerate(zip(padroes, predicoes)):
            print(f'Imagem {indice+1}')
            a = padrao
            #b = hp.outputs.iloc[indice, :].values
            b = predicao

            diferenca = differences(a, b)
            acerto = (len(a) - diferenca)/len(a)
            pixelsDistintos = diferenca/len(a)

            acertos.append(acerto)
            erros.append(pixelsDistintos)

            print(f'Total de {acerto*100}% de acerto')

        print('\n---------------------')
        print(f'Acerto médio: {np.mean(acertos)*100}%')
        print(f'Erro médio: {np.mean(erros)*100}%')


if __name__ == "__main__":
    figurasHPNeupy()
