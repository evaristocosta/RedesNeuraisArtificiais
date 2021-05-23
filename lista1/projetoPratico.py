import csv
import numpy as np
from lista1.perceptron import treinamentoPerceptronPrecisao, funcaoAtivacaoPerceptron
from lista1.adaline import treinamentoAdalinePrecisao, funcaoAtivacaoAdaline

taxaAprendizado = 0.01
precisao = 0.4
pesosPerceptron = None
pesosAdaline = None

with open('treino.csv') as csvfile:
    arquivo = csv.reader(csvfile)
    next(arquivo)
    entrada = []
    desejado = []
    for linha in arquivo:
        x = np.array(linha)
        x = np.insert(x, 0, -1)
        x = x.astype(float)
        entrada.append(x[:-1])
        desejado.append(x[-1])
    
    pesosPerceptron, eqmPerceptron = treinamentoPerceptronPrecisao(
        entrada, taxaAprendizado, desejado)
    print("Pesos do Perceptron: ")
    print(pesosPerceptron)
    print("Total de iterações")
    print(len(eqmPerceptron))

    pesosAdaline, eqmAdaline = treinamentoAdalinePrecisao(
        entrada, taxaAprendizado, precisao, desejado)
    print("Pesos do Adaline: ")
    print(pesosAdaline)
    print("Total de iterações")
    print(len(eqmAdaline))



with open('amostras.csv') as csvfile:
    arquivo = csv.reader(csvfile)
    next(arquivo)
    entradas = []

    for linha in arquivo:
        x = np.array(linha)
        x = np.insert(x, 0, -1)
        entradas.append(x.astype(float))

    print("Predição do Perceptron: ")
    for entrada in entradas:
        predicao = funcaoAtivacaoPerceptron(entrada, pesosPerceptron)
        print(predicao)

    print("Predição do Adaline: ")
    for entrada in entradas:
        predicao = funcaoAtivacaoAdaline(entrada, pesosAdaline)
        print(predicao)

