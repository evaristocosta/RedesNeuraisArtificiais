"""
    Resolução da questão 2: projeto prático.
"""

import csv
import numpy as np
from perceptron import treinamento_perceptron, funcao_ativacao_perceptron
from adaline import treinamento_adaline, funcao_ativacao_adaline

taxa_aprendizado = 0.01
precisao = 0.4
pesos_perceptron = None
pesos_adaline = None

# Etapa de treinamento
print("Treinamento")
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
    
    print("---------------------")
    print("Treino do Perceptron:")
    pesos_perceptron, eqm_perceptron = treinamento_perceptron(
        entrada, taxa_aprendizado, desejado=desejado)
    print("Pesos finais do Perceptron: ")
    print(pesos_perceptron)
    print("Total de iterações")
    print(len(eqm_perceptron))

    print("---------------------")
    print("Treino do Adaline:")
    pesos_adaline, eqm_adaline = treinamento_adaline(
        entrada, taxa_aprendizado, precisao=precisao, desejado=desejado)
    print("Pesos finais do Adaline: ")
    print(pesos_adaline)
    print("Total de iterações")
    print(len(eqm_adaline))


# Etapa de predição
print("---------------------")
print("\nResultados")
with open('amostras.csv') as csvfile:
    arquivo = csv.reader(csvfile)
    next(arquivo)
    entradas = []

    for linha in arquivo:
        x = np.array(linha)
        x = np.insert(x, 0, -1)
        entradas.append(x.astype(float))

    print("---------------------")
    print("Predição do Perceptron: ")
    for entrada in entradas:
        predicao = funcao_ativacao_perceptron(entrada, pesos_perceptron)
        print(predicao)

    print("---------------------")
    print("Predição do Adaline: ")
    for entrada in entradas:
        predicao = funcao_ativacao_adaline(entrada, pesos_adaline)
        print(predicao)

