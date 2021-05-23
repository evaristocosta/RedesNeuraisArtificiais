import matplotlib.pyplot as plt
import numpy as np

from lista1.perceptron import treinamentoPerceptron

taxaAprendizado = 0.1
repeticoes = 50

entrada = [[1, 0, 0], [1, 0, 1], [1, 1, 0],  [1, 1, 1]]
iteracoes = np.arange(0, repeticoes)

print("AND")
desejado = [-1, -1, -1, 1]
menorEqm = []

fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
fig.suptitle('MSE para cada execução da rede Perceptron')

for i in range(10):
    pesos, eqm = treinamentoPerceptron(
        entrada, taxaAprendizado, repeticoes, desejado)
    ax1.plot(iteracoes, np.array(eqm))
    menorEqm.append(eqm[-1])

print(min(menorEqm))

ax1.set(xlabel='Iterações', ylabel='MSE',
           title='AND')
ax1.grid()


print("OR")
desejado = [1, -1, -1, -1]
menorEqm = []

for i in range(10):
    pesos, eqm = treinamentoPerceptron(
        entrada, taxaAprendizado, repeticoes, desejado)
    ax2.plot(iteracoes, np.array(eqm))
    menorEqm.append(eqm[-1])

print(min(menorEqm))

ax2.set(xlabel='Iterações', 
           title='OR')
ax2.grid()


print("XOR")
desejado = [-1, 1, 1, -1]
menorEqm = []


for i in range(10):
    pesos, eqm = treinamentoPerceptron(
        entrada, taxaAprendizado, repeticoes, desejado)
    ax3.plot(iteracoes, np.array(eqm))
    menorEqm.append(eqm[-1])

print(min(menorEqm))

ax3.set(xlabel='Iterações', 
           title='XOR')
ax3.grid()

plt.show()
