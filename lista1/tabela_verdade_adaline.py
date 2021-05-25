""" 
    Resolução da questão 1: Adaline
"""

import matplotlib.pyplot as plt
import numpy as np

from adaline import treinamento_adaline

# Configurações iniciais
taxa_aprendizado = 0.1
repeticoes = 50

entrada = [[1, 0, 0], [1, 0, 1], [1, 1, 0],  [1, 1, 1]]
iteracoes = np.arange(0, repeticoes)

# Execução para porta AND
print("AND")
desejado = [-1, -1, -1, 1]
menor_eqm = []

fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
fig.suptitle('MSE para cada execução da rede Adaline')

for i in range(10):
    pesos, eqm = treinamento_adaline(
        entrada, taxa_aprendizado, repeticoes=repeticoes, desejado=desejado)
    ax1.plot(iteracoes, np.array(eqm))
    menor_eqm.append(eqm[-1])

print(min(menor_eqm))

ax1.set(xlabel='Iterações', ylabel='MSE', title='AND')
ax1.grid()

# -----------------------
# Execução para porta OR
print("OR")
desejado = [1, -1, -1, -1]
menor_eqm = []

for i in range(10):
    pesos, eqm = treinamento_adaline(
        entrada, taxa_aprendizado, repeticoes=repeticoes, desejado=desejado)
    ax2.plot(iteracoes, np.array(eqm))
    menor_eqm.append(eqm[-1])

print(min(menor_eqm))

ax2.set(xlabel='Iterações',  title='OR')
ax2.grid()

# -----------------------
# Execução para porta XOR
print("XOR")
desejado = [-1, 1, 1, -1]
menor_eqm = []

for i in range(10):
    pesos, eqm = treinamento_adaline(
        entrada, taxa_aprendizado, repeticoes=repeticoes, desejado=desejado)
    ax3.plot(iteracoes, np.array(eqm))
    menor_eqm.append(eqm[-1])

print(min(menor_eqm))

ax3.set(xlabel='Iterações',  title='XOR')
ax3.grid()

# Gráfico final
plt.show()
