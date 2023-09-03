import random

def funcaoDegrau(x):
    return 1 if x >= 0 else 0

matrizEntrada = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
]

vetorSaida = [0, 1, 1, 0]

pesos = [2 * random.random() - 1, 2 * random.random() - 1]

print('Pesos iniciais aleatórios: \n')
print(pesos, "\n")

iteration = 0

for iteration in range(10000):
    erro_total = 0
    for i in range(len(matrizEntrada)):
        entrada = matrizEntrada[i]
        saida = funcaoDegrau(sum(x * w for x, w in zip(entrada, pesos)))
        erro = vetorSaida[i] - saida
        pesos = [w + 0.1 * erro * x for x, w in zip(entrada, pesos)]
        erro_total += erro
    if erro_total == 0:
        break

print('Saídas após o treinamento:\n')
for i in range(len(matrizEntrada)):
    entrada = matrizEntrada[i]
    saida = funcaoDegrau(sum(x * w for x, w in zip(entrada, pesos)))
    print(f"Entrada: {entrada}, Saída: {saida}")
