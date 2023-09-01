import numpy as np

def funcaoDegrau(x):
    return 1 if x >= 0 else 0

# TABELA DO XOR
matrizEntrada = np.array([
    [0, 0],
    [1, 1],
    [1, 0],
    [0, 1]
])

# Saídas desejadas correspondentes às entradas
training_outputs = np.array([0, 0, 1, 1])

# Inicialização dos pesos sinápticos com valores aleatórios entre -1 e 1
np.random.seed(1)
pesos = 2 * np.random.random(2) - 1

print('Pesos iniciais aleatórios: \n')
print(pesos, "\n")

# Treinamento do perceptron (várias iterações)
for iteration in range(10000):
    erro_total = 0
    for i in range(len(matrizEntrada)):
        input_layer = matrizEntrada[i]
        output = funcaoDegrau(np.dot(input_layer, pesos))
        erro = training_outputs[i] - output
        pesos += 0.1 * erro * input_layer
        erro_total += erro

    # Verifique se o erro total é zero (todas as saídas corretas)
    if erro_total == 0:
        break

print('Saídas após o treinamento:\n')
for i in range(len(matrizEntrada)):
    input_layer = matrizEntrada[i]
    output = funcaoDegrau(np.dot(input_layer, pesos))
    print(f"Entrada: {input_layer}, Saída: {output}")
