import numpy as np
import matplotlib.pyplot as plt


# Função sigmoide para ativação
def funcaoSigmoide(x):
    return 1 / (1 + np.exp(-x))

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
pesos_camada_oculta = 2 * np.random.random((2, 4)) - 1
pesos_camada_saida = 2 * np.random.random((4, 1)) - 1

# Taxa de aprendizado
learning_rate = 0.1

# Treinamento da rede neural (várias iterações)
for iteration in range(10000):
    for i in range(len(matrizEntrada)):
        input_layer = matrizEntrada[i]

        # Propagação direta (cálculo das saídas)
        input_camada_oculta = np.dot(input_layer, pesos_camada_oculta)
        output_camada_oculta = funcaoSigmoide(input_camada_oculta)

        input_camada_saida = np.dot(output_camada_oculta, pesos_camada_saida)
        output_camada_saida = funcaoSigmoide(input_camada_saida)

        # Cálculo do erro
        erro_camada_saida = training_outputs[i] - output_camada_saida

        # Atualização manual dos pesos (sem backpropagation)
        delta_saida = erro_camada_saida * output_camada_saida * (1 - output_camada_saida)
        pesos_camada_saida += learning_rate * output_camada_oculta.reshape(4, 1) * delta_saida

        delta_oculta = (delta_saida.dot(pesos_camada_saida.T) * output_camada_oculta * (1 - output_camada_oculta))
        pesos_camada_oculta += learning_rate * input_layer.reshape(2, 1) * delta_oculta

# Teste com os dados de treinamento
print('Saídas após o treinamento:\n')
for i in range(len(matrizEntrada)):
    input_layer = matrizEntrada[i]

    input_camada_oculta = np.dot(input_layer, pesos_camada_oculta)
    output_camada_oculta = funcaoSigmoide(input_camada_oculta)

    input_camada_saida = np.dot(output_camada_oculta, pesos_camada_saida)
    output_camada_saida = funcaoSigmoide(input_camada_saida)

    print(f"Entrada: {input_layer}, Saída: {output_camada_saida[0]}")

# Inicializar uma matriz para armazenar as saídas
output_matrix = np.zeros((len(matrizEntrada), 1))

# Teste com os dados de treinamento e armazene as saídas
print('Saídas após o treinamento:\n')
for i in range(len(matrizEntrada)):
    input_layer = matrizEntrada[i]

    input_camada_oculta = np.dot(input_layer, pesos_camada_oculta)
    output_camada_oculta = funcaoSigmoide(input_camada_oculta)

    input_camada_saida = np.dot(output_camada_oculta, pesos_camada_saida)
    output_camada_saida = funcaoSigmoide(input_camada_saida)

    output_matrix[i] = output_camada_saida[0]

    print(f"Entrada: {input_layer}, Saída: {output_camada_saida[0]}")

# Plotar um gráfico de dispersão com cores mapeadas para as saídas
plt.figure(figsize=(8, 6))
plt.scatter(matrizEntrada[:, 0], matrizEntrada[:, 1], c=output_matrix[:, 0], cmap='coolwarm')
plt.title('Saídas após o Treinamento (Multi-Perceptron)')
plt.xlabel('Entrada 1')
plt.ylabel('Entrada 2')
plt.colorbar()
plt.grid(True)
plt.show()
