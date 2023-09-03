import numpy as np
import matplotlib.pyplot as plt

matrizEntrada = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

vetorSaida = np.array([0, 1, 1, 0])

def funcaoSigmoide(x):
    return 1 / (1 + np.exp(-x))

pesos_camada_oculta = 2 * np.random.random((2, 4)) - 1
pesos_camada_saida = 2 * np.random.random((4, 1)) - 1

# Taxa de aprendizado
taxa_aprendizagem = 0.1

# Treinamento da rede neural (várias iterações)
for iteration in range(10000):
    for i in range(len(matrizEntrada)):
        entrada = matrizEntrada[i]
        # Propagação direta (cálculo das saídas)
        input_camada_oculta = np.dot(entrada, pesos_camada_oculta)
        output_camada_oculta = funcaoSigmoide(input_camada_oculta)
        input_camada_saida = np.dot(output_camada_oculta, pesos_camada_saida)
        output_camada_saida = funcaoSigmoide(input_camada_saida)
        # Cálculo do erro
        erro_camada_saida = vetorSaida[i] - output_camada_saida
        # Atualização manual dos pesos (sem backpropagation)
        delta_saida = erro_camada_saida * output_camada_saida * (1 - output_camada_saida)
        pesos_camada_saida += taxa_aprendizagem * output_camada_oculta.reshape(4, 1) * delta_saida
        delta_oculta = (delta_saida.dot(pesos_camada_saida.T) * output_camada_oculta * (1 - output_camada_oculta))
        pesos_camada_oculta += taxa_aprendizagem * entrada.reshape(2, 1) * delta_oculta

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

print('\nSaídas após o treinamento (valores arredondados para 0 ou 1):\n')
output_matrix_arredondado = np.round(output_matrix)
for i in range(len(matrizEntrada)):
    print(f"Entrada: {matrizEntrada[i]}, Saída (arredondada): {int(output_matrix_arredondado[i][0])}")


'''
plt.figure(figsize=(8, 6))
plt.scatter(matrizEntrada[:, 0], matrizEntrada[:, 1], c=output_matrix[:, 0], cmap='coolwarm')
plt.title('Multi-Perceptron')
plt.xlabel('Entrada 1')
plt.ylabel('Entrada 2')
plt.axvline(x=0.5, color='k', linestyle='--')
plt.axhline(y=0.5, color='k', linestyle='--')
plt.colorbar()
plt.grid(True)
plt.show()
'''
