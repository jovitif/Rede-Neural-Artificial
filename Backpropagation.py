import numpy as np
import matplotlib.pyplot as plt

# Função sigmoide para ativação
def funcaoSigmoide(x):
    return 1 / (1 + np.exp(-x))

# Derivada da função sigmoide
def derivadaSigmoide(x):
    return x * (1 - x)

# TABELA DO XOR
matrizEntrada = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

# Saídas desejadas correspondentes às entradas
vetorSaida = np.array([[0, 1, 1, 0]]).T

# Inicialização dos pesos sinápticos com valores aleatórios entre -1 e 1 para as camadas oculta e de saída
pesos_camada_oculta = 2 * np.random.random((2, 4)) - 1
pesos_camada_saida = 2 * np.random.random((4, 1)) - 1

# Taxa de aprendizado
taxa_aprendizagem = 0.1

# Treinamento da rede neural (várias iterações)
for iteration in range(10000):
    # Camada oculta
    input_camada_oculta = np.dot(matrizEntrada, pesos_camada_oculta)
    output_camada_oculta = funcaoSigmoide(input_camada_oculta)

    # Camada de saída
    input_camada_saida = np.dot(output_camada_oculta, pesos_camada_saida)
    output_camada_saida = funcaoSigmoide(input_camada_saida)

    # Cálculo do erro
    erro_camada_saida = vetorSaida - output_camada_saida

    # Cálculo dos deltas (gradientes)
    delta_saida = erro_camada_saida * derivadaSigmoide(output_camada_saida)
    delta_oculta = delta_saida.dot(pesos_camada_saida.T) * derivadaSigmoide(output_camada_oculta)

    # Atualização dos pesos com base nos deltas (backpropagation)
    pesos_camada_saida += output_camada_oculta.T.dot(delta_saida) * taxa_aprendizagem
    pesos_camada_oculta += matrizEntrada.T.dot(delta_oculta) * taxa_aprendizagem

# Teste com os dados de treinamento
print('Saídas após o treinamento (valores reais):\n')
saidas_reais = []
for i in range(len(matrizEntrada)):
    input_layer = matrizEntrada[i]
    input_camada_oculta = np.dot(input_layer, pesos_camada_oculta)
    output_camada_oculta = funcaoSigmoide(input_camada_oculta)
    input_camada_saida = np.dot(output_camada_oculta, pesos_camada_saida)
    output_camada_saida = funcaoSigmoide(input_camada_saida)
    saidas_reais.append(output_camada_saida[0])
    print(f"Entrada: {input_layer}, Saída: {output_camada_saida[0]}")

print('\nSaídas após o treinamento (valores arredondados para 0 ou 1):\n')
saidas_arredondadas = []
for saida_real in saidas_reais:
    saida_arredondada = round(saida_real)
    saidas_arredondadas.append(saida_arredondada)
    print(f"Saída arredondada: {saida_arredondada}")









'''
# Converter as saídas para um formato que pode ser usado para plotagem
saidas = np.array(saidas)
# Plotar os pontos de saída
plt.figure(figsize=(8, 6))
plt.scatter(matrizEntrada[:, 0], matrizEntrada[:, 1], c=saidas, cmap='coolwarm')
plt.title('Saídas após o Treinamento')
plt.xlabel('Entrada 1')
plt.ylabel('Entrada 2')

# Adicionar uma linha que separa os pontos
plt.plot([0.5, 0.5], [0, 1], linestyle='--', color='black')  # Linha vertical
plt.plot([0, 1], [0.5, 0.5], linestyle='--', color='black')  # Linha horizontal

plt.colorbar()
plt.grid(True)
plt.show()
'''
