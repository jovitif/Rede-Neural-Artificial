import numpy as np

# Função sigmoide para ativação
def funcaoSigmoide(x):
    return 1 / (1 + np.exp(-x))

# Derivada da função sigmoide
def derivadaSigmoide(x):
    return x * (1 - x)

# TABELA DO XOR
matrizEntrada = np.array([
    [0, 0],
    [1, 1],
    [1, 0],
    [0, 1]
])

# Saídas desejadas correspondentes às entradas
training_outputs = np.array([[0, 0, 1, 1]]).T

# Inicialização dos pesos sinápticos com valores aleatórios entre -1 e 1 para as camadas oculta e de saída
np.random.seed(1)
pesos_camada_oculta = 2 * np.random.random((2, 4)) - 1
pesos_camada_saida = 2 * np.random.random((4, 1)) - 1

# Taxa de aprendizado
learning_rate = 0.1

# Treinamento da rede neural (várias iterações)
for iteration in range(10000):
    # Camada oculta
    input_camada_oculta = np.dot(matrizEntrada, pesos_camada_oculta)
    output_camada_oculta = funcaoSigmoide(input_camada_oculta)

    # Camada de saída
    input_camada_saida = np.dot(output_camada_oculta, pesos_camada_saida)
    output_camada_saida = funcaoSigmoide(input_camada_saida)

    # Cálculo do erro
    erro_camada_saida = training_outputs - output_camada_saida

    # Cálculo dos deltas (gradientes)
    delta_saida = erro_camada_saida * derivadaSigmoide(output_camada_saida)
    delta_oculta = delta_saida.dot(pesos_camada_saida.T) * derivadaSigmoide(output_camada_oculta)

    # Atualização dos pesos com base nos deltas (backpropagation)
    pesos_camada_saida += output_camada_oculta.T.dot(delta_saida) * learning_rate
    pesos_camada_oculta += matrizEntrada.T.dot(delta_oculta) * learning_rate

# Teste com os dados de treinamento
print('Saídas após o treinamento:\n')
for i in range(len(matrizEntrada)):
    input_layer = matrizEntrada[i]
    input_camada_oculta = np.dot(input_layer, pesos_camada_oculta)
    output_camada_oculta = funcaoSigmoide(input_camada_oculta)
    input_camada_saida = np.dot(output_camada_oculta, pesos_camada_saida)
    output_camada_saida = funcaoSigmoide(input_camada_saida)
    print(f"Entrada: {input_layer}, Saída: {output_camada_saida[0]}")
