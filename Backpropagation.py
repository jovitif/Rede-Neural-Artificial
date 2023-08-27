import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Definir os dados de treinamento
training_inputs = np.array([
    [0, 0, 1],
    [1, 1, 1],
    [1, 0, 1],
    [0, 1, 1]]
)

training_outputs = np.array([[0, 1, 1, 0]]).T

np.random.seed(1)

# Inicializar os pesos sinápticos das camadas intermediária e de saída
synaptic_weights_layer1 = 2 * np.random.random((3, 4)) - 1
synaptic_weights_layer2 = 2 * np.random.random((4, 1)) - 1

print('Random starting synaptic weights layer 1:')
print(synaptic_weights_layer1)

print('Random starting synaptic weights layer 2:')
print(synaptic_weights_layer2)

# Treinamento usando backpropagation
for iteration in range(60000):
    # Camada de entrada
    input_layer = training_inputs

    # Propagação para a camada intermediária
    outputs_layer1 = sigmoid(np.dot(input_layer, synaptic_weights_layer1))

    # Propagação para a camada de saída
    outputs_layer2 = sigmoid(np.dot(outputs_layer1, synaptic_weights_layer2))

    # Cálculo do erro nas saídas da camada de saída
    error_layer2 = training_outputs - outputs_layer2

    # Cálculo dos ajustes na camada de saída
    adjustments_layer2 = error_layer2 * sigmoid_derivative(outputs_layer2)

    # Retropropagação do erro para a camada intermediária
    error_layer1 = adjustments_layer2.dot(synaptic_weights_layer2.T)

    # Cálculo dos ajustes na camada intermediária
    adjustments_layer1 = error_layer1 * sigmoid_derivative(outputs_layer1)

    # Atualização dos pesos sinápticos
    synaptic_weights_layer2 += outputs_layer1.T.dot(adjustments_layer2)
    synaptic_weights_layer1 += input_layer.T.dot(adjustments_layer1)

print('Synaptic weights layer 1 after training:')
print(synaptic_weights_layer1)

print('Synaptic weights layer 2 after training:')
print(synaptic_weights_layer2)

print('Outputs after training:')
print(outputs_layer2)
