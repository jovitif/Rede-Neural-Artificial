import numpy as np

def signoid(x): # retorna qualuqer valor entre 0 e 1
    return 1 / (1+ np.exp(-x))

#Exemplos de imputs para treinamento
training_inputs = np.array([
    [0,0,1],
    [1,1,1],
    [1,0,1],
    [0,1,1]]
)

#Saida
training_outputs = np.array([[0,1,1,0]]).T

np.random.seed(1)

synaptic_weights = 2 * np.random.random((3,1))-1

print('Random starting synaptic weights:')
print(synaptic_weights)

for iteration in range(1):
    input_layer = training_inputs
    outputs = signoid(np.dot(input_layer, synaptic_weights))

print('Outputs after training:')
print(outputs)

