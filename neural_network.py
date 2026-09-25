"""
NN - Neural Network from Scratch
Pure Python implementation of a feedforward neural network with backpropagation.
Works with or without NumPy installed.
"""

import math
import random

# ==========================================
# Funções de Ativação e Derivadas
# ==========================================

def sigmoid(x):
    # Clipa x para evitar overflow numérico
    x = max(min(x, 50.0), -50.0)
    return 1.0 / (1.0 + math.exp(-x))

def sigmoid_derivative(output):
    return output * (1.0 - output)

def tanh(x):
    return math.tanh(x)

def tanh_derivative(output):
    return 1.0 - output ** 2

def relu(x):
    return max(0.0, x)

def relu_derivative(output):
    return 1.0 if output > 0 else 0.0

ACTIVATIONS = {
    'sigmoid': (sigmoid, sigmoid_derivative),
    'tanh': (tanh, tanh_derivative),
    'relu': (relu, relu_derivative)
}

# ==========================================
# Classe Neurônio
# ==========================================

class Neuron:
    def __init__(self, num_inputs, activation='tanh'):
        # Inicialização Xavier / Glorot
        limit = math.sqrt(2.0 / num_inputs) if num_inputs > 0 else 1.0
        self.weights = [random.uniform(-limit, limit) for _ in range(num_inputs)]
        self.bias = random.uniform(-0.1, 0.1)
        self.output = 0.0
        self.delta = 0.0
        self.activation_name = activation
        self.act_fn, self.act_derivative = ACTIVATIONS[activation]

    def forward(self, inputs):
        total = sum(w * i for w, i in zip(self.weights, inputs)) + self.bias
        self.output = self.act_fn(total)
        return self.output

# ==========================================
# Classe Camada (Layer)
# ==========================================

class Layer:
    def __init__(self, num_neurons, num_inputs_per_neuron, activation='tanh'):
        self.neurons = [Neuron(num_inputs_per_neuron, activation) for _ in range(num_neurons)]

    def forward(self, inputs):
        return [neuron.forward(inputs) for neuron in self.neurons]

# ==========================================
# Classe Rede Neural (NeuralNetwork)
# ==========================================

class NeuralNetwork:
    """
    Rede Neural Multicamadas (Multi-Layer Perceptron - MLP)
    Exemplo de arquitetura: [2, 4, 4, 1] -> 2 entradas, duas camadas ocultas de 4 neurônios, 1 saída.
    """
    def __init__(self, topology, hidden_activation='tanh', output_activation='sigmoid'):
        self.topology = topology
        self.layers = []
        
        # Cria as camadas ocultas
        for i in range(1, len(topology) - 1):
            self.layers.append(Layer(topology[i], topology[i - 1], activation=hidden_activation))
            
        # Cria a camada de saída
        self.layers.append(Layer(topology[-1], topology[-2], activation=output_activation))

    def forward(self, inputs):
        current_input = inputs
        for layer in self.layers:
            current_input = layer.forward(current_input)
        return current_input

    def backward(self, expected_outputs):
        # 1. Gradiente para a camada de saída
        output_layer = self.layers[-1]
        for i, neuron in enumerate(output_layer.neurons):
            error = expected_outputs[i] - neuron.output
            neuron.delta = error * neuron.act_derivative(neuron.output)

        # 2. Propaga o erro de trás para frente (Backpropagation)
        for layer_idx in range(len(self.layers) - 2, -1, -1):
            current_layer = self.layers[layer_idx]
            next_layer = self.layers[layer_idx + 1]

            for i, neuron in enumerate(current_layer.neurons):
                error = sum(next_neuron.weights[i] * next_neuron.delta for next_neuron in next_layer.neurons)
                neuron.delta = error * neuron.act_derivative(neuron.output)

    def update_weights(self, inputs, learning_rate=0.1):
        for layer_idx, layer in enumerate(self.layers):
            # As entradas da primeira camada são as entradas da rede, para as outras são as saídas da camada anterior
            prev_outputs = inputs if layer_idx == 0 else [n.output for n in self.layers[layer_idx - 1].neurons]
            
            for neuron in layer.neurons:
                for i in range(len(neuron.weights)):
                    neuron.weights[i] += learning_rate * neuron.delta * prev_outputs[i]
                neuron.bias += learning_rate * neuron.delta

    def train_step(self, inputs, targets, learning_rate=0.1):
        # Passo para a frente
        outputs = self.forward(inputs)
        # Passo para trás
        self.backward(targets)
        # Atualização dos pesos
        self.update_weights(inputs, learning_rate)
        # Cálculo de erro quadrático (Loss)
        loss = sum((t - o) ** 2 for t, o in zip(targets, outputs)) / len(targets)
        return outputs, loss

    def fit(self, dataset, epochs=5000, learning_rate=0.1, verbose_every=1000):
        print(f"Iniciando treinamento ({epochs} épocas, lr={learning_rate})...")
        for epoch in range(1, epochs + 1):
            total_loss = 0.0
            # Embaralha os dados em cada época
            random.shuffle(dataset)
            for inputs, targets in dataset:
                _, loss = self.train_step(inputs, targets, learning_rate)
                total_loss += loss
            
            avg_loss = total_loss / len(dataset)
            if epoch % verbose_every == 0 or epoch == 1 or epoch == epochs:
                print(f"Época {epoch:5d}/{epochs} - Loss Médio: {avg_loss:.6f}")

    def predict(self, inputs):
        return self.forward(inputs)

# ==========================================
# Demonstração Prática (Problema XOR)
# ==========================================

if __name__ == '__main__':
    print("=" * 60)
    print("🧠 NN: Treinando Rede Neural no Problema Clássico XOR")
    print("=" * 60)

    # Dataset XOR: 2 entradas -> 1 saída
    xor_data = [
        ([0.0, 0.0], [0.0]),
        ([0.0, 1.0], [1.0]),
        ([1.0, 0.0], [1.0]),
        ([1.0, 1.0], [0.0]),
    ]

    # Arquitetura: 2 entradas -> Camada Oculta com 4 neurônios -> 1 neurônio de saída
    nn = NeuralNetwork(topology=[2, 4, 1], hidden_activation='tanh', output_activation='sigmoid')

    # Treinamento
    nn.fit(xor_data, epochs=4000, learning_rate=0.2, verbose_every=1000)

    print("\n📊 Resultados Finais das Predições:")
    print("-" * 40)
    for inputs, targets in [([0, 0], [0]), ([0, 1], [1]), ([1, 0], [1]), ([1, 1], [0])]:
        pred = nn.predict(inputs)[0]
        status = "✅ Correto" if round(pred) == targets[0] else "❌ Incorreto"
        print(f"Entrada: {inputs} | Esperado: {targets[0]} | Predição: {pred:.4f} (Round: {round(pred)}) | {status}")
    print("=" * 60)
