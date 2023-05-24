import numpy as np
import pandas as pd

class ANN:
    def __init__(self, input_size, output_size, learning_rate, num_hidden_layers, num_nodes_hidden):
        self.input_size = input_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.num_hidden_layers = num_hidden_layers
        self.num_nodes_hidden = num_nodes_hidden
        
        self.weights = []
        self.biases = []
        
        self.initialize_parameters()

    def initialize_parameters(self):
        # Randomly initialize weights and biases for each layer
        layer_sizes = [self.input_size] + [self.num_nodes_hidden] * self.num_hidden_layers + [self.output_size]
        num_layers = len(layer_sizes)

        for i in range(1, num_layers):
            weight_matrix = np.random.randn(layer_sizes[i], layer_sizes[i-1]) / np.sqrt(layer_sizes[i-1])
            bias_matrix = np.zeros((layer_sizes[i], 1))
            self.weights.append(weight_matrix)
            self.biases.append(bias_matrix)

    def forward_propagation(self, X):
        activations = [X]
        layer_inputs = []

        for i in range(self.num_hidden_layers + 1):
            layer_input = np.dot(self.weights[i], activations[i]) + self.biases[i]
            layer_output = self.sigmoid(layer_input)
            layer_inputs.append(layer_input)
            activations.append(layer_output)

        return activations, layer_inputs

    def backward_propagation(self, X, y, activations, layer_inputs):
        num_examples = X.shape[1]
        delta_weights = [np.zeros_like(w) for w in self.weights]
        delta_biases = [np.zeros_like(b) for b in self.biases]

        # Calculate gradients for the output layer
        error = activations[-1] - y
        delta = error * self.sigmoid_derivative(layer_inputs[-1])
        delta_weights[-1] = np.dot(delta, activations[-2].T) / num_examples
        delta_biases[-1] = np.sum(delta, axis=1, keepdims=True) / num_examples

        # Backpropagate the error through hidden layers
        for i in range(self.num_hidden_layers-1, -1, -1):
            delta = np.dot(self.weights[i+1].T, delta) * self.sigmoid_derivative(layer_inputs[i])
            delta_weights[i] = np.dot(delta, activations[i].T) / num_examples
            delta_biases[i] = np.sum(delta, axis=1, keepdims=True) / num_examples

        return delta_weights, delta_biases

    def train(self, X, y, num_epochs):
        for epoch in range(num_epochs):
            activations, layer_inputs = self.forward_propagation(X)
            delta_weights, delta_biases = self.backward_propagation(X, y, activations, layer_inputs)

            # Update weights and biases using gradient descent
            for i in range(self.num_hidden_layers + 1):
                self.weights[i] -= self.learning_rate * delta_weights[i]
                self.biases[i] -= self.learning_rate * delta_biases[i]

            # Compute and print the training loss for every epoch
            loss = self.calculate_loss(activations[-1], y)
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss}")

    def predict(self, X):
        activations, _ = self.forward_propagation(X)
        predictions = np.argmax(activations[-1], axis=0)
        return predictions

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def calculate_loss(self, y_pred, y_true):
        num_examples = y_true.shape[1]
        loss = -np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)) / num_examples
        return loss
