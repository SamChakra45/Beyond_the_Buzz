import numpy as np
import pandas as pd

class ANN:
    def __init__(self, input_size, output_size, learning_rate, num_hidden_layers, num_nodes):
        self.input_size = input_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.num_hidden_layers = num_hidden_layers
        self.num_nodes = num_nodes
        
        self.parameters = {}
        self.cache = {}
        self.grads = {}
        
    def initialize_parameters(self):
        np.random.seed(1)
        
        # Initialize parameters for the hidden layers
        self.parameters['W1'] = np.random.randn(self.num_nodes, self.input_size) * 0.01
        self.parameters['b1'] = np.zeros((self.num_nodes, 1))
        
        for i in range(2, self.num_hidden_layers + 1):
            self.parameters[f'W{i}'] = np.random.randn(self.num_nodes, self.num_nodes) * 0.01
            self.parameters[f'b{i}'] = np.zeros((self.num_nodes, 1))
        
        # Initialize parameters for the output layer
        self.parameters[f'W{self.num_hidden_layers + 1}'] = np.random.randn(self.output_size, self.num_nodes) * 0.01
        self.parameters[f'b{self.num_hidden_layers + 1}'] = np.zeros((self.output_size, 1))
    
    def relu(self, Z):
        return np.maximum(0, Z)
    
    def softmax(self, Z):
        exp_Z = np.exp(Z)
        return exp_Z / np.sum(exp_Z, axis=0)
    
    def forward_propagation(self, X):
        A = X
        self.cache[f'A{0}'] = A
        for i in range(1, self.num_hidden_layers + 2):
            Z = np.dot(self.parameters[f'W{i}'], A) + self.parameters[f'b{i}']
            
            if i == self.num_hidden_layers + 1:
                A = self.softmax(Z)
            else:
                A = self.relu(Z)
            
            self.cache[f'A{i}'] = A
            self.cache[f'Z{i}'] = Z
        
        return A
    
    def compute_loss(self, Y, Y_hat):
        m = Y.shape[1]
        loss = -np.sum(np.multiply(Y, np.log(Y_hat))) / m
        return loss
    
    def backward_propagation(self, X, Y, Y_hat):
        m = X.shape[1]
        
        self.grads[f'dZ{self.num_hidden_layers + 1}'] = Y_hat - Y
        
        for i in range(self.num_hidden_layers + 1, 0, -1):
            self.grads[f'dW{i}'] = (1 / m) * np.dot(self.grads[f'dZ{i}'], self.cache[f'A{i-1}'].T)
            self.grads[f'db{i}'] = (1 / m) * np.sum(self.grads[f'dZ{i}'], axis=1, keepdims=True)
            
            if i != 1:
                self.grads[f'dZ{i-1}'] = np.dot(self.parameters[f'W{i}'].T, self.grads[f'dZ{i}']) * (self.cache[f'Z{i-1}'] > 0)
    
    def update_parameters(self):
        for i in range(1, self.num_hidden_layers + 2):
            self.parameters[f'W{i}'] -= self.learning_rate * self.grads[f'dW{i}']
            self.parameters[f'b{i}'] -= self.learning_rate * self.grads[f'db{i}']
    
    def train(self, X, Y, epochs):
        self.initialize_parameters()
        
        for epoch in range(epochs):
            Y_hat = self.forward_propagation(X)
            loss = self.compute_loss(Y, Y_hat)
            self.backward_propagation(X, Y, Y_hat)
            self.update_parameters()
            
            print(f"Epoch {epoch}: Loss = {loss}")
    
    def predict(self, X):
        Y_hat = self.forward_propagation(X)
        predictions = np.argmax(Y_hat, axis=0)
        return predictions


