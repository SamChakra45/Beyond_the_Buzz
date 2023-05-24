import numpy as np
import pandas as pd
from model import ANN
from sklearn.metrics import accuracy_score
from tensorflow import keras
from keras.datasets import mnist


(train_X, train_y), (test_X, test_y) = mnist.load_data()

train_X = train_X.flatten().reshape(60000,784)/255.0
train_X = train_X.T  
test_X = test_X.flatten().reshape(10000,784)/255.0
test_X = test_X.T

train_y = pd.get_dummies(train_y).values.T
test_y = pd.get_dummies(test_y).values.T

# Create an instance of the ANN model
input_size = train_X.shape[0]
output_size = train_y.shape[0]
learning_rate = 0.1
num_hidden_layers = 4
num_nodes_hidden = 128

model = ANN(input_size, output_size, learning_rate, num_hidden_layers, num_nodes_hidden)

# Train the model
num_epochs = 100
model.train(train_X, train_y, num_epochs)

# Make predictions on the test set
y_pred = model.predict(test_X)

# Calculate the accuracy of the model
accuracy = accuracy_score(np.argmax(test_y, axis=0), y_pred)
print(f"Accuracy: {accuracy}")
