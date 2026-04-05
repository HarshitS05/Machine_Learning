import numpy as np
import pandas as pd
from PIL import Image
from matplotlib import pyplot as plt
import os

classes = {'Jade':0, 'James':1, 'Jane':2, 'Joel':3, 'Jovi':4}

x = []
y = []

for class_name, label in classes.items():
    folder = f'C:/Users/harsh/Downloads/Train/{class_name}'
    for file in os.listdir(folder):
        img = Image.open(os.path.join(folder, file))
        img_flat = np.array(img).flatten()
        x.append(img_flat)
        y.append(label)

x = np.array(x) / 255.0
y = np.array(y)

def init_weights():
    w1 = np.random.randn(10, 784) * 0.01
    b1 = np.random.randn(10, 1) * 0.01
    w2 = np.random.randn(5, 10) * 0.01
    b2 = np.random.randn(5, 1) * 0.01
    return w1, b1, w2, b2

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return 1.0 * (x > 0)

def one_hot(y):
    one_hot_y = np.zeros((5, y.size))
    one_hot_y[y.astype(int), np.arange(y.size)] = 1
    return one_hot_y

def forward_propagation(w1, w2, b1, b2, x):
    z1 = np.dot(w1, x.T) + b1
    a1 = relu(z1)
    z2 = np.dot(w2, a1) + b2
    a2 = np.exp(z2) / np.sum(np.exp(z2), axis=0)
    return z1, z2, a1, a2

def backward_propagation(w1, w2, b1, b2, z1, z2, a1, a2, y, x, m):
    dw2 = 1/m * np.dot((a2 - y), a1.T)
    db2 = 1/m * np.sum(a2 - y, axis=1).reshape(5, 1)
    dw1 = 1/m * np.dot(np.dot(w2.T, (a2 - y)) * relu_derivative(z1), x)
    db1 = 1/m * np.sum(np.dot(w2.T, (a2 - y)) * relu_derivative(z1), axis=1).reshape(10, 1)
    return dw1, db1, dw2, db2

def update_weights(w1, b1, w2, b2, L, dw1, db1, dw2, db2):
    w1 = w1 - L * dw1
    b1 = b1 - L * db1
    w2 = w2 - L * dw2
    b2 = b2 - L * db2
    return w1, w2, b1, b2

def get_accuracy(a2, y):
    predictions = np.argmax(a2, axis=0)
    return np.sum(predictions == y) / y.size

def train(x, y, epochs, L):
    w1, b1, w2, b2 = init_weights()
    y_encoded = one_hot(y)
    for i in range(epochs):
        z1, z2, a1, a2 = forward_propagation(w1, w2, b1, b2, x)
        dw1, db1, dw2, db2 = backward_propagation(w1, w2, b1, b2, z1, z2, a1, a2, y_encoded, x, x.shape[0])
        w1, w2, b1, b2 = update_weights(w1, b1, w2, b2, L, dw1, db1, dw2, db2)
        if i % 20 == 0:
            print(f"Epoch {i} - Accuracy: {get_accuracy(a2, y) * 100:.2f}%")
    return w1, w2, b1, b2

w1, w2, b1, b2 = train(x, y, epochs=1000, L=0.01)

