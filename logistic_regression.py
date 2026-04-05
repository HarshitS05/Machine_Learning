import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Load data
data = pd.read_csv("crime_train.csv")

# Keep only important columns (drop text columns)
drop_cols = ['crime_description', 'Unnamed: 0', 'Num']
for col in drop_cols:
    if col in data.columns:
        data = data.drop(columns=[col])

# Convert target Yes/No → 1/0
data['closed'] = data['closed'].map({"Yes":1, "No":0})

# Drop rows with missing values
data = data.dropna()

# Manually one-hot encode categorical columns
data = pd.get_dummies(data, dummy_na=True)

# Split into features and labels
y = data['closed'].values
X = data.drop(columns=['closed'])

# Normalize features (important for gradient descent)
X = (X - X.mean()) / (X.std() + 1e-9)

# Add bias column
X['bias'] = 1

# Convert to NumPy
X = X.to_numpy()


# =========================================
# SIGMOID FUNCTION
# =========================================
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# =========================================
# LOSS (Binary Cross Entropy)
# =========================================
def loss(weights, X, y):
    predictions = sigmoid(X @ weights)
    return -(y * np.log(predictions + 1e-9) + (1-y) * np.log(1 - predictions + 1e-9)).mean()


# =========================================
# GRADIENT DESCENT (Vectorized)
# =========================================
def gradient_descent(weights, X, y, lr):
    m = len(y)
    predictions = sigmoid(X @ weights)
    error = predictions - y

    gradients = (1/m) * (X.T @ error)
    weights = weights - lr * gradients
    return weights


# =========================================
# TRAINING
# =========================================
epochs = 1000
lr = 0.01
weights = np.zeros(X.shape[1])

print("Training...")

for i in range(epochs):
    weights = gradient_descent(weights, X, y, lr)
    if i % 100 == 0:
        print(f"Epoch {i} | Loss = {loss(weights, X, y):.4f}")

print("\nTraining Complete ✅")
print("Final Weights:")
print(weights)


# =========================================
# PLOT: ACTUAL VS PREDICTED PROBABILITY
# =========================================
predicted_prob = sigmoid(X @ weights)

plt.figure(figsize=(10, 6))
plt.scatter(y, predicted_prob, alpha=0.3, color="blue")

# Perfect line reference
plt.plot([0,1],[0,1], 'r--')

plt.xlabel("Actual (0=No, 1=Yes)")
plt.ylabel("Predicted Probability")
plt.title("Model Prediction Accuracy (From Scratch)")
plt.show()
