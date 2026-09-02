import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


data = pd.read_csv("crime_train.csv")


drop_cols = ['crime_description', 'Unnamed: 0', 'Num']
for col in drop_cols:
    if col in data.columns:
        data = data.drop(columns=[col])


data['closed'] = data['closed'].map({"Yes":1, "No":0})


data = data.dropna()


data = pd.get_dummies(data, dummy_na=True)


y = data['closed'].values
X = data.drop(columns=['closed'])


X = (X - X.mean()) / (X.std() + 1e-9)


X['bias'] = 1


X = X.to_numpy()


# =========================================
# SIGMOID FUNCTION
# =========================================
def sigmoid(z):
    return 1 / (1 + np.exp(-z))



def loss(weights, X, y):
    predictions = sigmoid(X @ weights)
    return -(y * np.log(predictions + 1e-9) + (1-y) * np.log(1 - predictions + 1e-9)).mean()



def gradient_descent(weights, X, y, lr):
    m = len(y)
    predictions = sigmoid(X @ weights)
    error = predictions - y

    gradients = (1/m) * (X.T @ error)
    weights = weights - lr * gradients
    return weights



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



predicted_prob = sigmoid(X @ weights)

plt.figure(figsize=(10, 6))
plt.scatter(y, predicted_prob, alpha=0.3, color="blue")


plt.plot([0,1],[0,1], 'r--')

plt.xlabel("Actual (0=No, 1=Yes)")
plt.ylabel("Predicted Probability")
plt.title("Model Prediction Accuracy (From Scratch)")
plt.show()
