import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

df = pd.read_csv("spamhamdata.csv", sep='\t', header=None, 
                 names=['label', 'message'], encoding='latin-1', 
                 lineterminator='\n')
df['label'] = df['label'].str.strip()
df['message'] = df['message'].str.strip()
df = df.dropna().reset_index(drop=True)
print(df.shape)
print(df.head())

X = df['message'].values
y = df['label'].map({'ham': 0, 'spam': 1}).values

vectorizer = TfidfVectorizer(max_features=5000)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = vectorizer.fit_transform(X_train).toarray()
X_test = vectorizer.transform(X_test).toarray()

print(X_train.shape)  # should be (n, 5000)
print(y_train.shape)  # should be (n,)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def grad(theta, X,y):
    m=y.size
    error = sigmoid(X @ theta) - y
    gradient = 1/m * (X.T @ error)
    return gradient

def grad_descent(X, y, theta, L=0.001, epoch=100):
    for i in range(epoch):
        gradient = grad(theta, X, y)
        theta = theta - L * gradient
        if i % 10 == 0:
            loss = -np.mean(y * np.log(sigmoid(X @ theta) + 1e-8) + (1-y) * np.log(1 - sigmoid(X @ theta) + 1e-8))
            print(f"Epoch {i} loss: {loss:.4f}")
    return theta

theta  = np.zeros(X_train.shape[1])
theta = grad_descent(X_train, y_train, theta)

predictions = sigmoid(X_test @ theta)

print(df.shape)        # should be (5574, 2)
print(df.head())       # check it looks right
print(df.isnull().sum()) # check no nulls