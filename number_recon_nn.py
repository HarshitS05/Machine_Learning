import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

data=pd.read_csv('train _number_recon_nn.csv')
#print(data.head(10))  # first 10 rows

data=np.array(data)
m,n=data.shape
np.random.shuffle(data)

train_data=data[0:int(0.8*m),:]
test_data=data[int(0.8*m):,:]

x_train=train_data[:,1:]
x_train=x_train/255.0 # Normalize pixel values to [0,1]
y_train=train_data[:,0]

x_test=test_data[:,1:]
x_test=x_test/255.0
y_test=test_data[:,0]

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

def inint_weights():
    w1 = np.random.randn(10, 784) * 0.01  # small random values
    b1 = np.random.randn(10, 1)  * 0.01
    w2 = np.random.randn(10, 10) * 0.01
    b2 = np.random.randn(10, 1)  * 0.01
    return w1, b1, w2, b2

def relu(x):
    return np.maximum(0, x)

def softmax(x):
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=0)


def one_hot(y):
    one_hot_y = np.zeros((10, y.size))
    one_hot_y[y.astype(int), np.arange(y.size)] = 1
    return one_hot_y

def forward_propagation(w1,w2,b1,b2,x):
    z1=np.dot(w1,x.T)+b1
    a1=relu(z1)
    z2=np.dot(w2,a1)+b2
    a2=softmax(z2)
    return z1,z2,a1,a2

def relu_derivative(x):
    return 1.0*(x>0)

def backward_propagation(w1, w2, b1, b2, z1, z2, a1, a2, y, x, m):
    dw2 = 1/m * np.dot((a2 - y), a1.T)
    db2 = 1/m * np.sum(a2 - y, axis=1).reshape(10,1)
    dw1 = 1/m * np.dot(np.dot(w2.T, (a2 - y)) * relu_derivative(z1), x)
    db1 = 1/m * np.sum(np.dot(w2.T, (a2 - y)) * relu_derivative(z1), axis=1).reshape(10,1)
    return dw1, db1, dw2, db2

def update_parameters(w1,w2,b1,b2,L,dW1,dW2,db1,db2):
    w1=w1-L*dW1
    b1=b1-L*db1
    w2=w2-L*dW2    
    b2=b2-L*db2
    return w1,w2,b1,b2

def train(x_train,y_train,epochs,L):
    w1,b1,w2,b2=inint_weights()
    y_encoded = one_hot(y_train)
    for i in range(epochs):
        z1,z2,a1,a2=forward_propagation(w1,w2,b1,b2,x_train)
        dw1,db1,dw2,db2=backward_propagation(w1,w2,b1,b2,z1,z2,a1,a2,y_encoded,x_train,m)
        w1,w2,b1,b2=update_parameters(w1,w2,b1,b2,L,dw1,dw2,db1,db2)
        if(i%20==0):
            print(f"Epoch {i} completed")
    return w1,w2,b1,b2

w1,w2,b1,b2=train(x_train,y_train,epochs=100,L=0.01)

