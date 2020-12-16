import numpy as np
import matplotlib.pyplot as plt 

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def transfer_derivative(output):
    return output * (1.0 - output)

np.random.seed(12345)

m0=100
m=2*m0+1

X=2*np.arange(m+1)/m-1 ## input data is in the range [-1,+1], with steps 0.01
Y=X**2 ## output values: a parabola
X = X[:, np.newaxis]
Y = Y[:, np.newaxis]

niteration=100 ## number of iterations

H = 5
learning_rate = 0.3
W1 = np.random.randn(H, X.shape[0]) * 0.01
W2 = np.random.randn(Y.shape[0], H) * 0.01

a=2*((np.random.rand(H)-0.5))/100 ## random uniform in [-0.01,0.01]
b=2*((np.random.rand(H)-0.5))/100 ## random uniform in [-0.01,0.01]

W1 = np.ones_like(W1)
#W1 = W1[:, np.newaxis]
#W2 = W2[:, np.newaxis].T

print('W1:{}, W2:{}'.format(np.shape(W1), np.shape(W2)))
print('X:{},  Y:{}'.format(np.shape(X), np.shape(Y)))

def forward_propagation(X, W1, W2):
    Z1 = np.dot(W1, X)
    A1 = sigmoid(Z1)

    Z2 = np.dot(W2, A1)
    A2 = Z2
    
    cache = {"Z1": Z1,
            "A1": A1, 
            "Z2": Z2, 
            "A2": A2} 
    
    return A2, cache 

def backward_propagation(W1, W2, cache):
    A1 = cache['A1']
    A2 = cache['A2'] 

    # Backward propagation:
    dZ2 = A2 - Y 
    dW2 = np.dot(dZ2, A1.T)

    dZ1 = np.multiply(np.dot(W2.T, dZ2), transfer_derivative(A1))
    dW1 = np.dot(dZ1, X.T)

    W1 = W1 - learning_rate * dW1
    W2 = W2 - learning_rate * dW2

    return W1, W2

for i in range(niteration):
    # Forward propagation
    A2, cache = forward_propagation(X, W1, W2)

    cost = Y-A2

    # Backpropagation.
    W1, W2 = backward_propagation(W1, W2, cache)

    print ("Cost at iteration % i: % f" % (i, cost.sum()))

A2, cache = forward_propagation(X, W1, W2)

err = Y[m0+1] - A2[m0+1]
print('rr[m0+1]:{},  yy[m0+1]:{},  err:{}'.format(Y[m0+1],A2[m0+1],err))

plt.plot(X,Y, label='True data')
plt.plot(X,A2,label='Predicted')
plt.legend()
plt.show()

