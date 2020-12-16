import numpy as np
import matplotlib.pyplot as plt
from math import factorial as fact

# EX1
'''L = 15
k_ = 8#
#k_ = int(L/2)
eps = .3
R = 0.0
for k in range(k_,L+1):
    print('k ',k)
    R += (fact(L)/(fact(k)*fact(L-k)))*np.power(eps,k)*np.power(1-eps,L-k)
print('R ',R)'''

T = 5
x1 = [.1, .2, .4, .8, .8, .05, .08, .12, .33, .55, .66, .77, .22, .2, .3, .6, .5, .6, .25, .3, .5, .7, .6]
x2 = [.2, .65, .7, .6, .3, .1, .4, .66, .22, .65, .68, .55, .44, .1, .3, .4, .3, .15, .15, .5, .55, .2, .4]
bias = np.ones_like(x1)
labels = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]

#data = np.vstack((x1, x2)).T
data = np.vstack((x1, x2, bias)).T

print('data ', np.shape(data))

def perceptron(X, labels, sample_weight):
    # Inputs:
    # X: a 2d array, each row represents an example of the training set
    # labels: vector of the examples labels
    # sample_weight:vector of examples weights given by Adaboost
    # Output:
    # pred_labels: the label predicted for each example
    d = np.shape(X)[1]
    w = np.zeros(d)
    #print('features ',d)
    i = 1
    while (any([element <= 0 for element in [labels[ind] * np.dot(w, x) for ind, x in enumerate(X)]])):
        # misclassified examples
        mistakes = np.where([element <= 0 for element in [labels[ind] * np.dot(w, x) for ind, x in
                                                          enumerate(X)]])[0]

        pairs = zip(mistakes, sample_weight[mistakes])
        sorted_pairs = sorted(pairs, key=lambda t: t[1], reverse=True)
        # use the misclassified example with maximum weight given by Adaboost
        misclass = sorted_pairs[0][0]
        # weight update
        w = w + labels[misclass] * X[misclass]
        # labels prediction
        pred_labels = [1 if x > 0 else -1 for x in [np.dot(w, x) for x in X]]
        i += 1
        if (i > 201):
            break

    return pred_labels

from sklearn import metrics

m = len(labels)
print('m ',m)
D = np.full((np.shape(labels)), 1/m, dtype=float)
#print(D)
alfa = np.zeros((T), dtype=float)
for t in range(T):
    h_t = perceptron(X=data, labels=labels, sample_weight=D)
    h_t,labels = np.array(h_t), np.array(labels)
    #print('h_t ',h_t)
    #eps = (h_t != labels).mean()
    eps = ((h_t != labels) * D).mean()
    print(' eps ', eps)
    print()

    alfa[t] = .5*np.log((1-eps)/eps)
    #print(alfa)
    Z = 2*np.sqrt(eps*(1-eps))

    for i in range(m):
        D[i] *= (np.exp(-alfa[t]*labels[i]*h_t[i]))/Z

print('alfa ',alfa)
f = np.array([alfa[t]*np.array(perceptron(X=data, labels=labels, sample_weight=D)) for t in range(T)]).sum(axis=0)
#print('f ',np.shape(f))
#print(f)
f = np.array([ np.sign(i) for i in f])
#print(f)
#eps = np.mean((f != labels))
#eps = ((f != labels) * D).mean()
#eps = sum((f != labels) * D) / sum(D)
#print('train_err  ',eps)
#print('Accuracy: ', 1-eps)
print("Accuracy:", metrics.accuracy_score(labels, f))
