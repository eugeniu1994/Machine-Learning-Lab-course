import scipy.linalg as la
import matplotlib.pyplot as plt
import sklearn.svm
from cvxopt.solvers import qp
from cvxopt import matrix as cvxmatrix
import numpy as np
import torch

from torch.optim import SGD #just to test my implementation
from torch.nn import Module, Parameter, ParameterList

class neural_network(Module):
    def __init__(self, layers=[2,100,2], scale=.1, p=None, lr=None, lam=None):
        super().__init__()
        self.weights = ParameterList([Parameter(scale*torch.randn(m, n)) for m, n in zip(layers[:-1], layers[1:])])
        self.biases = ParameterList([Parameter(scale*torch.randn(n)) for n in layers[1:]])

        self.p = p
        self.lr = lr if lr is not None else 1e-3
        self.lam = lam if lam is not None else 0
        self.train = False

    def relu(self, X, W, b):
        if self.train:
            #h = np.maximum(0, X.mm(W) + b)
            scalar = torch.FloatTensor([0])
            h = X.mm(W) + b
            h = torch.max(h, scalar.expand_as(h))
            # bernoulli
            delta = torch.distributions.binomial.Binomial(probs=1 - self.p)
            delta = delta.sample(b.size()) * (1.0 / (1 - self.p))
            return delta*h
        else:
            #return np.maximum(0, (1-self.p)*(X.mm(W).detach().numpy()) + b)
            scalar = torch.FloatTensor([0])
            h = (1-self.p)*X.mm(W) + b
            return torch.max(h, scalar.expand_as(h))

    def softmax(self, X, W, b):
        Z = torch.exp(X.mm(W) + b)
        Y_hat = Z/torch.sum(Z, dim=1, keepdim=True)
        return Y_hat

    def forward(self, X):
        X = torch.tensor(X, dtype=torch.float)
        Z = X
        L = len(self.weights)
        for l in range(L-1):
            Z = self.relu(Z,self.weights[l],self.biases[l])
        Y_hat = self.softmax(Z,self.weights[-1], self.biases[-1])
        return Y_hat

    def predict(self, X):
        return self.forward(X).detach().numpy()

    def loss(self, ypred, ytrue):
        cross_entropy = -(ytrue * ypred.log() + (1 - ytrue) * (1 - ypred).log()).mean()
        return cross_entropy

    def fit_(self, X, y, nsteps=1000, bs=100, plot=False):
        X, y = torch.tensor(X), torch.tensor(y)
        optimizer = SGD(self.parameters(), lr=self.lr, weight_decay=self.lam)

        I = torch.randperm(X.shape[0])
        n = int(np.ceil(.1 * X.shape[0]))
        Xtrain, ytrain = X[I[:n]], y[I[:n]]
        Xval, yval = X[I[n:]], y[I[n:]]

        Ltrain, Lval, Aval = [], [], []
        for i in range(nsteps):
            optimizer.zero_grad()
            I = torch.randperm(Xtrain.shape[0])[:bs]
            self.train = True
            output = self.loss(self.forward(Xtrain[I]), ytrain[I])
            self.train = False
            Ltrain += [output.item()]
            output.backward()
            optimizer.step()

            outval = self.forward(Xval)
            Lval += [self.loss(outval, yval).item()]
            Aval += [np.array(outval.argmax(-1) == yval.argmax(-1)).mean()]

        print('Ltrain ',np.shape(Ltrain))
        print('Lval ', np.shape(Lval))
        print('Aval ', np.shape(Aval))
        plot = True
        if plot:
            plt.plot(range(nsteps), Ltrain, label='Training loss')
            plt.plot(range(nsteps), Lval, label='Validation loss')
            plt.plot(range(nsteps), Aval, label='Validation acc')
            plt.legend()
            plt.show()

    def fit_(self, X, y, nsteps=1000, bs=100, plot=False):
        X, y = torch.tensor(X), torch.tensor(y)
        def SGD(params, lr):
            print('params ',params)
            print(params)
            #for p in params:
            #    p[:] = p - lr * p.grad
            #w = w.data - lr * w.grad.data
            #b = b.data - lr * b.grad.data
            #w.grad.data.zero_()
            #b.grad.data.zero_()

        I = torch.randperm(X.shape[0])
        n = int(np.ceil(.1 * X.shape[0]))
        Xtrain, ytrain = X[I[:n]], y[I[:n]]
        Xval, yval = X[I[n:]], y[I[n:]]

        Ltrain, Lval, Aval = [], [], []
        for i in range(nsteps):
            I = torch.randperm(Xtrain.shape[0])[:bs]
            self.train = True
            output = self.loss(self.forward(Xtrain[I]), ytrain[I])
            self.train = False
            Ltrain += [output.item()]

            #output.backward() #compute gradients
            #optimizer.step()
            SGD(self.parameters(), self.lr)

            outval = self.forward(Xval)
            Lval += [self.loss(outval, yval).item()]
            Aval += [np.array(outval.argmax(-1) == yval.argmax(-1)).mean()]

        print('Ltrain ',np.shape(Ltrain))
        print('Lval ', np.shape(Lval))
        print('Aval ', np.shape(Aval))
        plot = True
        if plot:
            plt.plot(range(nsteps), Ltrain, label='Training loss')
            plt.plot(range(nsteps), Lval, label='Validation loss')
            plt.plot(range(nsteps), Aval, label='Validation acc')
            plt.legend()
            plt.show()
