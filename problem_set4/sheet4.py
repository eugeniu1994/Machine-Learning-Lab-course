import scipy.linalg as la
import matplotlib.pyplot as plt
import sklearn.svm
from cvxopt.solvers import qp
from cvxopt import matrix as cvxmatrix
import numpy as np
import torch

from torch.optim import SGD #just to compare with my implementation
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
        self.L = len(self.weights)

        self.store = {}

    def relu(self, X, W, b, layer = None):
        if self.train:
            #h = np.maximum(0, X.mm(W) + b)
            scalar = torch.FloatTensor([0])
            Z = X.mm(W) + b
            A = torch.max(Z, scalar.expand_as(Z))
            # bernoulli
            delta = torch.distributions.binomial.Binomial(probs=1 - self.p)
            delta = delta.sample(b.size()) * (1.0 / (1 - self.p))
            A = delta*A
            if layer is not None:
                self.store['W_' + layer] = W
                self.store['b_' + layer] = b
                self.store['f_' + layer] = A
                self.store['h_' + layer] = Z
            return A
        else:
            #return np.maximum(0, (1-self.p)*(X.mm(W).detach().numpy()) + b)
            scalar = torch.FloatTensor([0])
            Z = (1-self.p)*X.mm(W) + b
            A = torch.max(Z, scalar.expand_as(Z))
            if layer is not None:
                self.store['W_' + layer] = W
                self.store['b_' + layer] = b
                self.store['f_' + layer] = A
                self.store['h_' + layer] = Z
            return A

    def softmax(self, X, W, b, layer = None):
        Z = X.mm(W) + b
        exp = torch.exp(Z)
        Y_hat = exp/torch.sum(exp, dim=1, keepdim=True)
        if layer is not None:
            self.store['W_' + layer] = W
            self.store['b_' + layer] = b
            self.store['f_' + layer] = Y_hat
            self.store['h_' + layer] = Z
        return Y_hat

    def forward(self, X):
        self.store = {}
        X = torch.tensor(X, dtype=torch.float)
        Z = X
        self.store['f_0'] = Z
        for l in range(self.L-1):
            Z = self.relu(X=Z,W=self.weights[l],b=self.biases[l], layer=str(l+1))
        Y_hat = self.softmax(X=Z,W=self.weights[-1], b=self.biases[-1], layer=str(self.L))
        return Y_hat

    def predict(self, X):
        return self.forward(X).detach().numpy()

    def loss(self, ypred, ytrue):
        cross_entropy = -(ytrue * ypred.log() + (1 - ytrue) * (1 - ypred).log()).mean()
        return cross_entropy

    #this is using implemented SGD
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

        if plot:
            plt.plot(range(nsteps), Ltrain, label='Training loss')
            plt.plot(range(nsteps), Lval, label='Validation loss')
            plt.plot(range(nsteps), Aval, label='Validation acc')
            plt.legend()
            plt.show()

    #this our own implementation
    def fit(self, X, y, nsteps=1000, bs=100, plot=False):
        def relu_derivative(x):
            y = x.detach().clone()
            y[y >= 0] = 1
            y[y < 0] = 0
            return y

        X, y = torch.tensor(X), torch.tensor(y)
        I = torch.randperm(X.shape[0])
        n = int(np.ceil(.1 * X.shape[0]))
        Xtrain, ytrain = X[I[:n]], y[I[:n]]
        Xval, yval = X[I[n:]], y[I[n:]]
        Ltrain, Lval, Aval = [], [], []

        old_outval = 0
        early_stopping = 5
        for i in range(nsteps):
            I = torch.randperm(Xtrain.shape[0])[:bs]
            self.train = True
            output = self.loss(self.forward(Xtrain[I]), ytrain[I])
            self.train = False
            Ltrain += [output.item()]
            #print('Cost ',output.item())

            #print('store ',self.store.keys())
            # backpropagation - compute gradients-------------------------------------------------------
            y_hat = self.store['f_' + str(self.L)].detach().numpy()
            ytrue = ytrain[I].detach().numpy()
            deltaL = y_hat - ytrue

            _w = self.store['f_' + str(self.L - 1)].detach().numpy().T.dot(deltaL)
            _b = np.sum(deltaL, axis=0, keepdims=True)

            # SGD -  Update parameters
            u = torch.from_numpy(
                (self.weights[-1].detach().numpy() - (self.lr * _w)).astype(np.float32)).requires_grad_()
            updated_W = Parameter(u)
            self.weights[-1] = updated_W

            u = torch.from_numpy(
                (self.biases[-1].detach().numpy() - (self.lr * _b)).astype(np.float32)).requires_grad_()
            updated_b = Parameter(u)
            self.biases[-1] = updated_b

            for l in range(self.L-1, 0, -1):
                #print(l)
                h1 = self.store['h_'+str(l)]
                delta = deltaL.dot(self.store['W_'+str(l+1)].detach().numpy().T) * relu_derivative(h1).detach().numpy()
                _w = self.store['f_'+str(l-1)].detach().numpy().T.dot(delta)
                _b = np.sum(delta, axis=0, keepdims=True)
                deltaL = delta

                # SGD -  Update parameters
                u = torch.from_numpy(
                    (self.weights[l-1].detach().numpy() - (self.lr * _w)).astype(np.float32)).requires_grad_()
                updated_W = Parameter(u)
                self.weights[l-1] = updated_W

                u = torch.from_numpy(
                    (self.biases[l-1].detach().numpy() - (self.lr * _b)).astype(np.float32)).requires_grad_()
                updated_b = Parameter(u)
                self.biases[l - 1] = updated_b

            outval = self.forward(Xval)

            validation_loss = self.loss(outval, yval).item()
            if np.abs(old_outval-validation_loss)<1e-3:
                early_stopping -= 1
            old_outval = validation_loss

            Lval += [validation_loss]
            Aval += [np.array(outval.argmax(-1) == yval.argmax(-1)).mean()]
            if early_stopping==0:
                break

        if plot:
            plt.plot( Ltrain, label='Training loss')
            plt.plot( Lval, label='Validation loss')
            plt.plot( Aval, label='Validation acc')
            plt.legend()
            plt.show()

    def fit_Old_just_test(self, X, y, nsteps=1000, bs=100, plot=False):
        X, y = torch.tensor(X), torch.tensor(y)
        I = torch.randperm(X.shape[0])
        n = int(np.ceil(.1 * X.shape[0]))
        Xtrain, ytrain = X[I[:n]], y[I[:n]]
        Xval, yval = X[I[n:]], y[I[n:]]
        Ltrain, Lval, Aval = [], [], []

        def softmax_derivative(s):
            s = s[0]
            jacobian_m = torch.diag(s)
            for i in range(len(jacobian_m)):
                for j in range(len(jacobian_m)):
                    if i == j:
                        jacobian_m[i][j] = s[i] * (1 - s[i])
                    else:
                        jacobian_m[i][j] = -s[i] * s[j]
            return jacobian_m

        def relu_derivative(x):
            y = x.detach().clone()
            y[y >= 0] = 1
            y[y < 0] = 0
            return y

        for i in range(nsteps):
            I = torch.randperm(Xtrain.shape[0])[:bs]
            self.train = True

            X_,Y_ = Xtrain[I], ytrain[I]
            X_ = torch.tensor(X_, dtype=torch.float)

            f0 = X_
            W0 = self.weights[0]
            b0 = self.biases[0]

            h1 = f0.mm(W0) + b0
            f1 = torch.max(h1, torch.FloatTensor([0]).expand_as(h1))
            # bernoulli
            delta = torch.distributions.binomial.Binomial(probs=1 - self.p)
            delta = delta.sample(b0.size()) * (1.0 / (1 - self.p))
            f1 = delta * f1

            W1 = self.weights[1]
            b1 = self.biases[1]
            Z = f1.mm(W1) + b1
            exp = torch.exp(Z)
            y_hat = exp / torch.sum(exp, dim=1, keepdim=True)

            cost = self.loss(y_hat, ytrain[I])
            print('Cost ', cost)

            y_hat = y_hat.detach().numpy()
            ytrue = Y_.detach().numpy()

            Ltrain += [cost]
            self.train = False

            # backpropagation-----------------------------------
            delta2 = y_hat - ytrue
            _w1 = f1.detach().numpy().T.dot(delta2)
            _b1 = np.sum(delta2, axis=0, keepdims=True)

            delta1 = delta2.dot(W1.detach().numpy().T) * relu_derivative(h1).detach().numpy()
            _w0 = f0.detach().numpy().T.dot(delta1)
            _b0 = np.sum(delta1, axis=0, keepdims=True)
            #---------------------------------------------------------------------

            u = torch.from_numpy((self.weights[1].detach().numpy() - (self.lr * _w1)).astype(np.float32)).requires_grad_()
            updated_W = Parameter(u)
            self.weights[1] = updated_W

            u = torch.from_numpy((self.weights[0].detach().numpy() - (self.lr * _w0)).astype(np.float32)).requires_grad_()
            updated_W = Parameter(u)
            self.weights[0] = updated_W

            u = torch.from_numpy((self.biases[1].detach().numpy() - (self.lr * _b1)).astype(np.float32)).requires_grad_()
            updated_W = Parameter(u)
            self.biases[1] = updated_W

            u = torch.from_numpy((self.biases[0].detach().numpy() - (self.lr * _b0)).astype(np.float32)).requires_grad_()
            updated_W = Parameter(u)
            self.biases[0] = updated_W

            print('-----------------------------------------------------------------------',i)

            outval = self.forward(Xval)
            Lval += [self.loss(outval, yval).item()]
            Aval += [np.array(outval.argmax(-1) == yval.argmax(-1)).mean()]

            if i > 100:
                break

        plt.plot(Ltrain)
        plt.show()
        plot = True
        if plot:
            plt.plot(Ltrain, label='Training loss')
            plt.plot( Lval, label='Validation loss')
            plt.plot( Aval, label='Validation acc')
            plt.legend()
            plt.show()

# This is already implemented for your convenience
class svm_sklearn():
    """ SVM via scikit-learn """
    def __init__(self, kernel='linear', kernelparameter=1., C=1.):
        if kernel == 'gaussian':
            kernel = 'rbf'
        self.clf = sklearn.svm.SVC(C=C,
                                   kernel=kernel,
                                   gamma=1./(1./2. * kernelparameter ** 2),
                                   degree=kernelparameter,
                                   coef0=kernelparameter)

    def fit(self, X, y):
        self.clf.fit(X, y)
        self.X_sv = X[self.clf.support_, :]
        self.y_sv = y[self.clf.support_]

    def predict(self, X):
        return self.clf.decision_function(X)

def plot_boundary_2d(X, y, model):
    print('X:{}, y:{}'.format(np.shape(X), np.shape(y)))
    def make_meshgrid(x, y, h=.05):
        min_x, max_x = x.min() - 1, x.max() + 1
        min_y, max_y = y.min() - 1, y.max() + 1
        x_, y_ = np.meshgrid(np.arange(min_x, max_x, h), np.arange(min_y, max_y, h))
        return x_, y_

    x_, y_ = make_meshgrid(X[:, 0], X[:, 1])
    def plot_contours(clf, x_, y_, **params):
        Z = clf.predict(np.c_[x_.ravel(), y_.ravel()])
        if len(np.shape(Z)) != 1:
            Z = np.squeeze(Z[:, 0])

        Z = Z.reshape(np.shape(x_))
        plt.contourf(x_, y_, Z, **params)

    plot_contours(model, x_, y_, alpha=0.9)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=20, edgecolors='k')
    plt.title('Boundary')
    plt.show()

if __name__ == '__main__':
    print('Main')

    '''test_plot_boundary_2d---------------------------
        X:(60, 2), y:(60,)
        Z  (21760,)
        x_  (128, 170)
        1'''