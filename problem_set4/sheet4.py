import scipy.linalg as la
import matplotlib.pyplot as plt
import sklearn.svm
import cvxopt
cvxopt.solvers.options['show_progress'] = False
from cvxopt.solvers import qp
from cvxopt import matrix as cvxmatrix
import numpy as np
import torch
from scipy.spatial.distance import cdist
import itertools as it
import time
from itertools import combinations, permutations
import pickle

from torch.optim import SGD #just to compare with our own implementation
from torch.nn import Module, Parameter, ParameterList
from torch.autograd import Variable

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

def plot_boundary_2d(X, y, model, title='Boundary'):
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
    if isinstance(model, svm_sklearn) or isinstance(model, svm_qp):
        plt.scatter(model.X_sv[:, 0], model.X_sv[:, 1], c='r', s=40, marker = 'x', edgecolors='k')
    plt.title(title)
    plt.show()

class neural_network(Module):
    def __init__(self, layers=[2,100,2], scale=.1, p=None, lr=None, lam=None, use_ReLU = True):
        super().__init__()
        self.weights = ParameterList([Parameter(scale*torch.randn(m, n)) for m, n in zip(layers[:-1], layers[1:])])
        self.biases = ParameterList([Parameter(scale*torch.randn(n)) for n in layers[1:]])

        self.layers = layers

        self.p = p
        self.lr = lr if lr is not None else 1e-3
        self.lam = lam if lam is not None else 0.0
        self.train = False
        self.L = len(self.weights)

        self.memory = {}
        self.use_ReLU = use_ReLU

    def relu(self, X, W, b, layer = None):
        if self.train:
            scalar = torch.FloatTensor([0])
            Z = X.mm(W) + b
            A = torch.max(Z, scalar.expand_as(Z))
            delta = torch.distributions.binomial.Binomial(probs=1 - self.p) # bernoulli
            delta = delta.sample(b.size()) * (1.0 / (1 - self.p))
            A = delta*A
            if layer is not None:
                self.memory['W_' + layer] = W
                self.memory['b_' + layer] = b
                self.memory['A_' + layer] = A
                self.memory['Z_' + layer] = Z
            return A
        else:
            scalar = torch.FloatTensor([0])
            Z = (1-self.p)*X.mm(W) + b
            A = torch.max(Z, scalar.expand_as(Z))
            if layer is not None:
                self.memory['W_' + layer] = W
                self.memory['b_' + layer] = b
                self.memory['A_' + layer] = A
                self.memory['Z_' + layer] = Z
            return A

    def softmax(self, X, W, b, layer = None):
        Z = X.mm(W) + b
        exp = torch.exp(Z)
        Y_hat = exp/torch.sum(exp, dim=1, keepdim=True)
        if layer is not None:
            self.memory['W_' + layer] = W
            self.memory['b_' + layer] = b
            self.memory['A_' + layer] = Y_hat
            self.memory['Z_' + layer] = Z
        return Y_hat

    def forward(self, X):
        self.memory = {}
        X = torch.tensor(X, dtype=torch.float)
        A = X
        for l in range(self.L-1):
            A = self.relu(X=A,W=self.weights[l],b=self.biases[l], layer=str(l+1)) if self.use_ReLU else self.Sigmoid(X=A,W=self.weights[l],b=self.biases[l], layer=str(l+1))
        Y_hat = self.softmax(X=A,W=self.weights[-1], b=self.biases[-1], layer=str(self.L))
        return Y_hat

    def predict(self, X):
        return self.forward(X).detach().numpy()

    def loss(self, ypred, ytrue):
        cross_entropy = -(ytrue * ypred.log() + (1 - ytrue) * (1 - ypred).log()).mean()
        if self.lam > 0: #L2 regularization
            n = np.shape(ytrue)[0]
            L2_regularization = 0
            for l in range(self.L):
                w = self.weights[l].detach().clone().numpy()
                L2_regularization += np.sum(np.square(w))

            L2_regularization = (self.lam / (2 * n))*L2_regularization
            cost = cross_entropy + L2_regularization
        else:
            cost = cross_entropy

        return cost

    def relu_derivative(self, X):
        y = X.detach().clone()
        y[y >= 0] = 1
        y[y < 0] = 0
        return y

    def Sigmoid(self, X, W, b, layer = None):
        Z = X.mm(W) + b
        scalar = torch.FloatTensor([1]).expand_as(Z)
        A = scalar/(scalar + torch.exp(-Z))
        if layer is not None:
            self.memory['W_' + layer] = W
            self.memory['b_' + layer] = b
            self.memory['A_' + layer] = A
            self.memory['Z_' + layer] = Z
        return A

    def SigmoidDerivative(self, X):
        x = X.detach().clone().numpy()
        A =  np.multiply(x, 1 - x)
        x = torch.from_numpy(A.astype(np.float32)).requires_grad_()
        return x

    def softmax_derivative(self, s):
        s = s[0]
        Jacobian_matrix = torch.diag(s)
        for i in range(len(Jacobian_matrix)):
            for j in range(len(Jacobian_matrix)):
                if i == j:
                    Jacobian_matrix[i][j] = s[i] * (1 - s[i])
                else:
                    Jacobian_matrix[i][j] = -s[i] * s[j]
        return Jacobian_matrix

    def backpropagation(self, X, Y, bs):
        derivates = {}

        self.memory["A_0"] = X
        Y = Y.detach().clone().numpy()
        A = (self.memory["A_" + str(self.L)]).detach().clone().numpy()

        #dA = -np.divide(Y, A) + np.divide(1 - Y, 1 - A) #normal cross entropy derivative
        dA = A - Y                                       #stable cross entropy derivative

        if self.use_ReLU:
            dZ = (dA * self.relu_derivative(self.memory["Z_" + str(self.L)]).detach().clone().numpy()).T  #with relu derivative
        else:
            dZ = (dA * self.SigmoidDerivative(self.memory["Z_" + str(self.L)]).detach().clone().numpy()).T

        dW = dZ.dot(self.memory["A_" + str(self.L - 1)].detach().clone().numpy())
        db = np.sum(dZ, axis=1, keepdims=True)

        dAPrev = self.memory["W_" + str(self.L)].detach().clone().numpy().dot(dZ).T

        if self.lam > 0:  # L2 regularization
            derivates["dW" + str(self.L)] = dW + ((self.lam/bs)* self.weights[-1].detach().clone().numpy()).T
        else:
            derivates["dW" + str(self.L)] = dW

        derivates["db" + str(self.L)] = db

        for l in range(self.L - 1, 0, -1):
            #print('l is {}=--------------------------------'.format(l))
            if self.use_ReLU:
                dZ = (dAPrev * self.relu_derivative(self.memory["Z_" + str(l)]).detach().clone().numpy()).T
            else:
                dZ = (dAPrev * self.relu_derivative(self.memory["Z_" + str(l)]).detach().clone().numpy()).T

            dW = dZ.dot(self.memory["A_" + str(l - 1)].detach().clone().numpy())
            db = np.sum(dZ, axis=1, keepdims=True)

            if l > 1:
                dAPrev = self.memory["W_" + str(l)].detach().clone().numpy().dot(dZ).T

            if self.lam > 0:  # L2 regularization
                derivates["dW" + str(l)] = dW + ((self.lam/bs)* self.weights[l-1].detach().clone().numpy()).T
            else:
                derivates["dW" + str(l)] = dW

            derivates["db" + str(l)] = db
        return derivates

    def SGD_updates(self,derivates, bs, normalize=False):
        for l in range(1, self.L + 1):
            # print('level ',l)
            if normalize:
                updated_w = self.weights[l - 1].detach().numpy() - self.lr * derivates['dW' + str(l)].T
                updated_b = self.biases[l - 1].detach().numpy() - np.squeeze(self.lr * derivates['db' + str(l)])
            else:
                updated_w = self.weights[l - 1].detach().numpy() - (self.lr * derivates['dW' + str(l)]/bs).T
                updated_b = self.biases[l - 1].detach().numpy() - np.squeeze((self).lr * derivates['db' + str(l)]/bs)

            self.weights[l - 1] = Parameter(torch.from_numpy(updated_w.astype(np.float32)).requires_grad_())
            self.biases[l - 1] = Parameter(torch.from_numpy(updated_b.astype(np.float32)).requires_grad_())

    #backpropagation & weights update from scratch
    def fit(self, X, y, nsteps=1000, bs=100, plot=False):
        #np.random.seed(1)
        torch.manual_seed(1)
        X, y = torch.tensor(X), torch.tensor(y)
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
            #print('Cost ', output)
            #print(self.memory.keys())

            derivates = self.backpropagation(Xtrain[I], ytrain[I], bs)
            #print(derivates.keys())
            self.SGD_updates(derivates,bs,True)

            outval = self.forward(Xval)
            Lval += [self.loss(outval, yval).item()]
            Aval += [np.array(outval.argmax(-1) == yval.argmax(-1)).mean()]

        if plot:
            plt.plot(range(nsteps), Ltrain, label='Training loss '+str(round(Ltrain[-1],4)))
            plt.plot(range(nsteps), Lval, label='Validation loss '+str(round(Lval[-1],4)))
            plt.plot(range(nsteps), Aval, label='Validation acc '+str(round(Aval[-1],4)))
            plt.legend()
            plt.show()

        return Ltrain, Lval, Aval

    # this is using implemented SGD
    def fit_(self, X, y, nsteps=1000, bs=100, plot=False):
        print('here')
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
            plt.plot(range(nsteps), Ltrain, label='Training loss ' + str(round(Ltrain[-1], 4)))
            plt.plot(range(nsteps), Lval, label='Validation loss ' + str(round(Lval[-1], 4)))
            plt.plot(range(nsteps), Aval, label='Validation acc ' + str(round(Aval[-1], 4)))
            plt.legend()
            plt.show()
        return Ltrain, Lval, Aval

def buildKernel(X, X_train=None, kernel='linear', kernelparameter=0):
    if len(np.shape(X))==1:
        X = X[:, np.newaxis].T
    if X_train is None:
        X_train = X.copy()
    if len(np.shape(X_train))==1:
        X_train = X_train[:, np.newaxis].T
    if kernel == 'linear':
        K = X_train @ X.T
    elif kernel == 'polynomial':
        K = (X_train @ X.T + 1)**kernelparameter
    elif kernel == 'gaussian':
        K = cdist(X_train, X, 'euclidean')
        K = np.exp(-(K ** 2) / (2. * kernelparameter ** 2))
    else:
        raise Exception('unspecified kernel')
    return K

class svm_qp():
    def __init__(self, kernel='linear', kernelparameter=1., C=1.):
        self.kernel = kernel
        self.kernelparameter = kernelparameter
        self.C = C
        self.alpha_sv = None
        self.b = None
        self.X_sv = None
        self.Y_sv = None

    def fit(self, X, Y):
        #print('X:{},Y:{}'.format(np.shape(X), np.shape(Y)))
        m, n = X.shape

        K = buildKernel(X=X, X_train=X, kernel=self.kernel, kernelparameter=self.kernelparameter)
        P = np.outer(Y, Y) * K
        q = -np.ones(m)
        A = Y.reshape(1, -1)
        b = np.zeros(1)

        if self.C is None:
            G = np.diag(np.ones(m) * -1)
            h = np.zeros(m)
        else:
            diag = np.diag(np.ones(m) * -1)
            identity = np.identity(m)
            G = np.vstack((diag, identity))
            zeros = np.zeros(m)
            ones = np.ones(m) * self.C
            h = np.hstack((zeros, ones))

        alpha = np.array(qp(cvxmatrix(P, tc='d'),
                            cvxmatrix(q, tc='d'),
                            cvxmatrix(G, tc='d'),
                            cvxmatrix(h, tc='d'),
                            cvxmatrix(A, tc='d'),
                            cvxmatrix(b, tc='d'))['x']).flatten()

        suport_vectors = alpha > 1e-4 #keep only non zero alpha
        idx = np.arange(len(alpha))[suport_vectors]
        self.alpha_sv = alpha[suport_vectors]
        self.X_sv = X[suport_vectors]
        self.Y_sv = Y[suport_vectors]
        #print('{} SV from {}'.format(len(self.alpha_sv), m))

        self.b = 0.0
        for i in range(len(self.alpha_sv)):
            self.b += self.Y_sv[i]
            self.b -= np.sum(self.alpha_sv * self.Y_sv * K[idx[i], suport_vectors])
        if len(self.alpha_sv) > 0:
            self.b = self.b / len(self.alpha_sv)

    def predict(self, X, without_sign=False):
        ypred = np.zeros(len(X))
        for i in range(len(X)):
            for vector, vector_y, vector_x in zip(self.alpha_sv, self.Y_sv, self.X_sv):
                k = buildKernel(X=X[i], X_train=vector_x, kernel=self.kernel, kernelparameter=self.kernelparameter)
                ypred[i] += vector * vector_y * k
        prediction = ypred + self.b

        if without_sign:
            return prediction

        return np.sign(prediction)

def loss_function(Y_pred,Y_te):
    loss = float(np.sum(np.sign(Y_te) != np.sign(Y_pred))) / float(len(Y_te))
    return loss

def cv(X, y, method, params, loss_function=loss_function, nfolds=10, nrepetitions=5,roc_f=False, keep_bad=False):
    def updateProgress(total, progress, details, remain_time):
        length, msg = 100, ""
        progress = float(progress) / float(total)
        if progress >= 1.:
            progress, msg = 1, "\r\n"
        diez = int(round(length * progress))
        print(details)
        text = "\r[{}] {:.0f}% {}, remain:{}".format("#" * diez + "-" * (length - diez), round(progress * 100, 0), msg,
                                                     remain_time)
        print(text)
        print()

    last_times = []
    def get_remaining_time(step, total, time):
        last_times.append(time)
        len_last_t = len(last_times)
        if len_last_t > 120:
            last_times.pop(0)
        mean_time = np.mean(last_times)
        remain_s_tot = mean_time * (total - step + 1)
        minutes = remain_s_tot // 60
        seconds = remain_s_tot % 60
        return "{}m {}s".format(minutes, seconds)

    n, d = X.shape
    indexes = np.arange(0, n)
    folds_elements_rounded = int(n/nfolds) # elements of the  1 to (k-1) folds except the last fold

    # Create list with all possible params combinations
    combinations_params = list(it.product(*(params[Name] for Name in list(params))))
    n_combinations = len(combinations_params)
    print('Total combinations: {}'.format(n_combinations))
    # Init error function containing the average loss for every parameters combination
    error_combinations = np.zeros(n_combinations)

    # For loop over all possible params combinations
    runs = n_combinations * nrepetitions
    progress = 0
    for cur_index, local_parameter in enumerate(combinations_params):
        current_time = time.time()
        e_r_error = 0
        method_fold = method(*local_parameter)  # local_parameter is a list containing the parameters values used.
        for i in range(nrepetitions):
            indexes_shuffled = indexes.copy()
            np.random.shuffle(indexes_shuffled)
            e_cv_error = 0
            for cur_fold in range(nfolds):
                if cur_fold == (nfolds - 1):
                    test_fold_indexes = indexes_shuffled[(cur_fold * folds_elements_rounded):]
                    train_fold_indexes = np.delete(indexes_shuffled,
                                                   np.nonzero(np.isin(indexes_shuffled, test_fold_indexes)))
                else:
                    test_fold_indexes = indexes_shuffled[(cur_fold * folds_elements_rounded):(
                                cur_fold * folds_elements_rounded + folds_elements_rounded)]
                    train_fold_indexes = np.delete(indexes_shuffled,
                                                   np.nonzero(np.isin(indexes_shuffled, test_fold_indexes)))

                Y_test = y[test_fold_indexes]
                X_test = X[test_fold_indexes, :]

                X_train = X[train_fold_indexes, :]
                Y_train = y[train_fold_indexes]

                # Train Model on the fold training data and test it
                method_fold.fit(X_train, Y_train)
                y_pred = method_fold.predict(X_test) if roc_f==False else method_fold.predict(X_test,without_sign=True)

                # Sum Error
                e_cv_error += loss_function(Y_test, y_pred)

            e_r_error += (e_cv_error/nfolds)
            last_t = time.time() - current_time

            details = "kernel:{},kernelparam:{},regularizer:{}".format(local_parameter[0], local_parameter[1], local_parameter[2])
            updateProgress(runs, progress + 1, details, get_remaining_time(progress + 1, runs, last_t))
            progress += 1

        final_error = e_r_error / nrepetitions
        if roc_f==False:
            error_combinations[cur_index] = final_error

    if roc_f == False:
        if keep_bad:
            best_param_combination = combinations_params[np.argmax(error_combinations)]
        else:
            best_param_combination = combinations_params[np.argmin(error_combinations)]

        best_method = method(*best_param_combination)
        best_method.cvloss = np.min(error_combinations)
        best_method.fit(X, y)  # Train best model on all the dataset

        print('best method:', best_param_combination)
        print('best method loss: {}'.format(best_method.cvloss))
    else:
        best_method = method(*combinations_params)  # combinations_params are already the best
        best_method.tp = final_error[0]
        best_method.fp = final_error[1]

    return best_method

def roc_fun(y_true, y_hat, points = 2500, threshold = 0.0):
    y_hat = y_hat[:, np.newaxis]

    Negative = np.array(y_true.flatten() == -1).sum() #negative class
    Positive = len(y_true) - Negative                 #positive class
    assert Positive>0 and Negative>0, 'Negative or Positive is zero, zero division exception (No positive or negative classes, Inbalanced data)'

    B = np.linspace(-10, 10, points)[np.newaxis, :]
    PRED = (y_hat - B) > threshold
    TPR = PRED[y_true.flatten() == 1, :].sum(axis=0) / Positive
    FPR = PRED[y_true.flatten() == -1, :].sum(axis=0) / Negative

    return np.squeeze(np.array([TPR, FPR]))

def plot_roc_curve(fpr, tpr, title, aucScore):
    plt.plot(fpr, tpr, label='ROC - AUC:'+str(aucScore))
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title(title)
    plt.legend()
    plt.show()

def Assignment4():
    dataset = np.load('data/easy_2d.npz')
    print(dataset.files)
    X_tr = dataset['X_tr'].T
    Y_tr = dataset['Y_tr'].T
    X_te = dataset['X_te'].T
    Y_te = dataset['Y_te'].T
    print('(X_tr:{},  Y_tr:{}), (X_te:{}, Y_te:{})'.format(np.shape(X_tr), np.shape(Y_tr), np.shape(X_te), np.shape(Y_te)))

    fig, ax = plt.subplots(2, 1, figsize=(10, 10))
    ax[0].scatter(X_te[:, 0], X_te[:, 1], c=Y_te, s=50, cmap=plt.cm.RdYlGn)
    ax[0].set_title('Testing data')
    ax[1].scatter(X_tr[:, 0], X_tr[:, 1], c=Y_tr,s=50, cmap=plt.cm.RdYlGn)
    ax[1].set_title('Training data')
    plt.show()

    reg = list(np.linspace(1, 500, num=20))
    reg.append(None)
    params = {'kernel': ['gaussian'], 'kernelparameter': [.1, .5, .9, 1.0, 2.0, 3.0],
              'regularization': reg}

    CV = cv(X_tr, Y_tr, svm_qp, params, loss_function=loss_function, nrepetitions=2)
    #Perfect fit
    print('***************Best results**********************')
    cvloss, kernel, kernelparameter, C = CV.cvloss, CV.kernel, CV.kernelparameter, CV.C
    print('cvloss:{}, kernel:{}, kernelparameter:{} , C:{}'.format(cvloss,kernel,kernelparameter,C))
    Y_pred = CV.predict(X_tr)
    loss = float(np.sum(np.sign(Y_tr) != np.sign(Y_pred))) / float(len(Y_tr))
    plot_boundary_2d(X_tr, Y_tr, CV, 'Train data perfect fit, loss:'+str(loss))
    print('Train data, perfect fit loss', loss)

    Y_pred = CV.predict(X_te)
    loss = float(np.sum(np.sign(Y_te) != np.sign(Y_pred))) / float(len(Y_te))
    plot_boundary_2d(X_te, Y_te, CV, 'Test data perfect fit, loss:'+str(loss))
    print('Test data, perfect fit loss', loss)

    #Underfit fit
    print('***************Underfit results**********************')
    #model = svm_qp(kernel='linear', C=None)
    model = svm_qp(kernel='gaussian', kernelparameter=7, C=None)
    model.fit(X_tr, Y_tr)
    Y_pred = model.predict(X_tr)
    loss = float(np.sum(np.sign(Y_tr) != np.sign(Y_pred))) / float(len(Y_tr))
    plot_boundary_2d(X_tr, Y_tr, model, 'Underfit - Train data, loss:'+str(loss))
    print('Train data loss', loss)

    Y_pred = model.predict(X_te)
    loss = float(np.sum(np.sign(Y_te) != np.sign(Y_pred))) / float(len(Y_te))
    plot_boundary_2d(X_te, Y_te, model, 'Underfit - Test data, loss:'+str(loss))
    print('Test data loss', loss)

    # Overfit fit
    print('***************Overfit results**********************')
    model = svm_qp(kernel='gaussian', kernelparameter=.01, C=100000)
    model.fit(X_tr, Y_tr)
    Y_pred = model.predict(X_tr)
    loss = float(np.sum(np.sign(Y_tr) != np.sign(Y_pred))) / float(len(Y_tr))
    plot_boundary_2d(X_tr, Y_tr, model, 'Overfit - Train data, loss:'+str(loss))
    print('Train data loss', loss)

    Y_pred = model.predict(X_te)
    loss = float(np.sum(np.sign(Y_te) != np.sign(Y_pred))) / float(len(Y_te))
    plot_boundary_2d(X_te, Y_te, model, 'Overfit - Test data, loss:'+str(loss))
    print('Test data loss', loss)

    #-ROC-------------------------------------
    best_params = {'kernel': [CV.kernel], 'kernelparameter': [CV.kernelparameter],
                   'regularization': [CV.C]}
    CV = cv(X_tr, Y_tr, svm_qp, best_params, loss_function=roc_fun, nfolds = 10, nrepetitions=2, roc_f = True)
    AUC = round((np.abs(np.trapz(CV.tp, CV.fp))), 3)
    plot_roc_curve(CV.fp, CV.tp, '', AUC)

def Assignment5():
    dataset = np.load('data/iris.npz')
    print(dataset.files)
    X = dataset['X'].T
    Y = dataset['Y'].T
    print('X:{}, Y:{}'.format(np.shape(X), np.shape(Y)))
    print(Y)
    import  pandas as pd
    import seaborn as sns
    X_ = np.hstack((X, Y[:, np.newaxis]))
    data=pd.DataFrame(X_, columns=['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm','class'])
    sns.pairplot( data=data, vars=('SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm'), hue='class' )
    plt.show()

    #test with SVM-----------------------------------------------
    reg = list(np.linspace(1, 500, num=20))
    reg.append(None)
    params = {'kernel': ['polynomial', 'gaussian'], 'kernelparameter': [1., 2., 3.],
              'regularization': reg}
    from sklearn.model_selection import train_test_split
    from sklearn import metrics

    #split dataset in all possible combinations of one vs others
    dts = list(combinations([1, 2, 3], 2))
    for positive_class in reversed(dts):
        negative_class = (set([1, 2, 3]) - set(positive_class)).pop()
        print('Possitive:{} VS Negative:{}'.format(positive_class,negative_class))

        idx_negative = Y==negative_class
        idx_positive = Y != negative_class

        Y_data = Y.copy()
        Y_data[idx_positive] = 1
        Y_data[idx_negative] = -1
        X_data = X.copy()
        print('Y_data ',Y_data)
        # Split dataset into training set and test set
        X_train, X_test, y_train, y_test = train_test_split(X_data, Y_data, test_size=0.3,
                                                            random_state=100)

        print('X_tr:{}, Y_tr:{}, X_te:{}, Y_te:{}'.format(np.shape(X_train), np.shape(y_train), np.shape(X_test), np.shape(y_test)))

        # cv for each of the combination
        CV = cv(X_train, y_train, svm_qp, params, loss_function=loss_function,nfolds=20, nrepetitions=1)
        Y_pred = CV.predict(X_test)
        loss = loss_function(Y_pred, y_test)

        print('********************Results**********************')
        print("Accuracy:", metrics.accuracy_score(y_test, Y_pred))

        print('Possitive:{} VS Negative:{}'.format(positive_class, negative_class))
        cvloss, kernel, kernelparameter, C = CV.cvloss, CV.kernel, CV.kernelparameter, CV.C
        print('cvloss:{}, kernel:{}, kernelparameter:{}, C:{}'.format(cvloss, kernel, kernelparameter, C))
        print('Test loss ', loss)

        myX = X_data[:,:2].copy()
        CV.fit(myX, Y_data)

        # -ROC-------------------------------------
        best_params = {'kernel': [CV.kernel], 'kernelparameter': [CV.kernelparameter],
                       'regularization': [CV.C]}
        plot_boundary_2d(myX, Y, CV, 'CV model  {} VS {}, cvloss:{}'.format(positive_class,negative_class,cvloss))
        CV = cv(X_data, Y_data, svm_qp, best_params, loss_function=roc_fun, nrepetitions=2, roc_f=True)
        AUC = round((np.abs(np.trapz(CV.tp, CV.fp))), 3)
        plot_roc_curve(CV.fp, CV.tp, 'CV SVM  {} VS {}'.format(positive_class,negative_class), AUC)

        C = svm_qp(kernel='linear', C=1.)
        C.fit(myX,Y_data)
        Y_pred = C.predict(X_data[:,:2])
        loss = float(np.sum(np.sign(Y_data) != np.sign(Y_pred))) / float(len(Y_data))
        plot_boundary_2d(myX, Y, C, 'linear kernel  {} VS {}, loss:{}'.format(positive_class,negative_class,loss))
        best_params = {'kernel': ['linear'], 'kernelparameter': [1],
                       'regularization': [1]}
        CV = cv(X_data, Y_data, svm_qp, best_params, loss_function=roc_fun, nrepetitions=2, roc_f=True)
        AUC = round((np.abs(np.trapz(CV.tp, CV.fp))), 3)
        plot_roc_curve(CV.fp, CV.tp, 'linear kernel  {} VS {} SVM'.format(positive_class,negative_class), AUC)

        print('-------------------------------------------------------------------\n\n')
    print('class 2 and 3 arent linearly separable')
    from sklearn.preprocessing import StandardScaler
    X_scaled = StandardScaler().fit_transform(X)
    split_classes_and_plot_SVM_results(X_scaled, Y)

    #test with NN --------------------------------------------------------
    print('Predict with Neral Network')
    process_Iris_Dataset_with_Neural_Networks()

def process_Iris_Dataset_with_Neural_Networks():
    dataset = np.load('data/iris.npz')
    print(dataset.files)
    X = dataset['X'].T
    Y = dataset['Y'].T
    Y_copy = Y.copy()
    print('X:{}, Y:{}'.format(np.shape(X), np.shape(Y)))
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    new_y = []
    for i in Y:
        a = [0, 0, 0]
        a[int(i) - 1] = 1
        new_y.append(a)
    Y = np.array(new_y)
    #scale data
    X_scaled = StandardScaler().fit_transform(X)

    # Split the data set into training and testing
    X_train, X_test, Y_train, Y_test = train_test_split(
        X_scaled, Y, test_size=0.2, random_state=2)
    np.random.seed(11)
    torch.manual_seed(11)

    n_features = X.shape[1]
    n_classes = Y.shape[1]
    print('n_features:{}, n_classes:{}'.format(n_features,n_classes))
    print('X_train ',np.shape(X_train))
    print('Y_train ', np.shape(Y_train))
    learning_rate = 0.1
    def create_model(input_dim, output_dim, nodes, layer=1, name='model'):
        layers = [input_dim]
        for i in range(layer):
            layers.append(nodes)
        layers.append(output_dim)
        model = neural_network(layers=layers, lr=learning_rate, p=.05, lam=.05, use_ReLU=True)
        model.name = name
        return model

    models = [create_model(n_features, n_classes, 16, i, 'model_{}'.format(i))
              for i in range(1, 6)]
    for created_model in models:
        print('Model :{}, {} layers'.format(created_model.name, len(created_model.layers)))

    history_dict = {}
    print('Training models')
    for model in models:
        Ltrain, Lval, Aval = model.fit(X_train, Y_train, nsteps=50, bs=10, plot=False)
        history_dict[model.name] = [Ltrain, Lval, Aval, model]

    fig, (ax1, ax2) = plt.subplots(2, figsize=(10, 8))
    for model_name in history_dict:
        val_acc = history_dict[model_name][2]
        val_loss = history_dict[model_name][1]
        ax1.plot(val_acc, label=model_name)
        ax2.plot(val_loss, label=model_name)
        print('Model :{},  Val loss :{} ,  Val Accuracy :{}'.format(model_name, val_loss[-1],val_acc[-1]))

    ax1.set_ylabel('val accuracy')
    ax2.set_ylabel('val loss')
    ax2.set_xlabel('epochs')
    ax1.legend()
    ax2.legend()
    plt.show()

    from sklearn.metrics import roc_curve, auc
    plt.figure(figsize=(10, 8))
    plt.plot([0, 1], [0, 1], 'k--')
    for model_name in history_dict:
        model = history_dict[model_name][3]

        Y_pred = model.predict(X_test)
        fpr, tpr, threshold = roc_curve(Y_test.ravel(), Y_pred.ravel())

        plt.plot(fpr, tpr, label='{}, AUC = {:.3f}'.format(model_name, auc(fpr, tpr)))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend()
    plt.show()

    Y_pred = models[0].predict(X_scaled)
    split_classes_and_plot_NN_results(X_scaled, Y_pred)

def split_classes_and_plot_SVM_results(X,Y):
    print('X ', np.shape(X))
    print('Y ', np.shape(Y))
    dts = list(permutations([0, 1, 2, 3], 2))
    print('dts ',dts)
    columns = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
    _, axs = plt.subplots(4, 4,figsize=(15,15))

    n_classes = 3
    plot_colors = "rbg"
    plot_colors_arr = ["r", "b", "g"]
    for j in range(len(dts)):
        cls = dts[j]

        for ran in range(3):
            axs[cls[0], cls[0]].scatter([0.], [(ran - 1) * .1], c=plot_colors_arr[ran], s=90)
            axs[cls[0], cls[0]].annotate('class {}'.format(ran + 1), (0.005, (ran - 1) * .1))

        axs[cls[0], cls[0]].set_xticks([])
        axs[cls[0], cls[0]].set_yticks([])

        pair = cls
        X_pair = X[:, pair]

        clf = svm_sklearn(kernel='gaussian', C=1.)
        clf.fit(X_pair, Y)

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
            axs[cls[1], cls[0]].contourf(x_, y_, Z, **params)

        plot_contours(clf, x_, y_, alpha=0.9)
        for i, color in zip(range(n_classes), plot_colors):
            idx = np.where(Y == (i+1))
            axs[cls[1], cls[0]].scatter(X_pair[idx, 0], X_pair[idx, 1], c=color,cmap=plt.cm.RdYlBu, edgecolor='black', s=40)

        axs[cls[1], cls[0]].scatter(clf.X_sv[:, 0], clf.X_sv[:, 1], c='r', s=40, marker='x', edgecolors='k')

    for ax, col in zip(axs[0], columns):
        ax.set_title(col)
    for ax, row in zip(axs[:, 0], columns):
        ax.set_ylabel(row, rotation='vertical', size='large')

    plt.show()

def split_classes_and_plot_NN_results(X,Y_pred):
    print('X ', np.shape(X))
    print('Y ', np.shape(Y_pred))
    dts = list(permutations([0, 1, 2, 3], 2))
    print('dts ',dts)
    columns = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
    fig, axs = plt.subplots(4, 4,figsize=(15,15))
    plot_colors = ["r","b","g"]
    for j in range(len(dts)):
        cls = dts[j]
        axs[cls[1], cls[0]].scatter(X[:,cls[0]],X[:,cls[1]], c = Y_pred, s=60)
        for ran in range(3):
            axs[cls[0], cls[0]].scatter([0.], [(ran-1)*.1], c=plot_colors[ran], s = 90)
            axs[cls[0], cls[0]].annotate('class {}'.format(ran+1), (0.005, (ran-1)*.1))

        axs[cls[0], cls[0]].set_xticks([])
        axs[cls[0], cls[0]].set_yticks([])

    for ax, col in zip(axs[0], columns):
        ax.set_title(col)
    for ax, row in zip(axs[:, 0], columns):
        ax.set_ylabel(row, rotation='vertical', size='large')

    plt.show()

def zero_one_loss_multiple(y_pred, y_true):
    # Computes zero one loss for a multi classification task : add 0 for correct classification
    #    and 1 for an incorrect classification.
    #    Input:    y_pred      - torch array nxK with n predictions for k-possible classifications
    #              y_true      - torch array nxK with correct classifcations as 1
    #   Output:    loss        - float, zero one loss
    
    n = y_true.shape[0]
    k = y_true.shape[1]
    Z = torch.zeros((n, k))
    Z[torch.arange(n), torch.argmax(y_pred, 1)] = 1

    loss = (n - float(torch.sum(Z*y_true)))/n

    return loss

def zero_one_loss(y_pred, y_true):
    #Compute zero one loss function value for binary classification
    #Definition:  apply_pca(X, m)
    #Input:       y_pred                  - Nx1 numpy array, predicted values
    #             y_true                  - Nx1 numpy arrays , true classification values
    #Output:      loss                    - float, zero one loss function
    
    y_pred_sign = np.sign(y_pred)
    y_pred_sign[y_pred_sign == 0] = -1
    loss = (1/len(y_pred))*len(np.argwhere((y_true * y_pred_sign < 0)))
    return loss

def cross_entropy_loss(ypred, ytrue):
    cross_entropy = -(ytrue * ypred.log() + (1 - ytrue) * (1 - ypred).log()).mean()
    return cross_entropy

def Assignment6():
    # Which models to perform cv or train
    neural_net = True  # To train neural networks
    svm = False  # To train SVM for each digit binary class problem
    apply_svm = False  # To load already trained models and predict by comparing each models output

    # Load training and test data
    train_data = torch.load(
        r'/Users/albertorodriguez/Desktop/Current Courses/ML_Lab/4 Topic/problem_set4/stubs/MNIST/processed/training.pt')
    test_data = torch.load(
        r'/Users/albertorodriguez/Desktop/Current Courses/ML_Lab/4 Topic/problem_set4/stubs/MNIST/processed/test.pt')

    X_train = train_data[0]
    y_train = train_data[1]

    X_te = test_data[0]
    y_te = test_data[1]

    n_tr = X_train.shape[0]
    n_te = X_te.shape[0]

    # Build ytrain ones matrix == One-hot-encode

    y_train_ones = torch.zeros((X_train.shape[0], 10))
    y_train_ones[torch.arange(X_train.shape[0]), y_train] = 1

    y_test_ones = torch.zeros((X_te.shape[0], 10))
    y_test_ones[torch.arange(X_te.shape[0]), y_te] = 1
    # Reshape Images
    X_train = X_train.reshape((n_tr, 28 * 28))
    X_te = X_te.reshape((n_te, 28 * 28))

    # Avoid 0 as input by mapping pixel values to range 0.01-1 (Re-scaling data)
    X_train = X_train * 0.99 / 255 + 0.01
    X_te = X_te * 0.99 / 255 + 0.01

    print('X train shape: {}, X_test shape: {}'.format(X_train.shape, X_te.shape))
    print('Y_train shape: {}'.format(y_train_ones.shape))
    print('Classes: {}'.format(torch.unique(y_train)))

    # NEURAL NETWORK

    if neural_net:
        # best_model = imp.neural_network(layers=[28 * 28, 100, 100, 10], scale=.1, p=0.1, lr=0.1, lam=1e-4)
        # best_model.fit(X_train, y_train_ones, nsteps=3000, bs=168, plot=True)

        # Set epochs number and batchsizenumber

        # params = {'layers': [[784, 32, 10]], 'scale': [.1], 'p': [0.1, 0.5, 0.8],
        #          'lr': [0.01, 0.1, 0.5], 'lam': np.logspace(-3, 2, 5)}

        params = {'layers': [[784, 32, 10]], 'scale': [.1], 'p': [0, 0.1, 0.5],
                  'lr': [0.01, 0.1, 0.5], 'lam': [0, 0.1, 1]}
        # CV
        best_model = cv(X=X_train, y=y_train_ones, method=neural_network, params=params,
                            loss_function=zero_one_loss_multiple, nfolds=5, nrepetitions=1, roc_f=False,
                            verbose=True)

        # Compute error on the test set
        y_te_pred = torch.as_tensor(best_model.predict(X_te))
        loss_acc = zero_one_loss_multiple(y_pred=y_te_pred, y_true=y_test_ones)
        loss_test = cross_entropy_loss(ypred=y_te_pred, ytrue=y_test_ones)

        print('Best Model Test loss: {}'.format(loss_test))
        print('Best Model Test acuracy: {}'.format(1 - loss_acc))

        # Plot weight vectors
        fig, axes = plt.subplots(10, 10)
        # weigths global min / max -> all images on the same scale
        vmin, vmax = float(best_model.weights[0].min()), float(best_model.weights[0].max())

        weights_first_layer = best_model.weights[0].data.numpy()

        for coef, ax in zip(weights_first_layer.T, axes.ravel()):
            ax.matshow(coef.reshape(28, 28), cmap=plt.cm.gray, vmin=.5 * vmin,
                       vmax=.5 * vmax)
            ax.set_xticks(())
            ax.set_yticks(())

        plt.show()

    if svm:
        # One vs rest classification
        cur_digits = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        # cur_digits = [0]
        y_train_np = y_train.data.numpy()
        X_train_np = X_train.data.numpy()
        X_test_np = X_te.data.numpy()
        y_test_np = y_te.data.numpy()

        X_train_np = X_train_np[:, :]
        y_train_np = y_train_np[:]

        for index, cur_digit in enumerate(cur_digits):
            y_train_cur = y_train_np.copy()
            y_test_cur = y_test_np.copy()

            y_train_cur[y_train_np == cur_digit] = 1

            y_train_cur[y_train_np != cur_digit] = -1

            y_test_cur[y_te == cur_digit] = 1
            y_test_cur[y_te != cur_digit] = -1

            # svm_model = svm_qp(kernel='poly', kernelparameter=3, C= 0.001)
            # params = {'kernel': ['poly'], 'kernelparameter':[2, 3, 4], 'C': np.log(-2,2,5)}
            # best_model = imp.cv(X=X_train, y=y_train_ones, method=imp.svm_sklearn, params=params,
            #                    loss_function=imp.zero_one_loss_multiple, nfolds=5, nrepetitions=1, roc_f=False,
            #                    verbose=True)

            svm_model = svm_sklearn(kernel='poly', kernelparameter=3, C=0.001)
            svm_model.fit(X_train_np, y_train_cur)
            # print(svm_model.X_sv.shape)
            # print('fit ready')
            y_pred_tr = svm_model.predict(X_train_np)
            y_pred_test = svm_model.predict(X_test_np)

            print(len(np.argwhere(np.sign(y_pred_test) == -1)))
            # Compute accuracy on train and test data
            accuracy_train = 1 - zero_one_loss(y_pred=np.sign(y_pred_tr), y_true=y_train_cur)

            accuracy_test = 1 - zero_one_loss(y_pred=np.sign(y_pred_test), y_true=y_test_cur)

            # Compute true pos rate
            pos_all = len(np.argwhere(y_test_cur == 1))
            true_pos = len(np.argwhere(y_test_cur[y_test_cur == 1] * np.sign(y_pred_test[y_test_cur == 1]) == 1))
            true_pos_rate = true_pos / pos_all

            print('Cur Digit: {}'.format(cur_digit))
            print('True Pos Rate in percentage: {}'.format(true_pos_rate))
            print('One vs Rest SVM Classification for digit: {}'.format(cur_digit))
            print('Train Accuracy: {}, Test Accuracy: {}'.format(accuracy_train, accuracy_test))

            # SAVE MODEL
            filename = 'finalized_model_svm_digit_' + str(cur_digit) + '.sav'
            pickle.dump(svm_model, open(filename, 'wb'))

            # plot 5 random support vectors for each class == 10 figures per digit
            X_sv = svm_model.X_sv.T
            y_sv = svm_model.y_sv
            # Randomly choose 5 support vectors for each class
            X_sv_pos = X_sv[:, y_sv == 1]
            X_sv_neg = X_sv[:, y_sv == -1]

            indexes_pos = np.random.choice(np.arange(X_sv_pos.shape[1]), 5, replace=False)
            indexes_neg = np.random.choice(np.arange(X_sv_neg.shape[1]), 5, replace=False)

            X_random_pos = X_sv_pos[:, indexes_pos]  # dx5 array
            X_random_neg = X_sv_neg[:, indexes_neg]  # dx5 array

            if X_sv.shape[1] >= 2:
                fig, axes = plt.subplots(1, 5)
                for coef, ax in zip(X_random_pos.T, axes.ravel()):
                    ax.matshow(coef.reshape(28, 28), cmap=plt.cm.gray)
                    ax.set_xticks(())
                    ax.set_yticks(())

            print('-------')
            fig.savefig(fname='digit_' + str(cur_digit) + '_pos_class')

            if X_sv.shape[1] >= 2:
                fig, axes = plt.subplots(1, 5)
                for coef, ax in zip(X_random_neg.T, axes.ravel()):
                    ax.matshow(coef.reshape(28, 28), cmap=plt.cm.gray)
                    ax.set_xticks(())
                    ax.set_yticks(())

            print('-------')
            fig.savefig(fname='digit_' + str(cur_digit) + '_neg_class')

    if apply_svm:
        X_test_np = X_te.data.numpy()
        y_test_np = y_te.data.numpy()

        y_train_np = y_train.data.numpy()
        X_train_np = X_train.data.numpy()

        models = {}
        # load models
        for digit_model in range(10):
            with open(
                    r'/Users/albertorodriguez/Desktop/Current Courses/ML_Lab/4 Topic/problem_set4/stubs/finalized_model_svm_digit_' + str(
                            digit_model) + '.sav', 'rb') as f:
                models['model_' + str(digit_model)] = pickle.load(f)

        # Init results matrix Y_pred
        Y_pred_test = np.zeros((X_test_np.shape[0], 10))
        Y_pred_train = np.zeros((X_train_np.shape[0], 10))

        for digit, model_cur in enumerate(models):
            print(model_cur)
            m = models[model_cur]
            Y_pred_test[:, digit] = m.predict(X_test_np)
            Y_pred_train[:, digit] = m.predict(X_train_np)

        # Convert to pred numbers
        y_pred_test_numbers = np.argmax(Y_pred_test, axis=1)
        y_pred_train_numbers = np.argmax(Y_pred_train, axis=1)

        test_acc = np.argwhere(y_pred_test_numbers - y_test_np == 0) / X_test_np.shape[0] * 100
        train_acc = np.argwhere(y_pred_train_numbers - y_train_np == 0) / X_train_np.shape[0] * 100

        print('Accuracy on the Training data: {}'.format(train_acc))
        print('Accuracy on the Test data: {}'.format(test_acc))

if __name__ == '__main__':
    print('Main')
    #Assignment4()
    #Assignment5()
    #Assignment6()
