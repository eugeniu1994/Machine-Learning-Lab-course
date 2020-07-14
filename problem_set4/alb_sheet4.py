mport scipy.linalg as la
import matplotlib.pyplot as plt
import sklearn.svm
from cvxopt.solvers import qp
from cvxopt import matrix as cvxmatrix
import numpy as np
import torch
from scipy.spatial.distance import cdist
import cvxopt
from torch.nn import Module, Parameter, ParameterList
from torch.optim import SGD
import time
import itertools as it

class svm_qp():
    """ Support Vector Machines via Quadratic Programming """

    def __init__(self, kernel='linear', kernelparameter=1., C=1.):
        self.kernel = kernel
        self.kernelparameter = kernelparameter
        self.C = C
        self.alpha_sv = None
        self.b = None
        self.X_sv = None
        self.Y_sv = None
    
    def fit(self, X, Y):
        '''
         Input:        X     - nparray, dxn matrix
                       Y     - nparray, nx1 matrix with labels
        '''
        # INSERT_CODE
        X = X.T

        print('X shape:  {}'.format(X.shape))
        # read dimensions
        d, n = X.shape
        Y = Y.reshape((-1, 1))
        # Here you have to set the matrices as in the general QP problem
        q = -np.ones(n).reshape((-1, 1))

        # Compute Kernel matrix K
        if self.kernel == 'linear':
            K = X.T @ X
        elif self.kernel == 'polynomial':
            K = (X.T @ X + 1)**self.kernelparameter
        elif self.kernel == 'gaussian':
            K = cdist(X.T, X.T, 'euclidean')
            K = np.exp(-(K ** 2) / (2. * self.kernelparameter ** 2))
        else:
            pass

        P = (Y @ Y.T) * K
        q = -1 * np.ones(n).reshape((-1, 1))
        G = np.vstack((np.eye(n), -1*np.eye(n)))
        h = np.hstack((self.C*np.ones(n), np.zeros(n)))
        A = Y.T
        b = 0.0
        #print()
        #P = 
        #q = 
        #G = 
        #h = 
        #A =   # hint: this has to be a row vector
        #b =   # hint: this has to be a scalar
        
        # this is already implemented so you don't have to
        # read throught the cvxopt manual
        alpha = np.array(qp(cvxmatrix(P, tc='d'),
                            cvxmatrix(q, tc='d'),
                            cvxmatrix(G, tc='d'),
                            cvxmatrix(h, tc='d'),
                            cvxmatrix(A, tc='d'),
                            cvxmatrix(b, tc='d'))['x']).flatten()

        sv_threshold = 1e-5

        indexes_sv = np.argwhere((alpha > sv_threshold)).squeeze()


        self.alpha_sv_mask = alpha > 0
        self.alpha_sv = alpha[indexes_sv]
        self.X_sv = X[:, indexes_sv] # dxn array
        self.Y_sv = np.squeeze(Y)[indexes_sv] # n, array

        self.b = Y[0,0] - np.sum(self.alpha_sv * self.Y_sv * np.squeeze(K[0, indexes_sv]))

    def predict(self, X):
        '''
        Input:     X     - np_array, dxn array
        '''
        X = X.T
        # Compute Kernel matrix K
        if self.kernel == 'linear':
            K = self.X_sv.T @ X
        elif self.kernel == 'polynomial':
            K = (self.X_sv.T  @ X + 1)**self.kernelparameter
        elif self.kernel == 'gaussian':
            K = cdist(self.X_sv.T , X.T, 'euclidean')
            K = np.exp(-(K ** 2) / (2. * self.kernelparameter ** 2))
        else:
            pass

        y_pred = np.sum(K * (self.alpha_sv * self.Y_sv).reshape((-1, 1)), axis=0) + self.b
        return y_pred


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

'''
def plot_boundary_2d(X, y, model):
    # Classes
    cl_bin = np.unique(y)
    # INSERT CODE
    fig, ax = plt.subplots(1, 1)

    ax.scatter(X[y == cl_bin[0], 0], X[y == cl_bin[0], 1], c='b', label =str(cl_bin[0]) + 'Class')
    ax.scatter(X[y == cl_bin[1], 0], X[y == cl_bin[1], 1], c='r', label=str(cl_bin[1]) + ' Class')

    # If method is svm object
    if isinstance(model, svm_qp):
        X_sv = model.X_sv.T
        ax.scatter(X_sv[:, 0], X_sv[:, 1], marker='X', c='orange', label='Support Vectors')

    # Plot Hyperplane
    n_grid = 200
    x_grid = np.linspace(np.min(X[:, 0])- 4, np.max(X[:, 0])+4, n_grid)
    y_grid = np.linspace(np.min(X[:, 1])-4, np.max(X[:, 1])+4, n_grid)

    X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
    X_hyperplane = np.zeros((np.size(X_grid), 2))
    X_hyperplane[:, 0] = X_grid.flatten()
    X_hyperplane[:, 1] = Y_grid.flatten()


    # fit
    y_grid_labels = model.predict(X_hyperplane)

    if isinstance(model, neural_network):
        y_grid_labels = (-y_grid_labels).argmax(1)

    cp = ax.contourf(X_grid, Y_grid, y_grid_labels.reshape([n_grid, n_grid]), alpha=0.4)
    fig.colorbar(cp)  # Add a colorbar to a plot
    ax.set_title('Filled Contours Plot')

    plt.legend()
    plt.show()
'''


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
        plt.scatter(model.X_sv[:, 0], model.X_sv[:, 1], c='r', s=40, marker='x', edgecolors='k')

    plt.title(title)
    plt.show()

def sqdistmat(X, Y=False):
    if Y is False:
        X2 = sum(X**2, 0)[np.newaxis, :]
        D2 = X2 + X2.T - 2*np.dot(X.T, X)
    else:
        X2 = sum(X**2, 0)[:, np.newaxis]
        Y2 = sum(Y**2, 0)[np.newaxis, :]
        D2 = X2 + Y2 - 2*np.dot(X.T, Y)
    return D2

def buildKernel(X, Y=False, kernel='linear', kernelparameter=0):
    d, n = X.shape
    if Y.isinstance(bool) and Y is False:
        Y = X
    if kernel == 'linear':
        K = np.dot(X.T, Y)
    elif kernel == 'polynomial':
        K = np.dot(X.T, Y) + 1
        K = K**kernelparameter
    elif kernel == 'gaussian':
        K = sqdistmat(X, Y)
        K = np.exp(K / (-2 * kernelparameter**2))
    else:
        raise Exception('unspecified kernel')
    return K

class neural_network(Module):
    def __init__(self, layers=[2, 100, 2], scale=.1, p=None, lr=None, lam=None):
        super().__init__()
        self.weights = ParameterList([Parameter(scale*torch.randn(m, n, requires_grad=True)) for m, n in zip(layers[:-1], layers[1:])])
        self.biases = ParameterList([Parameter(scale*torch.randn(n, requires_grad=True)) for n in layers[1:]])

        self.p = p
        self.lr = lr if lr is not None else 1e-3
        self.lam = lam
        self.train = False

    def relu(self, X, W, b):
        # YOUR CODE HERE!
        a = X @ W + b
        # ReLu function =0 all where a<0, else z=a
        a[a < 0] = 0
        z = a
        if self.train:
            if self.p is not None:
                drop_out = torch.from_numpy(np.random.binomial(size=z.shape[1], n=1, p= (1-self.p)))
                z = drop_out * z
            else:
                pass
        else:
            pass

        return z

    def softmax(self, X, W, b):
        z = X @ W + b
        norm_factor = torch.sum(torch.exp(z), dim=1)
        y =(torch.exp(z.T)/norm_factor).T
        return y

    def forward(self, X):
        X = torch.tensor(X, dtype=torch.float)
        # YOUR CODE HERE!
        # First Hidden layer output A_1 and Z_1

        Z = self.relu(X, self.weights[0], self.biases[0])
        # The rest of the Hidden Layers
        total_layers = len(self.weights) + 1
        hidden_layers = total_layers - 2

        for layer in range(1, hidden_layers):
            w_cur = self.weights[layer]
            Z = self.relu(Z, w_cur, self.biases[layer])

        # Output layer
        y = self.softmax(Z, self.weights[-1], self.biases[-1])
        return y

    def predict(self, X):
        return self.forward(X).detach().numpy()

    def loss(self, ypred, ytrue):
        #Y_true = torch.zeros((ypred.shape[0], ypred.shape[1]))
        #Y_true[torch.arange(ypred.shape[0]), ytrue] = 1
        # YOUR CODE HERE!
        weights_sum = 0
        for weights in self.weights:
            W = weights.data
            weights_sum += torch.sum(W)**2
        n = ytrue.shape[0]
        loss = (-1/n)*torch.sum(ytrue * torch.log(ypred)) + self.lam * weights_sum
        return loss

    def fit(self, X, y, nsteps=857, bs=168, plot=False):
        X, y = torch.tensor(X), torch.tensor(y)
        optimizer = SGD(self.parameters(), lr=self.lr, weight_decay=self.lam)

        #bs = X.shape[0]
        I = torch.randperm(X.shape[0])
        n = int(np.ceil(.1 * X.shape[0]))
        Xtrain, ytrain = X[I[:n]], y[I[:n]]
        Xval, yval = X[I[n:]], y[I[n:]]

        Ltrain, Lval, Aval = [], [], []
        for i in range(nsteps):
            #print('Total steps: {}, Current step: {}'.format(nsteps, i))
            #print('Train Loss: {}'.format(Ltrain))
            # zero gradients
            optimizer.zero_grad()

            I = torch.randperm(Xtrain.shape[0])[:bs]
            self.train = True
            # Compute loss value by making a forward pass
            output = self.loss(self.forward(Xtrain[I]), ytrain[I])
            self.train = False

            # Add loss value to the Loss train list to keep track of it
            Ltrain += [output.item()]

            # gradient = backward pass
            output.backward()

            # Update weights
            # optimizer.step()
            optimizer.step()


            # Compute prediction for the validation ser X_val by making a forward pass
            # Compute loss of the validation
            outval = self.forward(Xval) # output val is a numpy array, no torch tensor-> see forward method
            Lval += [self.loss(outval, yval).item()]
            Aval += [np.array(outval.argmax(-1) == yval.argmax(-1)).mean()]

            #print('Current Train Loss: {}'.format(Ltrain[-1]))

        if plot:
            plt.plot(range(nsteps), Ltrain, label='Training loss')
            plt.plot(range(nsteps), Lval, label='Validation loss')
            plt.plot(range(nsteps), Aval, label='Validation acc')
            plt.legend()
            plt.show()

def zero_one_loss(y_pred, y_true):
    '''  Compute zero one loss function value for binary classification
    Definition:  apply_pca(X, m)
    Input:       y_pred                  - Nx1 numpy array, predicted values
                 y_true                  - Nx1 numpy arrays , true classification values
    Output:      loss                    - float, zero one loss function
    '''
    y_pred_sign = np.sign(y_pred)
    y_pred_sign[y_pred_sign == 0] = -1
    loss = (1/len(y_pred))*len(np.argwhere((y_true * y_pred_sign < 0)))
    return loss

# Cross Validation from Assignment 3
def cv(X, y, method, params, loss_function=zero_one_loss, nfolds=10, nrepetitions=5, roc_f=False, verbose = True):
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

    # Data info
    n, d = X.shape
    indexes = np.arange(0, n)
    folds_elements_rounded = int(n/nfolds) # elements of the  1 to (k-1) folds except the last fold

    # Create list with all possible params combinations
    combinations_params = list(it.product(*(params[Name] for Name in list(params))))
    n_combinations = len(combinations_params)
    print('Total combinations: {}'.format(n_combinations))
    print('Total Models no evaluate: {}'.format(n_combinations*nrepetitions*nfolds))
    model_number = 0
    # Init error function containing the average loss for every parameters combination
    error_combinations = np.zeros(n_combinations)

    # For loop over all possible params combinations
    runs = n_combinations * nrepetitions
    progress = 0
    for cur_index, local_parameter in enumerate(combinations_params):
        current_time = time.time()
        # Compute mean r-CV-error for fixed parameters
        e_r_error = 0
        # Init method with current parameters combination
        method_fold = method(*local_parameter)  # local_parameter is a list containing the parameters values used.
        for i in range(nrepetitions):
            # Randomly split the data set ny shuffling the indexes
            indexes_shuffled = indexes.copy()
            np.random.shuffle(indexes_shuffled)
            e_cv_error = 0
            # Go over all folds
            for cur_fold in range(nfolds):
                print('Model Number:{}'.format(model_number))
                model_number += 1
                # extract Training data
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
                y_pred = method_fold.predict(X_test)

                # Sum Error
                e_cv_error += loss_function(Y_test, y_pred)

            e_r_error += (e_cv_error/nfolds)
            last_t = time.time() - current_time

            if verbose:
                details = "kernel:{},kernelparam:{},regularizer:{}".format(local_parameter[0], local_parameter[1], local_parameter[2])
                updateProgress(runs, progress + 1, details, get_remaining_time(progress + 1, runs, last_t))
                progress += 1

        final_error = e_r_error / nrepetitions
        if roc_f==False:
            error_combinations[cur_index] = final_error

    # Look for minimum error and which params combination it corresponds to
    if roc_f==False:
        best_param_combination = combinations_params[np.argmin(error_combinations)]
        #best_param_combination = combinations_params[np.argmax(error_combinations)]

        # Init model with the "best" parameters combination and save its correspondent compu   ted cv-loss value
        best_method = method(*best_param_combination)
        best_method.cvloss = np.min(error_combinations)
        best_method.fit(X, y) # Train best model on all the dataset

        print('best method:', best_param_combination)
        print('best method loss: {}'.format(best_method.cvloss))
    else:
        best_method = method(*combinations_params) #combinations_params are already the best
        best_method.tp = final_error[0]
        best_method.fp = final_error[1]

    return best_method

def zero_one_loss_multiple(y_pred, y_true):
    ''' Computes zero one loss for a multi classification task : add 0 for correct classification
        and 1 for an incorrect classification.

        Input:    y_pred      - torch array nxK with n predictions for k-possible classifications
                  y_true      - torch array nxK with correct classifcations as 1

       Output:    loss        - float, zero one loss
    '''
    n = y_true.shape[0]
    k = y_true.shape[1]
    Z = torch.zeros((n, k))
    Z[torch.arange(n), torch.argmax(y_pred, 1)] = 1

    loss = (n - float(torch.sum(Z*y_true)))/n

    return loss
