""" ps3_implementation.py

PUT YOUR NAME HERE:
<FIRST_NAME><LAST_NAME>


Write the functions
- cv
- zero_one_loss
- krr
Write your implementations in the given functions stubs!


(c) Daniel Bartz, TU Berlin, 2013
"""
import numpy as np
import scipy.linalg as la
import itertools as it
import time
import pylab as pl
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import cdist
from matplotlib import pyplot as plt


def zero_one_loss(y_true, y_pred):
    ''' your header here!
    '''


def mean_absolute_error(y_true, y_pred):
    ''' Compute the mean absolute error for a given predicted y and true y values

    Input:   - y_true : np arrayy, n dim vector containing the true values for a classification or regression task
             - y_pred : np array, n dim vector containing the model's predicted values
    '''
    n = len(y_true)
    mae = (1/n)* np.sum(np.abs(y_true - y_pred))
    return mae


def cv(X, y, method, params, loss_function=zero_one_loss, nfolds=10, nrepetitions=5):
    ''' Perform Cross Validation
    '''
    # Data info
    n, d = X.shape
    indexes = np.arange(0, n)
    folds_elements_rounded = int(n/nfolds) # elements of the  1 to (k-1) folds except the last fold

    # Create list with all possible params combinations
    combinations_params = list(it.product(*(params[Name] for Name in list(params))))
    n_combinations = len(combinations_params)

    # Init error function containing the average loss for every parameters combination
    error_combinations = np.zeros(n_combinations)

    # For loop over all possible params combinations

    for cur_index, local_parameter in enumerate(combinations_params):
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
                # extract Training data
                if cur_fold == (nfolds - 1):
                    test_fold_indexes = indexes_shuffled[(cur_fold * folds_elements_rounded):]

                    y_fold_test = y[test_fold_indexes]
                    X_fold_test = X[test_fold_indexes, :]

                    train_fold_indexes = np.delete(indexes_shuffled, test_fold_indexes)
                    X_fold_train = X[train_fold_indexes, :]
                    y_fold_train = y[train_fold_indexes]
                else:
                    test_fold_indexes = indexes_shuffled[(cur_fold * folds_elements_rounded):(
                            cur_fold * folds_elements_rounded + folds_elements_rounded)]
                    train_fold_indexes = np.delete(indexes_shuffled, test_fold_indexes)

                    y_fold_test = y[test_fold_indexes]
                    X_fold_test = X[test_fold_indexes, :]

                    X_fold_train = X[train_fold_indexes, :]
                    y_fold_train = y[train_fold_indexes]

                # Train Model on the fold training data and test it
                method_fold.fit(X_fold_train, y_fold_train)
                y_pred = method_fold.predict(X_fold_test)

                # Sum Error
                e_cv_error += loss_function(y_pred, y_fold_test)

            e_r_error += e_cv_error / nfolds

        final_error = e_r_error / nrepetitions
        error_combinations[cur_index] = final_error

    # Look for minimum error and which params combination it corresponds to
    best_param_combination = combinations_params[int(np.argwhere(error_combinations == np.min(error_combinations)))]

    # Init model with the "best" parameters combination and save its correspondent computed cv-loss value
    best_method = method(*best_param_combination)
    best_method.cvloss = np.min(error_combinations)
    best_method.fit(X, y) # Train best model on all the dataset

    return best_method


class krr():
    ''' your header here!
    '''
    def __init__(self, kernel='linear', kernelparameter=1, regularization=0):
        self.kernel = kernel
        self.kernelparameter = kernelparameter
        self.regularization = regularization

    def fit(self, X, y, kernel=False, kernelparameter=False, regularization=False):
        ''' your header here!
        '''
        n, d = X.shape

        if kernel is not False:
            self.kernel = kernel
        if kernelparameter is not False:
            self.kernelparameter = kernelparameter
        if regularization is not False:
            self.regularization = regularization

        # Compute K matrix
        if self.kernel == 'linear':
            K = X @ X.T

        elif self.kernel == 'polynomial':
            K = (X @ X.T + 1)**self.kernelparameter

        elif self.kernel == 'gaussian':
            K = cdist(X, X, 'euclidean')
            K = np.exp(-(K ** 2) / (2. * self.kernelparameter ** 2))

        if self.regularization == 0:
            w, U = la.eigh(K)
            #U = np.real(U)
            U_T = U.T
            L = np.diag(np.real(w))          # Matrix with eigenvalue on its diagonal
            mean_eig_value = np.mean(np.real(w))
            candidates = np.logspace(np.log10(np.max([np.min(w), 1e-5])), np.log10(np.max(w)))

            error_cv = np.zeros(len(candidates))
            for index, c in enumerate(candidates):
                E = np.diag(1/(w + c * np.ones(n)))
                S = U @ L @ E @ U_T
                Sy = S @ y
                S_diag = np.diag(S)
                error_cv[index] = np.sum( (1/n) * ((y - Sy)/(1 - S_diag)))

            # Choose C with minim error
            #print()
            self.regularization = float(candidates[np.argwhere(error_cv == np.min(error_cv))])
            self.alpha = la.inv((K + self.regularization*np.eye(n))) @ y
            self.alpha = self.alpha[:, np.newaxis]

        else:
            self.alpha = la.inv((K + self.regularization*np.eye(n))) @ y
            self.alpha = self.alpha[:, np.newaxis]

        self.X_train = X
        return self

    def predict(self, X):
        ''' your header here!
        '''
        # Calculate K matrix
        # Compute K matrix
        if self.kernel == 'linear':
            K = self.X_train @ X.T

        elif self.kernel == 'polynomial':
            K = (self.X_train @ X.T  + 1) ** self.kernelparameter

        elif self.kernel == 'gaussian':
            K = cdist(self.X_train, X, 'euclidean')
            K = np.exp(-(K ** 2) / (2. * self.kernelparameter ** 2))

        # Predictions
        y_pred = (self.alpha.T @ K).T
        return y_pred

def noisysincfunction(N, noise):
    ''' noisysincfunction - generate data from the "noisy sinc function"
        % usage
        %     [X, Y] = noisysincfunction(N, noise)
        %
        % input
        %     N: number of data points
        %     noise: standard variation of the noise
        %
        % output
        %     X: (1, N)-matrix uniformly sampled in -2pi, pi
        %     Y: (1, N)-matrix equal to sinc(X) + noise
        %
        % description
        %     Generates N points from the noisy sinc function
        %
        %        X ~ uniformly in [-2pi, pi]
        %        Y = sinc(X) + eps, eps ~ Normal(0, noise.^2)
        %
        % author
        %     Mikio Braun
    '''
    X = np.sort(2 * np.pi * np.random.rand(1, N) ) - np.pi
    Y = np.sinc(X) + noise * np.random.randn(1, N)
    return X.reshape(-1, 1), Y.flatten()

Xtr, Ytr = noisysincfunction(100, 0.1)
model = krr(kernel='gaussian', kernelparameter=0.5, regularization=0)
model.fit(X=Xtr, y=Ytr)
y_pred = model.predict(X=Xtr)

plt.plot(Xtr, Ytr)
plt.plot(Xtr, y_pred)
plt.show()