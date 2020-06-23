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
from scipy.spatial.distance import cdist



def zero_one_loss(y_true, y_pred):
    ''' your header here!
    '''


def mean_absolute_error(y_true, y_pred):
    ''' Compute the mean absolute error for a given predicted y and true y values

    Input:   - y_true : np arrayy, n dim vector containing the true values for a classification or regression task
             - y_pred : np array, n dim vector containing the model's predicted values
    '''
    n = len(y_true)
    mae = (1/n) * np.sum(np.abs(y_true - y_pred))
    return mae


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
        self.X_train = X.copy()
        n, d = X.shape
        if kernel is not False:
            self.kernel = kernel
        if kernelparameter is not False:
            self.kernelparameter = kernelparameter
        if regularization is not False:
            self.regularization = regularization

        else:
            pass

        # Compute K matrix
        if self.kernel == 'linear':
            K = X @ X.T

        elif self.kernel == 'polynomial':
            K = (X @ X.T + 1)**self.kernelparameter

        elif self.kernel == 'gaussian':
            K = cdist(X, X, 'euclidean')
            K = np.exp(-(K ** 2) / (2. * self.kernelparameter ** 2))

        else:
            pass

        if self.regularization == 0:
            w, U = la.eigh(K)
            U_T = U.T
            L = np.diag(w)          # Matrix with eigenvalue on its diagonal
            #mean_eig_value = np.mean(np.real(w))
            candidates = np.logspace(np.log10(np.max([np.min(w), 1e-5])), np.log10(np.max(w)))
            #candidates = np.linspace(0, mean_eig_value, 1000)
            error_cv = np.zeros(len(candidates))
            for index, c in enumerate(candidates):
                E = np.diag(1/(w + c * np.ones(n)))
                S = U @ L @ E @ U_T
                Sy = S @ y
                S_diag = np.diag(S)
                error_cv[index] = (1/n) * np.sum(((y - Sy)/(1 - S_diag))**2)

            # Choose C with minim error
            self.regularization = float(candidates[np.argmin(error_cv)])
            self.alpha = la.inv((K + self.regularization*np.eye(n))) @ y
            self.alpha = np.squeeze(self.alpha)

        else:
            self.alpha = (la.inv(K + self.regularization*np.eye(n)) @ y)


        return self

    def predict(self, X):
        ''' your header here!
        '''
        # Calculate K matrix
        # Compute K matrix
        if self.kernel == 'linear':
            K = self.X_train @ X.T

        elif self.kernel == 'polynomial':
            K = (self.X_train @ X.T + 1) ** self.kernelparameter

        elif self.kernel == 'gaussian':
            K = cdist(self.X_train, X, 'euclidean')
            K = np.exp(-(K ** 2) / (2. * self.kernelparameter ** 2))

        # Predictions
        y_pred = (self.alpha @ K).T
        return y_pred

def cv(X, y, method, params, loss_function=mean_absolute_error, nfolds=10, nrepetitions=5, roc_f = False):
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
        if len_last_t > 25:
            last_times.pop(0)
        mean_time = sum(last_times) // len_last_t
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

    # Init error function containing the average loss for every parameters combination
    error_combinations = np.zeros(n_combinations)
    # For loop over all possible params combinations
    runs = n_combinations * nrepetitions
    progress = 0
    method.auc = 0.0
    for cur_index, local_parameter in enumerate(combinations_params):
        # Compute mean r-CV-error for fixed parameters
        e_r_error = 0
        # Init method with current parameters combination
        method_fold = method(*local_parameter)  # local_parameter is a list containing the parameters values used.
        current_time = time.time()
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
                #e_cv_error += loss_function(y_pred, y_fold_test)
                loss = loss_function(Y_test, y_pred)
                e_cv_error += loss

            e_r_error += e_cv_error / nfolds

            last_t = time.time() - current_time

            details = "kernel:{},kernelparam:{},regularizer:{}".format(local_parameter[0], local_parameter[1], local_parameter[2])
            updateProgress(runs, progress + 1, details, get_remaining_time(progress + 1, runs, last_t))
            progress += 1


        final_error = e_r_error / nrepetitions

        if roc_f:
            tp, fp = final_error[0,], final_error[1, :]
            auc = np.abs(np.trapz(tp, fp))
            error_combinations[cur_index] = auc
            if method.auc < auc:
                method.tp = np.squeeze(tp)
                method.fp = np.squeeze(fp)
        else:
            error_combinations[cur_index] = final_error

    # Look for minimum error and which params combination it corresponds to
    best_param_combination = combinations_params[np.argmin(error_combinations)]

    # Init model with the "best" parameters combination and save its correspondent computed cv-loss value
    best_method = method(*best_param_combination)
    best_method.cvloss = np.min(error_combinations)
    best_method.fit(X, y)  # Train best model on all the dataset

    print('Best Model: {}'.format(best_param_combination))
    print('Best Model Train CV loss: {}'.format(best_method.cvloss))

    return best_method
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

def squared_error_loss(y_true, y_pred):

    ''' returns the squared error loss
    '''
    assert(len(y_true) == len(y_pred))
    loss = np.mean( (y_true - y_pred)**2 )
    return loss
