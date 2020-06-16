import numpy as np
import scipy.linalg as la
import itertools as it
import time
import pylab as pl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from random import randrange
import random
import pickle
from scipy.spatial.distance import cdist
<<<<<<< HEAD
from scipy.stats import norm
from scipy.spatial.distance import cdist
import pandas as pd
import scipy
=======
import pandas as pd
>>>>>>> cb8a8af9b42a1cb08668215adf0559a508cf9de5

def zero_one_loss(y_true, y_pred):  # return number of misclassified labels
    n= len(y_true)
    return (1.0/n)*( n - np.sum(y_true == y_pred))

def mean_absolute_error(y_true, y_pred):
    return (1.0 / len(y_true)) * (np.abs(y_pred - y_true).sum())

def squared_error_loss(y_true, y_pred):
    assert (len(y_true) == len(y_pred))
    loss = np.mean((y_true - y_pred) ** 2)
    return loss

class _krr():
    def __init__(self, kernel='linear', kernelparameter=1, regularization=0):
        self.kernel = kernel
        self.kernelparameter = kernelparameter
        self.regularization = regularization

    def Kernels(self, Xtr, Xts):
        n = np.shape(Xtr)[0]
        m = np.shape(Xts)[0]
        k = np.empty((n, m), dtype=np.float)
        if self.kernel == "linear":
            for i in range(n):
                for j in range(m):
                    k[i][j] = np.dot(Xtr[i], Xts[j])
        elif self.kernel == "polynomial":
            for i in range(n):
                for j in range(m):
                    k[i][j] = np.power(np.dot(Xtr[i], Xts[j]) + 1, self.kernelparameter)
        elif self.kernel == "gaussian":
            for i in range(n):
                for j in range(m):
                    k[i][j] = np.exp((-np.linalg.norm(Xtr[i] - Xts[j]) ** 2) / 2 * self.kernelparameter ** 2)
        else:
            print('Not implemented yet=======================================================================')
        return k

    def fit(self, X, y, kernel=False, kernelparameter=False, regularization=False):
        if kernel is not False:
            self.kernel = kernel
        if kernelparameter is not False:
            self.kernelparameter = kernelparameter
        if regularization is not False:
            self.regularization = regularization

        #print('fit X:{}, y:{}'.format(np.shape(X), np.shape(y)))
        n = np.shape(X)[0]
        self.data = X
        k = self.Kernels(X, X)
        if self.regularization == 0:
            #D, U = np.linalg.eig(k)  # eigh  faster,
            D, U = np.linalg.eigh(k)
            # logarithmically spaced candidates around the mean of the eigenvalues
            regularizer = np.logspace(np.log10(np.max([np.min(D), 1e-5])), np.log10(np.max(D)))
            Uy = np.dot(U.T, y)  # precompute for speed up
            error = np.inf
            # print('regularizer ', np.shape(regularizer))
            best_reg = 0
            for reg in regularizer:
                # print('regularizer ',reg)
                UD_reg = U * (D / (D + reg))
                S = np.sum((UD_reg) * U, axis=1)  # hat matrix diag

                if (np.sum(S == 1) > 0):
                    continue
                y_hat = np.dot(UD_reg, Uy)
                eps = (1 / n) * np.power(((y.T - y_hat) / (1 - S)), 2).sum()
                if (eps < error):
                    error = eps
                    best_reg = reg

            D=np.squeeze(D)
            Uy = np.squeeze(Uy)
            Uy = Uy / (D + best_reg)
            self.alpha = np.dot(U, Uy)
        else:
            J = k + self.regularization * np.eye(n)
            if np.linalg.matrix_rank(J) == J.shape[0]:  # check singularity
                self.alpha = np.dot(np.linalg.inv(J),y)
                #self.alpha = np.linalg.solve(J, y.T)
            else:
                print('Matrix is singular')
                J = J + (np.eye(np.shape(J)[0]) * 0.5)  # regularization
                self.alpha = np.dot(np.linalg.inv(J), y)
                #self.alpha = np.linalg.solve(J, y.T)

        return self

    def predict(self, X):
        y_hat = np.dot(self.Kernels(X, self.data), self.alpha).T
        return y_hat

def cv(X, y, method, params, loss_function=zero_one_loss, nfolds=10, nrepetitions=5):
    if loss_function is None:
        loss_function = mean_absolute_error
    method = method()

    def cross_validation_split_dataset(dataset, folds=5):
        dts_split = list()
        dts_copy = list(dataset)
        size_fold = int(len(dataset) / folds)
        for i in range(folds):
            fold = list()
            while len(fold) < size_fold:
                idx = randrange(len(dts_copy))
                fold.append(dts_copy.pop(idx))
            dts_split.append(fold)
        return np.array(dts_split)

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
        if len_last_t > 5:
            last_times.pop(0)
        mean_time = sum(last_times) // len_last_t
        remain_s_tot = mean_time * (total - step + 1)
        minutes = remain_s_tot // 60
        seconds = remain_s_tot % 60
        return "{}m {}s".format(minutes, seconds)

    # version from handbook
    print('X:{},  y:{}'.format(np.shape(X), np.shape(y)))
    n,d = np.shape(X)
    best_kernel, best_kernelparam, best_reg = False,False,False
    best_error=np.inf

    theta = [(g, f, s) for g in params['kernel'] for f in params['kernelparameter'] for s in params['regularization']]
    runs = len(theta)*nrepetitions
    progress = 0
    for i in range(nrepetitions):
        #print('Step ', i)
        for t in theta:
            current_time = time.time()
            # print('========{}========='.format(t))
            scores = []
            X_train = cross_validation_split_dataset(X, folds=nfolds)
            Y_train = cross_validation_split_dataset(y, folds=nfolds)
            #print('X_train {}, Y_train:{} '.format(np.shape(X_train), np.shape(Y_train)))
            error = 0
            indices = np.arange(X_train.shape[0])
            for j in range(nfolds):
                X_Pj = np.ravel(X_train.copy()[indices != j, :]) if n==1 else (X_train.copy()[indices != j, :]).reshape(-1,d)
                Y_Pj = np.ravel(Y_train.copy()[indices != j, :])

                X_test = X_train.copy()[j]
                Y_test = Y_train.copy()[j]
                #print('X_Pj {},  Y_Pj:{}'.format(np.shape(X_Pj), np.shape(Y_Pj)))
                #print('X_test {},  Y_test:{}'.format(np.shape(X_test), np.shape(Y_test)))

                method.fit(X=X_Pj, y=Y_Pj, kernel=t[0], kernelparameter=t[1], regularization=t[2])
                y_hat = method.predict(X_test)

                loss = loss_function(Y_test, y_hat)
                #print(loss)
                #scores.append(loss)
                error += loss

            #average_loss = np.mean(scores)
            average_loss = error / nfolds
            #print('Average loss {}'.format(average_loss))
            if len(np.shape(average_loss)) == 0:
                if average_loss < best_error:
                    best_error = average_loss
                    best_kernel = t[0]
                    best_kernelparam = t[1]
                    best_reg = t[2]
            else:
                if all(average_loss.flat < best_error):
                    best_error = average_loss.flat
                    print('best_error ', np.shape(best_error))
                    best_kernel = t[0]
                    best_kernelparam = t[1]
                    best_reg = t[2]

                #method.mean = np.mean(scores, axis=1)
            last_t = time.time() - current_time

            #details = "Average loss: {}, kernel:{},kernelparam:{},regularizer:{}".format(average_loss,t[0],t[1],t[2])
            details = "Average loss: {}, kernel:{},kernelparam:{},regularizer:{}".format('~', t[0], t[1], t[2])
            updateProgress(runs, progress + 1,details,get_remaining_time(progress + 1, runs, last_t))
            progress += 1

    print("Best training loss: {}, kernel:{},kernelparam:{}, regularizer:{}".format(best_error,best_kernel,best_kernelparam,best_reg))
    method.fit(X=X, y=y, kernel=best_kernel, kernelparameter=best_kernelparam, regularization=best_reg)
    method.cvloss = best_error
    return method

<<<<<<< HEAD
def _cv(X, y, method, params, loss_function=zero_one_loss, nfolds=10, nrepetitions=5):
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
    print('Best loss ', np.min(error_combinations))
    return best_method

class krr():
    ''' your header here!
    '''
    def __init__(self, kernel='linear', kernelparameter=1, regularization=0):
        self.kernel = kernel
        self.kernelparameter = kernelparameter
        self.regularization = regularization

    def fit(self, X, y, kernel=False, kernelparameter=False, regularization=False):
        n, d = X.shape

        if kernel is not False:
            self.kernel = kernel
        if kernelparameter is not False:
            self.kernelparameter = kernelparameter
        if regularization is not False:
            self.regularization = regularization

        try:
            # Compute K matrix
            if self.kernel == 'linear':
                K = X @ X.T
            elif self.kernel == 'polynomial':
                K = (X @ X.T + 1)**self.kernelparameter
            elif self.kernel == 'gaussian':
                K = cdist(X, X, 'euclidean')
                K = np.exp(-(K ** 2) / (2. * self.kernelparameter ** 2))

        except Exception as errorEx:
            print('Compute K matrix error ', str(errorEx))
            raise Exception("Sorry, "+str(errorEx))


        if self.regularization == 0:
            w, U = la.eigh(K)
            U_T = U.T
            L = np.diag(np.real(w))          # Matrix with eigenvalue on its diagonal
            #mean_eig_value = np.mean(np.real(w))
            candidates = np.logspace(np.log10(np.max([np.min(w), 1e-5])), np.log10(np.max(w)))
            #candidates = np.linspace(0, mean_eig_value, 1000)
            #print(candidates)
            error_cv = np.zeros(len(candidates))
            for index, c in enumerate(candidates):
                E = np.diag(1/(w + c * np.ones(n)))
                S = U @ L @ E @ U_T
                Sy = S @ y
                S_diag = np.diag(S)
                error_cv[index] = (1/n) * np.sum(((y - Sy)/(1 - S_diag))**2)

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

=======
>>>>>>> cb8a8af9b42a1cb08668215adf0559a508cf9de5
def Assignment3():
    plot_a = False  # plot ass (a) distances against energy differences
    train_cv = True
    optimal_train = False
    # Load dataset
    mat = scipy.io.loadmat('../data/qm7.mat')
    X_not = mat['X']
    y = mat['T'].T

    X = np.zeros((7165, 23))
    n, d = X.shape
    print('X shape: {} , y_shape: {}'.format(X.shape, y.shape))

    # Compute eigenvalues of each Coloumb matrix and set them as features
    for j in range(X_not.shape[0]):
        eigvalues, eivectors = np.linalg.eig(X_not[j, :, :])
        X[j, :] = eigvalues

    # -----------------------------------------------------------------------------------------------------------------
    # (a) Plot the distances ∥xi−xj∥ against the absolute difference of energies|yi−yj| for all pairs of data points.

    # -> Distances ∥xi−xj∥
    x_all_dist = cdist(X, X, 'euclidean')
    y_all_dist = cdist(y, y, 'euclidean')  # 1DIM euclidean equivalent to the absolute difference

    # Extract upper matrix without the main diagonal where dist=0
    x_dist = x_all_dist[np.triu_indices_from(x_all_dist, k=1)]
    y_dist = y_all_dist[np.triu_indices_from(y_all_dist, k=1)]

    if plot_a:
        # Pandas data frame, use e.g 4.000.000  datapoints intead of 25.000.000
        df = pd.DataFrame({"x_diff": x_dist, "y_diff": y_dist})

        dfSample = df.sample(10000000)  # This is the importante line
        xdataSample, ydataSample = dfSample["x_diff"], dfSample["y_diff"]

        # Plot distances
        plt.figure(num=' Distance against energy difference')
        plt.scatter(xdataSample, ydataSample, s=1)


    # -----------------------------------------------------------------------------------------------------------------
    # (b) Shuffle the data randomly and fix a train/test split of size 5000/2165.
    indexes = np.arange(0, n)
    indexes_shuffled = indexes.copy()
    np.random.shuffle(indexes_shuffled)

    X_train = X[indexes_shuffled[:5000], :]
    X_test = X[indexes_shuffled[5000:], :]

    y_train = y[indexes_shuffled[:5000], :]
    y_test = y[indexes_shuffled[5000:], :]

    print('X_train shape: {} , y train shape: {}'.format(X_train.shape, y_train.shape))
    print('X_test shape: {} , y test shape: {}'.format(X_test.shape, y_test.shape))

    # -----------------------------------------------------------------------------------------------------------------
    # (c) Use five fold cross-validation to estimate on 2500 random training samples:
    indexes_train = np.arange(0, 5000)
    np.random.shuffle(indexes_train)

    X_cv = X_train[indexes_train[:2500], :]
    y_cv = y_train[indexes_train[:2500], :]

    # Width parameter σ of the Gaussian kernel. Candidates are quantiles of pairwise Euclidean distances.
    gaussian_width = np.quantile(x_dist, np.linspace(0.01, 1, 20))  # 1%, 2%,...99% quantiles of pairwise eucl diatances
    print(gaussian_width)
    print(np.max(x_dist))
    # Regularization parameter C. Use logarithmically scaled values between 10−7 and 100 as candidates.
    regularization_c = np.logspace(-7, 0, 10)

    # Dictionary with parameter for the cv function
    params = {'kernel': ['gaussian'], 'kernelparameter': gaussian_width,
              'regularization': regularization_c}

    if train_cv:
<<<<<<< HEAD
        cvkrr = cv(X_cv, y_cv, krr, params, loss_function=mean_absolute_error,
=======
        cvkrr = imp.cv(X_cv, y_cv, imp.krr, params, loss_function=imp.mean_absolute_error,
>>>>>>> cb8a8af9b42a1cb08668215adf0559a508cf9de5
                   nrepetitions=10, nfolds=5)

    else:
        pass

    # -----------------------------------------------------------------------------------------------------------------
    # (d) Keep C and σ fixed and plot the MAE on the test set as a function of the number n of training samples with
    # n from 100 to 5000.

    if optimal_train:
        C = 0
        width = 25
        n_model_train = np.logspace(np.log10(100), np.log10(3000), 10).astype('int')
        errors = np.zeros(10)

        for index, n_cur in enumerate(n_model_train):
            # Train
            print(n_cur)
<<<<<<< HEAD
            model = krr(kernel='gaussian', kernelparameter=width, regularization=C)
            model.fit(X=X_train[:n_cur, :], y= y_train[:n_cur, :])
            # Test model on the test set
            y_pred = model.predict(X=X_test)
            errors[index] = mean_absolute_error(y_test, y_pred)
=======
            model = imp.krr(kernel='gaussian', kernelparameter=width, regularization=C)
            model.fit(X=X_train[:n_cur, :], y= y_train[:n_cur, :])
            # Test model on the test set
            y_pred = model.predict(X=X_test)
            errors[index] = imp.mean_absolute_error(y_test, y_pred)
>>>>>>> cb8a8af9b42a1cb08668215adf0559a508cf9de5

        # Plot


        plt.figure(num='error')
        plt.scatter(n_model_train, errors)
        plt.figure(num='difference')
        print('max y: {} , min y: {}'.format(np.max(y_test), np.min(y_test)))
        plt.scatter(np.arange(0, 2165), np.squeeze(y_pred.squeeze()/y_test.squeeze()))
    else:
        pass

    # -----------------------------------------------------------------------------------------------------------------
    # (e) show scatter plots of points (yi,y􏰀i) for 1000 data points

    # to be done

    #print()
    plt.show()

<<<<<<< HEAD
def roc_fun(y_true, y_hat, points = 500):
    Negative = np.array(y_true.flatten() == -1).sum() #negative class
    Positive = len(y_true) - Negative                 #positive class
    B = np.linspace(-2, 2, points)[np.newaxis, :]     #variation of bias term
    #print('Shape of y_true', np.shape(y_true))
    #print('shape of B ', np.shape(B))
    #prediction = (y_hat.T - B) < 0
    PRED = (y_hat - B) < 0
    #print('Prediction:{}'.format(np.shape(prediction)))
    #print('Negative:{}, Positive:{}'.format(Negative,Positive))
    #TPR = PRED[y_true.flatten() == -1, :].sum(axis=0) / Negative
    #FPR = PRED[y_true.flatten() == 1, :].sum(axis=0) / Positive

    TPR = PRED[y_true.flatten() == 1, :].sum(axis=0) / Positive
    FPR = PRED[y_true.flatten() == -1, :].sum(axis=0) / Negative

    result = np.array([TPR, FPR])
    return result
=======

>>>>>>> cb8a8af9b42a1cb08668215adf0559a508cf9de5

def Assignment4():
    def readDataSet(filename):
        data = np.array([i.strip().split() for i in open(filename).readlines()]).T
        return data.astype(np.float)

    def getDataset(name):
        _Xtrain = readDataSet('./data/U04_'+str(name)+'-xtrain.dat')
        _Xtest = readDataSet('./data/U04_'+str(name)+'-xtest.dat')
        _Ytrain = readDataSet('./data/U04_'+str(name)+'-ytrain.dat')
        _Ytest = readDataSet('./data/U04_'+str(name)+'-ytest.dat')

        return _Xtrain, _Xtest, _Ytrain, _Ytest

    dataset_names = ['banana'] # ['banana','diabetis','flare-solar','image','ringnorm']
    k=0
    results = {}
    for name in dataset_names:
        Xtr, Xtest, Ytr, Ytest = getDataset(name)
        print('Dataset {}, Xtr:{}, Ytr:{},    Xtest:{}, Ytest:{}'.format(name, np.shape(Xtr), np.shape(Ytr), np.shape(Xtest), np.shape(Ytest)))

        #LOOCV-------------------------------------------------------------------------
        params = {'kernel': ['gaussian'], 'kernelparameter': np.logspace(-2, 2, 15),
                  'regularization': [0]}
        params = {'kernel': ['gaussian'], 'kernelparameter': [1.0],
                  'regularization': [0]}
        cvkrr = cv(Xtr, Ytr, krr, params, loss_function=squared_error_loss, nrepetitions=3)
        y_pred = cvkrr.predict(Xtest)
        print('Testing loss {} (Proper CV)'.format(squared_error_loss(Ytest,y_pred)))
        if k==0:
            plt.figure()
            plt.subplot(2, 2, 1)
            plt.scatter(Xtr[:, 0], Xtr[:, 1], c=Ytr)
            plt.title('Banana training data set')

            plt.subplot(2, 2, 2)
            plt.scatter(Xtest[:, 0], Xtest[:, 1], c=Ytest)
            plt.title('Banana testing data set')

            plt.subplot(2, 2, 3)
            plt.scatter(Xtest[:, 0], Xtest[:, 1], c=y_pred)
            plt.title('Banana CV own implementation')

            plt.subplot(2, 2, 4)
            from sklearn.kernel_ridge import KernelRidge
            clf = KernelRidge(kernel='rbf', alpha=1.0)
            clf.fit(Xtr, Ytr)
            preds = clf.predict(Xtest)
            print('sklearn loss is {}'.format(squared_error_loss(Ytest,preds)))
            pl.scatter(Xtest[:, 0], Xtest[:, 1], c=preds)
            pl.title('banana with sklearn ')

            plt.show()
        k+=1

        MyDict = dict()
        MyDict['cvloss'] = cvkrr.cvloss
        MyDict['kernel'] = cvkrr.kernel
        MyDict['kernelparameter'] = cvkrr.kernelparameter
        MyDict['regularization'] = cvkrr.regularization
        MyDict['y_pred'] = y_pred
        results[name] = MyDict

        #Proper CV---------------------------------------------------------------------
        #to do...
        params = {'kernel': [cvkrr.kernel], 'kernelparameter': [cvkrr.kernelparameter],
                  'regularization': [cvkrr.regularization]}
        cvkrr = cv(Xtr, Ytr, krr, params, loss_function=roc_fun, nrepetitions=3)
        TPR, FPR = cvkrr.cvloss[:int(len(cvkrr.cvloss)/2)],cvkrr.cvloss[int(len(cvkrr.cvloss)/2):]

        print('TPR:{},  FPR:{}'.format(np.shape(TPR), np.shape(FPR)))

        AUC = np.abs(np.trapz(TPR, FPR))
        print('AUC ', AUC)

        plt.plot(FPR, TPR)
        plt.ylabel('TPR')
        plt.xlabel('FPR')
        plt.title(' Dataset:{}, AUC:{} '.format(name, round(AUC,3)))
        plt.show()

    with open('results.p', 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    print('Main')
    Assignment4()
