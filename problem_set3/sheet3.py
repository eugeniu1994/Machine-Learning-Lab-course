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
import pandas as pd

def zero_one_loss(y_true, y_pred):  # return number of misclassified labels
    n= len(y_true)
    return (1.0/n)*( n - np.sum(y_true == y_pred))

def mean_absolute_error(y_true, y_pred):
    return (1.0 / len(y_true)) * (np.abs(y_pred - y_true).sum())

def squared_error_loss(y_true, y_pred):
    assert (len(y_true) == len(y_pred))
    loss = np.mean((y_true - y_pred) ** 2)
    return loss

class krr():
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
        #X_train = cross_validation_split_dataset(X, folds=nfolds)
        #Y_train = cross_validation_split_dataset(y, folds=nfolds)
        #print('X_train {}, Y_train:{} '.format(np.shape(X_train), np.shape(Y_train)))

        for t in theta:
            current_time = time.time()
            # print('========{}========='.format(t))
            scores = []
            X_train = cross_validation_split_dataset(X, folds=nfolds)
            Y_train = cross_validation_split_dataset(y, folds=nfolds)
            #print('X_train {}, Y_train:{} '.format(np.shape(X_train), np.shape(Y_train)))

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
                scores.append(loss)

            average_loss = np.average(scores)
            average_loss = np.mean(scores)
            #print('Average loss {}'.format(average_loss))
            if average_loss < best_error:
                best_error = average_loss
                best_kernel = t[0]
                best_kernelparam = t[1]
                best_reg = t[2]
            last_t = time.time() - current_time

            details = "Average loss: {}, kernel:{},kernelparam:{},regularizer:{}".format(average_loss,t[0],t[1],t[2])
            updateProgress(runs, progress + 1,details,get_remaining_time(progress + 1, runs, last_t))
            progress += 1

    method.fit(X=X, y=y, kernel=best_kernel, kernelparameter=best_kernelparam, regularization=best_reg)
    method.cvloss = best_error
    print("Best training loss: {}, kernel:{},kernelparam:{}, regularizer:{}".format(best_error,best_kernel,best_kernelparam,best_reg))
    return method

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
        cvkrr = imp.cv(X_cv, y_cv, imp.krr, params, loss_function=imp.mean_absolute_error,
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
            model = imp.krr(kernel='gaussian', kernelparameter=width, regularization=C)
            model.fit(X=X_train[:n_cur, :], y= y_train[:n_cur, :])
            # Test model on the test set
            y_pred = model.predict(X=X_test)
            errors[index] = imp.mean_absolute_error(y_test, y_pred)

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

    dataset_names = ['banana','diabetis','flare-solar','image','ringnorm']
    k=0
    results = {}
    for name in dataset_names:
        Xtr, Xtest, Ytr, Ytest = getDataset(name)
        print('Dataset {}, Xtr:{}, Ytr:{},    Xtest:{}, Ytest:{}'.format(name, np.shape(Xtr), np.shape(Ytr), np.shape(Xtest), np.shape(Ytest)))
        params = {'kernel': ['gaussian'], 'kernelparameter': np.logspace(-2, 2, 10),'regularization': [0]}
        #params = {'kernel': ['gaussian'], 'kernelparameter': [1.0], 'regularization': [0]}
        cvkrr = cv(Xtr, Ytr, krr, params, loss_function=squared_error_loss, nrepetitions=2)
        y_pred = cvkrr.predict(Xtest)
        print('Testing loss {}'.format(squared_error_loss(Ytest,y_pred)))
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

        print(str(name) + ' done ')
        print()

    with open('results.p', 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    print('Main')
    Assignment4()
