import numpy as np
import scipy.linalg as la
import itertools as it
import time
import pylab as pl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from random import randrange
import random

def zero_one_loss(y_true, y_pred):  # return number of misclassified labels
    n= len(y_true)
    return (1.0/n)*(len(y_true) - np.sum(y_true == y_pred))

def mean_absolute_error(y_true, y_pred):
    return (1.0 / len(y_true)) * (np.abs(y_pred - y_true).sum())

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

        n = np.shape(X)[0]
        self.data = X
        k = self.Kernels(X, X)
        if self.regularization == 0:
            print('LOOCV----------------------')
            D, U = np.linalg.eigh(k)  # eigh handle complex number warning
            # logarithmically spaced candidates around the mean of the eigenvalues
            regularizer = np.logspace(np.log10(np.max([np.min(D), 1e-5])), np.log10(np.max(D)))
            Uy = np.dot(U.T, y.T)  # precompute for speed up
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

            self.J = np.dot(U, Uy / (D + best_reg))
        else:
            alfa_hat = k + self.regularization * np.eye(n)

            if np.linalg.matrix_rank(alfa_hat) == alfa_hat.shape[0]:  # check singularity
                self.J = np.dot(np.linalg.inv(alfa_hat),y)
                #self.J = np.linalg.solve(alfa_hat, y.T)
            else:
                alfa_hat = alfa_hat + (np.eye(np.shape(alfa_hat)[0]) * 0.5)  # regularization
                self.J = np.dot(np.linalg.inv(alfa_hat), y)
                # self.J = np.linalg.solve(alfa_hat, y.T)

        return self

    def predict(self, X):
        k = self.Kernels(X, self.data)
        y_hat = np.dot(k, self.J).T
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
        return np.squeeze(np.array(dts_split))

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
    best_kernel, best_kernelparam, best_reg = False,False,False
    best_error=np.inf

    theta = [(f, s) for f in params['kernelparameter'] for s in params['regularization']]
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

            indices = np.arange(X_train.shape[0])
            for j in range(nfolds):
                X_Pj = np.ravel(X_train.copy()[indices != j, :])
                Y_Pj = np.ravel(Y_train.copy()[indices != j, :])

                X_test = X_train.copy()[j]
                Y_test = Y_train.copy()[j]
                #print('X_Pj {},  Y_Pj:{}'.format(np.shape(X_Pj), np.shape(Y_Pj)))
                #print('X_test {},  Y_test:{}'.format(np.shape(X_test), np.shape(Y_test)))

                method.fit(X=X_Pj, y=Y_Pj, kernel=params['kernel'][0], kernelparameter=t[0], regularization=t[1])
                y_hat = method.predict(X_test)
                loss = loss_function(Y_test, y_hat)
                #print(loss)
                scores.append(loss)

            average_loss = np.average(scores)
            #print('Average loss {}'.format(average_loss))
            if average_loss < best_error:
                best_error = average_loss
                best_kernel = params['kernel'][0]
                best_kernelparam = t[0]
                best_reg = t[1]
            last_t = time.time() - current_time

            details = "Average loss: {}, kernel:{},kernelparam:{},regularizer:{}".format(average_loss,params['kernel'][0],t[0],t[1])
            updateProgress(runs, progress + 1,details,get_remaining_time(progress + 1, runs, last_t))
            progress += 1

    method.fit(X=X, y=y, kernel=best_kernel, kernelparameter=best_kernelparam, regularization=best_reg)
    method.cvloss = best_error

    return method

if __name__ == '__main__':
    print('Main')
