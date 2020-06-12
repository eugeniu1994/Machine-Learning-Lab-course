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