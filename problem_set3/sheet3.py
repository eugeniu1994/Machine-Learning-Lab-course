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
import scipy
import pandas as pd
import pickle

def zero_one_loss(y_true, y_pred):  # return number of misclassified labels
    n= len(y_true)
    s = np.sum(y_true == np.sign(y_pred))
    return (1.0/n)*( n - s)

def mean_absolute_error(y_true, y_pred):
    return (1.0 / len(y_true)) * (np.abs(y_pred - y_true).sum())

def squared_error_loss(y_true, y_pred):
    assert (len(y_true) == len(y_pred))
    loss = np.mean((y_true - y_pred) ** 2)
    return loss

def cv(X, y, method, params, loss_function=mean_absolute_error, nfolds=10, nrepetitions=5, roc_f=False):
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
        if len_last_t > 70:
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
    print('Total combinations: {}'.format(n_combinations))
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
            #print('repetition number:{}'.format(i))
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
                e_cv_error += loss_function(Y_test, y_pred)

            e_r_error += (e_cv_error/nfolds)
            last_t = time.time() - current_time

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

class krr():
    def __init__(self, kernel='linear', kernelparameter=1, regularization=0):
        self.kernel = kernel
        self.kernelparameter = kernelparameter
        self.regularization = regularization

    def fit(self, X, y, kernel=False, kernelparameter=False, regularization=False):
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
            UL = U @ L
            for index, c in enumerate(candidates):
                E = np.diag(1/(w + c * np.ones(n)))
                S = UL @ E @ U_T
                Sy = S @ y
                S_diag = np.diag(S)
                error_cv[index] = (1/n) * np.sum(((y - Sy)/(1 - S_diag))**2)

            # Choose C with minim error
            self.regularization = float(candidates[np.argmin(error_cv)])

            #self.alpha = la.inv((K + self.regularization*np.eye(n))) @ y
            #self.alpha = np.squeeze(self.alpha)
            self.alpha = np.linalg.solve((K + self.regularization * np.eye(n)), y).T
        else:
            #self.alpha = np.dot(la.inv(K + self.regularization*np.eye(n)), y).squeeze()
            self.alpha = np.linalg.solve((K + self.regularization * np.eye(n)), y).T

        return self

    def predict(self, X):
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
        cvkrr = cv(X_cv, y_cv, krr, params, loss_function=mean_absolute_error,
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
            model = krr(kernel='gaussian', kernelparameter=width, regularization=C)
            model.fit(X=X_train[:n_cur, :], y= y_train[:n_cur, :])
            # Test model on the test set
            y_pred = model.predict(X=X_test)
            errors[index] = mean_absolute_error(y_test, y_pred)

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

def roc_fun(y_true, y_hat, points = 1500, threshold = 0.0):
    y_hat = y_hat[:, np.newaxis]
    Negative = np.array(y_true.flatten() == -1).sum() #negative class
    Positive = len(y_true) - Negative                 #positive class
    assert Positive>0 and Negative>0, 'Negative or Positive is zero, zero division exception (No positive or negative classes, Inbalanced data)'

    B = np.linspace(-3, 3, points)[np.newaxis, :]
    PRED = (y_hat - B) > threshold

    TPR = PRED[y_true.flatten() == 1, :].sum(axis=0) / Positive
    FPR = PRED[y_true.flatten() == -1, :].sum(axis=0) / Negative

    return np.squeeze(np.array([TPR, FPR]))

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

    def plot_roc_curve(fpr, tpr, title, aucScore):
        plt.plot(fpr, tpr, label='ROC - AUC:'+str(aucScore))
        plt.plot([0, 1], [0, 1], linestyle='--')
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.title(title)
        plt.legend()
        plt.show()

    from sklearn.metrics import roc_curve #just to test my results, remove later

    dataset_names = ['banana'] # ['banana','diabetis','flare-solar','image','ringnorm']
    k=0
    results = {}
    nrep = 2
    for name in dataset_names:
        Xtr, Xtest, Ytr, Ytest = getDataset(name)
        print('Dataset {}, Xtr:{}, Ytr:{},    Xtest:{}, Ytest:{}'.format(name, np.shape(Xtr), np.shape(Ytr), np.shape(Xtest), np.shape(Ytest)))

        # b) LOOCV-------------------------------------------------------------------------
        params = {'kernel': ['gaussian'], 'kernelparameter': [.1, .5, .9, .95, 1.0],
                  'regularization': [0]}
        cvkrr = cv(Xtr, Ytr, krr, params, loss_function=squared_error_loss, nrepetitions=nrep)
        y_pred_gauss = cvkrr.predict(Xtest)
        gauss_test_loss = squared_error_loss(Ytest,y_pred_gauss)
        fpr, tpr, thresholds = roc_curve(Ytest, y_pred_gauss)
        auc_gauss = np.abs(np.trapz(tpr, fpr))
        print('AUC for ' + str(name) + ',  LOOCV: %.2f' % auc_gauss)
        cvloss_gauss, kernelparameter_gauss, regularization_gauss = cvkrr.cvloss, cvkrr.kernelparameter, cvkrr.regularization

        params = {'kernel': ['polynomial'], 'kernelparameter': [1,2,3,4,5,6],
                  'regularization': [0]}
        cvkrr = cv(Xtr, Ytr, krr, params, loss_function=squared_error_loss, nrepetitions=nrep)
        y_pred_poly = cvkrr.predict(Xtest)
        poly_test_loss = squared_error_loss(Ytest, y_pred_poly)
        fpr, tpr, thresholds = roc_curve(Ytest, y_pred_poly)
        auc_poly = np.abs(np.trapz(tpr, fpr))
        print('AUC for '+str(name)+',  LOOCV: %.2f' % auc_poly)

        MyDict = dict()
        if poly_test_loss < gauss_test_loss:
            MyDict['cvloss'] = cvkrr.cvloss
            MyDict['kernel'] = cvkrr.kernel
            MyDict['kernelparameter'] = cvkrr.kernelparameter
            MyDict['regularization'] = cvkrr.regularization
            MyDict['y_pred'] = y_pred_poly
            results[name] = MyDict
            print('Dataset {}, test Loss:{}, kernel:{}, param:{}'.format(name,poly_test_loss, cvkrr.kernel,cvkrr.kernelparameter))
        else:
            MyDict['cvloss'] = cvloss_gauss
            MyDict['kernel'] = 'gaussian'
            MyDict['kernelparameter'] = kernelparameter_gauss
            MyDict['regularization'] = regularization_gauss
            MyDict['y_pred'] = y_pred_gauss
            results[name] = MyDict
            print('Dataset {}, test Loss:{}, kernel:{}, param:{}'.format(name, gauss_test_loss, 'gaussian',kernelparameter_gauss))

        if k==0:
            fig, ax = plt.subplots(3, 1, figsize=(9, 14))
            ax[0].scatter(Xtest[:, 0], Xtest[:, 1], c=Ytest)
            ax[0].set_title('Banana testing data set')

            ax[1].scatter(Xtest[:, 0], Xtest[:, 1], c=y_pred_gauss)
            ax[1].set_title('gaussian {}, predicted, test loss:{}, AUC :{}'.format(kernelparameter_gauss,round(gauss_test_loss,2),round(auc_gauss,2)))

            ax[2].scatter(Xtest[:, 0], Xtest[:, 1], c=y_pred_poly)
            ax[2].set_title('polynomial {}, predicted, test loss:{}, AUC :{}'.format(cvkrr.kernelparameter, round(poly_test_loss, 2), round(auc_poly, 2)))

            plt.show()


        # c) Proper CV + d) =================================================================================
        functions, l = [squared_error_loss, mean_absolute_error, zero_one_loss], ['squared_error_loss', 'mean_absolute_error', 'zero_one_loss']

        for f in range(len(functions)):
            loss_function, lf = functions[f], l[f]

            params = {'kernel': ['gaussian'], 'kernelparameter': [.1, .5, .9, 1.0],
                      'regularization': np.logspace(-2, 2, 10)}
            cvkrr = cv(Xtr, Ytr, krr, params, loss_function=loss_function, nrepetitions=nrep)
            y_pred_gauss = cvkrr.predict(Xtest)
            gauss_test_loss = loss_function(Ytest, y_pred_gauss)
            fpr, tpr, thresholds = roc_curve(Ytest, y_pred_gauss)
            auc_gauss = np.abs(np.trapz(tpr, fpr))
            print('AUC for ' + str(name) + ',  CV: %.2f' % auc_gauss)
            cvloss_gauss, kernelparameter_gauss, regularization_gauss = cvkrr.cvloss, cvkrr.kernelparameter, cvkrr.regularization
            #----------------------------------------------------------------------------------
            params = {'kernel': ['polynomial'], 'kernelparameter': [1, 2, 3, 4, 5, 6],
                      'regularization': np.logspace(-2, 2, 10)}
            cvkrr = cv(Xtr, Ytr, krr, params, loss_function=loss_function, nrepetitions=nrep)
            y_pred_poly = cvkrr.predict(Xtest)
            poly_test_loss = loss_function(Ytest, y_pred_poly)
            fpr, tpr, thresholds = roc_curve(Ytest, y_pred_poly)
            auc_poly = np.abs(np.trapz(tpr, fpr))
            print('AUC for ' + str(name) + ',  CV: %.2f' % auc_poly)
            #----------------------------------------------------------------------------------

            if poly_test_loss < gauss_test_loss:
                best_params = {'kernel': [cvkrr.kernel], 'kernelparameter': [cvkrr.kernelparameter],
                               'regularization': [cvkrr.regularization]}
                loss = ' '+str(round(poly_test_loss,3))
            else:
                best_params = {'kernel': ['gaussian'], 'kernelparameter': [kernelparameter_gauss],
                               'regularization': [regularization_gauss]}
                loss = ' '+str(round(gauss_test_loss,3))

            print('best_params for roc_fun',best_params)

            roc_cvkrr = cv(Xtr, Ytr, krr, best_params, loss_function=roc_fun, nrepetitions=nrep, roc_f=True)
            AUC = (np.abs(np.trapz(roc_cvkrr.tp, roc_cvkrr.fp)))
            print('The best AUC: ' % AUC)
            plot_roc_curve(roc_cvkrr.fp, roc_cvkrr.tp, '{} - CV, loss:{}'.format(name, lf+loss), AUC)

            if k == 0:
                fig, ax = plt.subplots(3, 1, figsize=(9, 14))
                ax[0].scatter(Xtest[:, 0], Xtest[:, 1], c=Ytest)
                ax[0].set_title('CV-Banana testing data set')

                ax[1].scatter(Xtest[:, 0], Xtest[:, 1], c=y_pred_gauss)
                ax[1].set_title('CV-gaussian {}, predicted, test loss:{}, {}'.format(kernelparameter_gauss,round(gauss_test_loss, 2),lf))

                ax[2].scatter(Xtest[:, 0], Xtest[:, 1], c=y_pred_poly)
                ax[2].set_title('CV-polynomial {}, predicted, test loss:{}, {}'.format(cvkrr.kernelparameter,round(poly_test_loss, 2),lf))

                plt.show()
        k+=1

    with open('results.p', 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    print('Main')
    Assignment4()


