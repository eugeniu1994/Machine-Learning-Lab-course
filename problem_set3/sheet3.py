
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

def zero_one_loss(y_true, y_pred): 
    n= len(y_true)
    s = np.sum(y_true == np.sign(y_pred))
    return (1.0/n)*( n - s)

def mean_absolute_error(y_true, y_pred):
    return (1.0 / len(y_true)) * (np.abs(y_pred - y_true).sum())

def squared_error_loss(y_true, y_pred):
    assert (len(y_true) == len(y_pred))
    loss = np.mean((y_true - y_pred) ** 2)
    return loss

def cv(X, y, method, params, loss_function=mean_absolute_error, nfolds=10, nrepetitions=5, roc_f=False, verbose = True):
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
            J = K + self.regularization * np.eye(n)
            if np.linalg.matrix_rank(J) != J.shape[0]:  # check singularity
                print('Matrix is singular=================================')
                J = J + (np.eye(np.shape(J)[0]) * 0.5)  # regularization
            self.alpha = np.linalg.solve(J, y).T

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
    plot_a = True    # plot ass (a) distances against energy differences
    train_cv = False  # perform 5fold CV
    optimal_train = True  # plot error(N) of ass 3d
    over_under_fit = True # plot under-, over- and optimal fited data for train and test set

    # Load dataset
    mat = scipy.io.loadmat('./data/qm7.mat')
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
        # Pandas data frame, use e.g 10.000.000 datapoints intead of 25.000.000
        df = pd.DataFrame({"x_diff": x_dist, "y_diff": y_dist})

        dfSample = df.sample(10000000)  # This is the importante line
        xdataSample, ydataSample = dfSample["x_diff"], dfSample["y_diff"]

        # Plot distances
        fig = plt.figure(num='Distance against energy difference', figsize=(7.2, 4.45))
        ax = fig.add_subplot(111)
        ax.scatter(xdataSample, ydataSample, s=1)
        ax.set_xlabel('d')
        ax.set_ylabel(r'$ \Delta E [kcal/mol]$')
        plt.tick_params(axis='both', which='major', labelsize=11)
        plt.tick_params(axis='both', which='minor', labelsize=10)

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
    y_cv = np.squeeze(y_train[indexes_train[:2500], :])

    # Width parameter σ of the Gaussian kernel. Candidates are quantiles of pairwise Euclidean distances.
    gaussian_width = np.quantile(x_dist, [0.1, 0.5, 0.9])  # 1%, 2%,...99% quantiles of pairwise eucl diatances
    print(gaussian_width)
    # Regularization parameter C. Use logarithmically scaled values between 10−7 and 100 as candidates.
    regularization_c = np.logspace(-7, 0, 10)


    # Dictionary with parameter for the cv function
    params = {'kernel': ['gaussian'], 'kernelparameter': gaussian_width,
              'regularization': regularization_c}

    if train_cv:
        cvkrr = cv(X_cv, y_cv, krr, params, loss_function=mean_absolute_error,
                   nrepetitions=5, nfolds=5)
        # Error on the test set
        y_pred = cvkrr.predict(X_test)
        test_error = mean_absolute_error(y_true=np.squeeze(y_test), y_pred=y_pred)
        print('Test Error: {}'.format(test_error))
    else:
        pass

    # -----------------------------------------------------------------------------------------------------------------
    # (d) Keep C and σ fixed and plot the MAE on the test set as a function of the number n of training samples with
    # n from 100 to 5000.

    if optimal_train:
        C = 2.1544346900318867e-05
        width = 23.78351003
        n_model_train = np.logspace(np.log10(100), np.log10(5000), 10).astype('int')
        errors = np.zeros(10)

        for index, n_cur in enumerate(n_model_train):
            # Train
            model = krr(kernel='gaussian', kernelparameter=width, regularization=C)
            model.fit(X=X_train[:n_cur, :], y= np.squeeze(y_train[:n_cur, :]))
            # Test model on the test set
            y_pred = model.predict(X=X_test)
            errors[index] = mean_absolute_error(np.squeeze(y_test), y_pred)

        # plot error(N_train)
        plt.figure(figsize=(5, 5))
        ax = plt.gca()
        ax.plot(n_model_train, errors, '-ob')
        ax.set_xlabel('N')
        ax.set_ylabel('MAE')
        plt.tick_params(axis='both', which='major', labelsize=11)
        plt.tick_params(axis='both', which='minor', labelsize=10)

    # -----------------------------------------------------------------------------------------------------------------
    # (e) Three models: 1 that underfits, 1 that fits well and 1 that overfits + data showing this
#2.15443
    if over_under_fit:
        C_under =2.1e-02
        C_over = 1e-09
        C_optimal = 2.1544346900318867e-05

        C_all = [C_under, C_over, C_optimal]
        width = 23.78351003

        y_test = np.squeeze(y_test)

        # randomly select 1000 points for training
        indexes_train = np.arange(0, 5000)
        np.random.shuffle(indexes_train)

        X_cv = X_train[indexes_train[:1000], :]
        y_cv = np.squeeze(y_train[indexes_train[:1000], :])


        for c_cur in C_all:
            model = krr(kernel='gaussian', kernelparameter=width, regularization=c_cur)
            model.fit(X=X_cv, y=y_cv)

            y_pred_train = model.predict(X_cv)
            y_pred_test = model.predict(X_test)

            # Plot

            fig = plt.figure(num='Model TRAIN for C='+str(c_cur), figsize=(7.2, 4.45))
            ax = fig.add_subplot(111)
            ax.scatter(y_cv, y_cv, label='ideal')
            ax.scatter(y_cv, y_pred_train, label='predicted')

            ax.set_xlabel('y_{test}')
            ax.set_ylabel('y_{test/predicted}')

            ax.set_xlabel('y_{train}')
            ax.set_ylabel('y_{train/predicted}')

            plt.tick_params(axis='both', which='major', labelsize=11)
            plt.tick_params(axis='both', which='minor', labelsize=10)

            fig = plt.figure(num='Model TEST for C='+str(c_cur), figsize=(7.2, 4.45))
            ax = fig.add_subplot(111)
            ax.scatter(y_test, y_test, label='ideal')
            ax.scatter(y_test, y_pred_test, label='predicted')

            ax.set_xlabel('y_{test}')
            ax.set_ylabel('y_{test/predicted}')

            plt.tick_params(axis='both', which='major', labelsize=11)
            plt.tick_params(axis='both', which='minor', labelsize=10)

            print('Regularization c: {} , Gaussian width: {}'.format(c_cur, width))
            print('MAE ERROR TRAIN: {}'.format(mean_absolute_error(y_pred_train, y_cv)))
            print('MAE ERROR TEST: {}'.format(mean_absolute_error(y_pred_test, y_test)))
            print('----')
    else:
        pass

    plt.legend()
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

    dataset_names = ['banana','diabetis','flare-solar','image','ringnorm']
    k=0
    results = {}
    nrep = 5
    verbose = False
    for name in dataset_names:
        Xtr, Xtest, Ytr, Ytest = getDataset(name)
        print('Dataset {}, Xtr:{}, Ytr:{},    Xtest:{}, Ytest:{}'.format(name, np.shape(Xtr), np.shape(Ytr), np.shape(Xtest), np.shape(Ytest)))

        # b) LOOCV-------------------------------------------------------------------------
        #linear
        params = {'kernel': ['linear'], 'kernelparameter': [0],
                  'regularization': [0]}
        cvkrr = cv(Xtr, Ytr, krr, params, loss_function=squared_error_loss, nrepetitions=nrep, verbose = verbose)
        y_pred_linear = cvkrr.predict(Xtest)
        linear_test_loss = squared_error_loss(Ytest, y_pred_linear)
        cvloss, kernel, kernelparameter, regularization = cvkrr.cvloss, cvkrr.kernel, cvkrr.kernelparameter, cvkrr.regularization
        y_pred = y_pred_linear
        test_loss = linear_test_loss
        #gaussian======================================
        params = {'kernel': ['gaussian'], 'kernelparameter': [.1, .5, .9, .95, 1.0],
                  'regularization': [0]}
        cvkrr = cv(Xtr, Ytr, krr, params, loss_function=squared_error_loss, nrepetitions=nrep,verbose = verbose)
        y_pred_gauss = cvkrr.predict(Xtest)
        gauss_test_loss = squared_error_loss(Ytest,y_pred_gauss)
        if gauss_test_loss < linear_test_loss:
            test_loss = gauss_test_loss
            cvloss = cvkrr.cvloss
            kernel = cvkrr.kernel
            kernelparameter = cvkrr.kernelparameter
            regularization = cvkrr.regularization
            y_pred = y_pred_gauss
        #polynomial=====================================
        params = {'kernel': ['polynomial'], 'kernelparameter': [1,2,3,4,5],
                  'regularization': [0]}
        cvkrr = cv(Xtr, Ytr, krr, params, loss_function=squared_error_loss, nrepetitions=nrep,verbose = verbose)
        y_pred_poly = cvkrr.predict(Xtest)
        poly_test_loss = squared_error_loss(Ytest, y_pred_poly)
        if poly_test_loss < test_loss:
            test_loss = poly_test_loss
            cvloss = cvkrr.cvloss
            kernel = cvkrr.kernel
            kernelparameter = cvkrr.kernelparameter
            regularization = cvkrr.regularization
            y_pred = y_pred_poly
        loss = ' ' + str(round(test_loss, 3))

        MyDict = dict()
        MyDict['cvloss'] = cvloss
        MyDict['kernel'] = kernel
        MyDict['kernelparameter'] = kernelparameter
        MyDict['regularization'] = regularization
        MyDict['y_pred'] = y_pred
        results[name] = MyDict

        best_params = {'kernel': [kernel], 'kernelparameter': [kernelparameter],
                       'regularization': [regularization]}
        print('best_params ',best_params)
        roc_cvkrr = cv(Xtr, Ytr, krr, best_params, loss_function=roc_fun, nrepetitions=nrep, roc_f=True,verbose = verbose)
        AUC = round((np.abs(np.trapz(roc_cvkrr.tp, roc_cvkrr.fp))),3)
        plot_roc_curve(roc_cvkrr.fp, roc_cvkrr.tp, '{} - LOOCV, loss:{}'.format(name, 'squared_error_loss' + loss), AUC)

        print('Dataset:{}, cvloss:{}, test loss:{},kernel:{},kernelparameter:{},regularization:{}, AUC:{}'.format(str(name),
                                                                                                                  cvloss,test_loss,kernel,
                                                                                                                  kernelparameter,regularization,AUC))

        if k==0:
            fig, ax = plt.subplots(2, 1, figsize=(10, 10))
            ax[0].scatter(Xtest[:, 0], Xtest[:, 1], c=Ytest)
            ax[0].set_title('Banana testing data')

            ax[1].scatter(Xtest[:, 0], Xtest[:, 1], c=y_pred)
            ax[1].set_title('Banana predicted')

            plt.show()

        # c) Proper CV + d) =================================================================================
        functions, l = [squared_error_loss, mean_absolute_error, zero_one_loss], ['squared_error_loss', 'mean_absolute_error', 'zero_one_loss']
        for f in range(len(functions)):
            loss_function, lf = functions[f], l[f]

            params = {'kernel': ['gaussian'], 'kernelparameter': [.1, .5, .9, .95, 1.0],
                      'regularization': np.logspace(-2, 2, 10)}
            cvkrr = cv(Xtr, Ytr, krr, params, loss_function=loss_function, nrepetitions=nrep)
            y_pred_gauss = cvkrr.predict(Xtest)
            gauss_test_loss = loss_function(Ytest, y_pred_gauss)
            cvloss, kernel, kernelparameter, regularization = cvkrr.cvloss, cvkrr.kernel, cvkrr.kernelparameter, cvkrr.regularization
            y_pred = y_pred_gauss
            test_loss = gauss_test_loss
            #----------------------------------------------------------------------------------
            params = {'kernel': ['polynomial'], 'kernelparameter': [1, 2, 3, 4, 5],
                      'regularization': np.logspace(-2, 2, 10)}
            cvkrr = cv(Xtr, Ytr, krr, params, loss_function=loss_function, nrepetitions=nrep)
            y_pred_poly = cvkrr.predict(Xtest)
            poly_test_loss = loss_function(Ytest, y_pred_poly)
            if poly_test_loss < gauss_test_loss:
                test_loss = poly_test_loss
                cvloss = cvkrr.cvloss
                kernel = cvkrr.kernel
                kernelparameter = cvkrr.kernelparameter
                regularization = cvkrr.regularization
                y_pred = y_pred_poly
            loss = ' ' + str(round(test_loss, 3))

            best_params = {'kernel': [kernel], 'kernelparameter': [kernelparameter],
                           'regularization': [regularization]}
            print('best_params for roc_fun',best_params)

            roc_cvkrr = cv(Xtr, Ytr, krr, best_params, loss_function=roc_fun, nrepetitions=nrep, roc_f=True)
            AUC = round((np.abs(np.trapz(roc_cvkrr.tp, roc_cvkrr.fp))), 3)
            plot_roc_curve(roc_cvkrr.fp, roc_cvkrr.tp, '{} - CV, loss:{}'.format(name, lf + loss),AUC)

            if k == 0:
                fig, ax = plt.subplots(2, 1, figsize=(10, 10))
                ax[0].scatter(Xtest[:, 0], Xtest[:, 1], c=Ytest)
                ax[0].set_title('Banana testing data')
                ax[1].scatter(Xtest[:, 0], Xtest[:, 1], c=y_pred)
                ax[1].set_title('Banana predicted')

                plt.show()

            print('Dataset:{}, cvloss:{}, test loss:{},kernel:{},kernelparameter:{},regularization:{}, AUC:{}'.format(str(name),cvloss,test_loss,
                                                                                                              kernel,kernelparameter,regularization,AUC))

        k+=1

    with open('results.p', 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    print('Main')
    #Assignment3()
    #Assignment4()

    see_results_parameters = False  #best results for assignment 4 (set True to visualize)
    if see_results_parameters:
        data = set(['cvloss', 'kernel', 'kernelparameter', 'regularization'])
        with open('results.p', 'rb') as f:
            x = pickle.load(f)
            for dictionary in x:
                print('Data set :' + dictionary)
                for key, value in x[dictionary].items():
                    if key in data:
                        print(str(key) + ':' + str(value))
                    else:
                        print(str(key) + ':', np.shape(value)) #y_pred

                print('\n')










