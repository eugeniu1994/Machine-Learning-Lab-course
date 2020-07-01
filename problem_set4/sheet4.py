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

from torch.optim import SGD #just to compare with my implementation
from torch.nn import Module, Parameter, ParameterList

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
    plt.title(title)
    plt.show()

class neural_network(Module):
    def __init__(self, layers=[2,100,2], scale=.1, p=None, lr=None, lam=None):
        super().__init__()
        self.weights = ParameterList([Parameter(scale*torch.randn(m, n)) for m, n in zip(layers[:-1], layers[1:])])
        self.biases = ParameterList([Parameter(scale*torch.randn(n)) for n in layers[1:]])

        self.p = p
        self.lr = lr if lr is not None else 1e-3
        self.lam = lam if lam is not None else 0.0
        self.train = False
        self.L = len(self.weights)

        self.memory = {}

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
            A = self.relu(X=A,W=self.weights[l],b=self.biases[l], layer=str(l+1))
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
                L2_regularization +=np.sum(np.square(w))

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

    def softmax_derivative(self, s):
        s = s[0]
        jacobian_m = torch.diag(s)
        for i in range(len(jacobian_m)):
            for j in range(len(jacobian_m)):
                if i == j:
                    jacobian_m[i][j] = s[i] * (1 - s[i])
                else:
                    jacobian_m[i][j] = -s[i] * s[j]
        return jacobian_m

    def backpropagation(self, X, Y, bs):
        derivates = {}

        self.memory["A_0"] = X
        Y = Y.detach().clone().numpy()
        A = (self.memory["A_" + str(self.L)]).detach().clone().numpy()


        #dA = -np.divide(Y, A) + np.divide(1 - Y, 1 - A) #normal cross entropy derivative
        dA = A - Y                                       #stable cross entropy derivative

        #here add softmax instead of relu_derivative---to do
        dZ = (dA * self.relu_derivative(self.memory["Z_" + str(self.L)]).detach().clone().numpy()).T          #with relu derivative
        #dZ = np.dot(self.softmax_derivative(self.memory["Z_" + str(self.L)].T).detach().clone().numpy(), dA).T #with softmax derivative

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

            print('Cost ', output)
            #print(self.memory.keys())

            derivates = self.backpropagation(Xtrain[I], ytrain[I], bs)
            #print(derivates.keys())
            self.SGD_updates(derivates,bs,True) #normalize

            outval = self.forward(Xval)
            Lval += [self.loss(outval, yval).item()]
            Aval += [np.array(outval.argmax(-1) == yval.argmax(-1)).mean()]

        if plot:
            plt.plot(range(nsteps), Ltrain, label='Training loss')
            plt.plot(range(nsteps), Lval, label='Validation loss')
            plt.plot(range(nsteps), Aval, label='Validation acc')
            plt.legend()
            plt.show()

    # this is using implemented SGD
    def fit_(self, X, y, nsteps=1000, bs=100, plot=False):
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
            plt.plot(range(nsteps), Ltrain, label='Training loss')
            plt.plot(range(nsteps), Lval, label='Validation loss')
            plt.plot(range(nsteps), Aval, label='Validation acc')
            plt.legend()
            plt.show()

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
        print('X:{},Y:{}'.format(np.shape(X), np.shape(Y)))
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

        suport_vectors = alpha > 1e-4 #keep only non zero alphas
        idx = np.arange(len(alpha))[suport_vectors]
        self.alpha_sv = alpha[suport_vectors]
        self.X_sv = X[suport_vectors]
        self.Y_sv = Y[suport_vectors]
        print("%d support vectors out of %d points" % (len(self.alpha_sv), m))

        self.b = 0.0
        for i in range(len(self.alpha_sv)):
            self.b += self.Y_sv[i]
            self.b -= np.sum(self.alpha_sv * self.Y_sv * K[idx[i], suport_vectors])
        self.b = self.b / len(self.alpha_sv)

        self.w = np.sum((alpha @ Y) * X, axis=0)

    def predict(self, X):
        #prediction = np.dot(X, self.w) + self.b
        ypred = np.zeros(len(X))
        for i in range(len(X)):
            for vector, vector_y, vector_x in zip(self.alpha_sv, self.Y_sv, self.X_sv):
                k = buildKernel(X=X[i], X_train=vector_x, kernel=self.kernel, kernelparameter=self.kernelparameter)
                ypred[i] += vector * vector_y * k
        prediction = ypred + self.b

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

def roc_fun(y_true, y_hat, points = 500, threshold = 0.0):
    y_hat = y_hat[:, np.newaxis]
    Negative = np.array(y_true.flatten() == -1).sum() #negative class
    Positive = len(y_true) - Negative                 #positive class
    assert Positive>0 and Negative>0, 'Negative or Positive is zero, zero division exception (No positive or negative classes, Inbalanced data)'

    B = np.linspace(-3, 3, points)[np.newaxis, :]
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

    params = {'kernel': ['gaussian'], 'kernelparameter': [.1, .5, .9, 1.0, 2.0, 3.0],
              'regularization': np.linspace(1, 500, num=20)}

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
    model = svm_qp(kernel='linear', C=None)
    #model = svm_qp(kernel='gaussian', C=None) #to do find parameters for underfitting
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
    CV = cv(X_tr, Y_tr, svm_qp, best_params, loss_function=roc_fun, nrepetitions=2, roc_f = True)
    AUC = round((np.abs(np.trapz(CV.tp, CV.fp))), 3)
    plot_roc_curve(CV.fp, CV.tp, '', AUC)

if __name__ == '__main__':
    print('Main')
    Assignment4()