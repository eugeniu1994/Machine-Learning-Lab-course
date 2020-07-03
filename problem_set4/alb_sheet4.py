""" ps4_implementation.py

PUT YOUR NAME HERE:
<FIRST_NAME><LAST_NAME>


Complete the classes and functions
- svm_qp
- plot_svm_2d
- neural_network
Write your implementations in the given functions stubs!


(c) Felix Brockherde, TU Berlin, 2013
    Jacob Kauffmann, TU Berlin, 2019
"""
import scipy.linalg as la
import matplotlib.pyplot as plt
import sklearn.svm
from cvxopt.solvers import qp
from cvxopt import matrix as cvxmatrix
import numpy as np
import torch
from scipy.spatial.distance import cdist
import cvxopt
from torch.nn import Module, Parameter, ParameterList

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

        print(alpha)
        sv_threshold = 1e-5

        indexes_sv = np.argwhere((alpha > sv_threshold)).squeeze()

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
