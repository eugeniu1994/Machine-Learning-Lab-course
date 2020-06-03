from __future__ import division  # always use float division
import numpy as np
from scipy.spatial.distance import cdist  # fast distance matrices
from scipy.cluster.hierarchy import dendrogram  # you can use this
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D  # for when you create your own dendrogram
import random
from itertools import combinations_with_replacement
from matplotlib.pyplot import cm
import scipy.io as spio
import scipy as sp
from scipy.stats import multivariate_normal

def kmeans(X, k, max_iter=100, init_total_random=False):
    """ Performs k-means clustering
    Input:
    X: (n x d) data matrix with each datapoint in one column
    k: number of clusters
    max_iter: maximum number of iterations
    Output:
    mu: (k x d) matrix with each cluster center in one column
    r: assignment vector
    """
    n,d = np.shape(X)

    #Randomly initialize centroids as data points
    random_indexes = random.sample(range(n), k)
    mu = X[random_indexes, :]
    print(mu.shape)
    # Randomly initialize centroids by total random numbers for 2D data
    if init_total_random:
        x_max = np.max(X[:, 0])
        y_max = np.max(X[:, 1])

        mu = (x_max + y_max)/2 * np.random.rand(k, d)


    mu_old = np.zeros(mu.shape)  # store old centroids
    r = np.zeros(n)

    loss = 0
    plot = False
    for j in range(max_iter):
        #Compute distance to every centroid
        distances = cdist(X, mu, 'euclidean')
        # Assign data to closest centroid
        r_prim = r.copy()
        r = np.argmin(distances, axis=1)
        loss = np.sum(distances[np.arange(n), r] ** 2)
        if (r == r_prim).all():
            break
        # Compute new cluster center
        mu = new_centroids(X=X, r=r, k=k)

        print('Iterations: {},  Loss:{}'.format(j, loss))

    print('Final loss: {}'.format(loss))
    print('------')

    if plot:
        for i in range(n):
            plt.scatter(X[i, 0], X[i, 1], s=100, c='b')
        plt.scatter(mu[:, 0], mu[:, 1], marker='*', c='g', s=150)
        plt.show()

    return mu, r, loss

def new_centroids(X, r, k):
    """ Computes new cenctroids matrix for a given dataset and its correspondent assignment vector.
        Handle empty cluster by reinitializing them at a random data point

    Input:
    X: (n x d) data matrix
    r: (nx1) array/vector , assignment vector indicating to which cluster every datapoint be longs
    k: int, number of clusters

    Output:
    mu: (kxd) array, with the k- new centroids, with the kth cetroid corresponding to the kth cluster
    """
    n, d = X.shape
    # Compute new cluster center
    mu = np.zeros((k, d))
    for i in range(k):
        if i in r:
            # In this case the cluster is not empty and we can compute the new centroid as usual
            mu[i] = np.mean(X[r == i], axis=0)
        else:
            # In this case the cluster is empty and we reinitialize the cluster at some random data point
            mu[i] = X[random.randrange(n), :]

    return mu

def norm_pdf(X, mu, C):
    """ Computes probability density function for multivariate gaussian
    Input:
    X: (n x d) data matrix with each datapoint
    mu: vector for center
    C: covariance matrix
    Output:
    pdf value for each data point
    """
    d = np.shape(X)[1]
    #determinant when the matrix is non-singular.
    if np.linalg.det(C) == 0: #pseudo determinant
        #print('Matrix is not invertible')
        eig_values = np.linalg.eigh(C) # np.linalg.eig(C)
        C_det = 1.0
        for i in range(len(eig_values)):
            C_det *= np.product(eig_values[i][eig_values[i] > 1e-12])
        #C_det = np.product(eig_values[eig_values > 1e-12])      # this gives:TypeError: '>' not supported between instances of 'tuple' and 'float'
        C_inv = np.linalg.pinv(C)
    else:
        C_det = np.linalg.det(C)
        C_inv = np.linalg.inv(C)

    down = np.sqrt((2 * np.pi) ** d * C_det) # , #down = ((2 * np.pi) ** d/2) * np.sqrt(C_det)
    up = -np.einsum('...i,ij,...j->...', X - mu, C_inv, X - mu)/2  #  einsum (x-mu)T.Sigma-1.(x-mu)
    pdf = np.exp(up) / down

    return pdf

def em_gmmv(X, k, max_iter=100, init_kmeans=False, eps=1e-3, Iterations = False):
    """ Implements EM for Gaussian Mixture Models
        Input:
        X: (n x d) data matrix
        k: number of clusters
        max_iter: maximum number of iterations
        init_kmeans: whether kmeans should be used for initialisation
        eps: when log likelihood difference is smaller than eps, terminate loop
        Output:
        mpi: 1 x k matrix of priors
        mu: (k x d) matrix with each cluster center in one column
        sigma: list of d x d covariance matrices
    """
    n,d = np.shape(X)
    if init_kmeans:
        print('Init by k-means ')
        mu, _, _ = kmeans(X, k=k)
        mu = np.asmatrix(mu)
    else:
        print('Init by random ')
        rand_row = np.random.randint(low=0, high=n, size=k)
        mu = np.asmatrix([X[row_idx, :] for row_idx in rand_row])
    sigma = np.array([np.eye(d) for _ in range(k)])
    mpi = np.ones(k) / k
    g = np.full((n, k), fill_value=1 / k) #gamma

    logLik = 1.0
    prev_logLik = 0

    def Step_E():
        logLik = 0
        for j in range(k):
            pdf = norm_pdf(X, np.ravel(mu[j, :]), sigma[j, :])
            g[:, j] = pdf
            logLik += np.log(pdf.sum())
        up = g * mpi
        down = up.sum(axis=1)[:, np.newaxis]
        g[:,:] = up / down
        return logLik

    def Step_M():
        for j in range(k):
            nk = g[:, j].sum()
            mpi[j] = nk/n

            sigma_j = np.zeros((d, d))
            for i in range(n):
                sigma_j += g[i, j] * ((X[i, :] - mu[j, :]).T * (X[i, :] - mu[j, :]))

            mu[j] = (X * g[:,j][:, np.newaxis]).sum(axis=0) / nk
            sigma[j] = sigma_j / nk

    iter = 0
    while (abs(logLik - prev_logLik) > eps and iter < max_iter):
        prev_logLik = logLik

        logLik=Step_E()
        Step_M()

        iter += 1
        #print('Iter:{},  log-likelihood:{}, error:{}'.format(iter,logLik,abs(logLik - prev_logLik)))
    print('Finished at {} iter, Log-likelihood:{}'.format(iter,logLik))
    if Iterations:
        return mpi, mu, sigma, logLik, iter
    return mpi, mu, sigma, logLik

def plot_gmm_solution(X, mu, sigma, title='', ax=None):
    """ Plots covariance ellipses for GMM
    Input:
    X: (n x d) data matrix
    mu: (k x d) matrix
    sigma: list of d x d covariance matrices
    """
    ps2_test = False
    if ax is None:
        ps2_test = True
        fig, ax = plt.subplots(figsize=(6, 6))

    ax.scatter(X[:, 0], X[:, 1], s=50, c='tab:blue')
    ax.scatter(mu[:, 0].A1, mu[:, 1].A1, c='r', s=150, marker='x',lw=2)
    t = np.linspace(0, 2 * np.pi, 100)
    for i in range(np.shape(mu)[0]):
        u = mu[i,0]  # x-position center
        v = mu[i,1] # y-position center

        p= .9
        s = -2 * np.log(1 - p)
        #print(sigma)
        D,V = np.linalg.eig(sigma[i] * s)
        a = (V * np.sqrt(D)) @ [np.cos(t), np.sin(t)]
        ax.plot(a[0,:] + u, a[1,:] + v, c='g',lw=2)

    ax.set_title(title)
    ax.grid(color='lightgray', linestyle='--')
    custom_lines = [Line2D([0], [0], color='tab:blue', lw=1, marker='o'),
                    Line2D([0], [0], color='g', lw=4),
                    Line2D([0], [0], color='r', lw=1, marker='x')]
    ax.legend(custom_lines, ['Data points', 'GMM Covariance', 'Mean vectors'])
    if ps2_test:
        plt.show()

def Assignment8():
    # Set params
    k_means=False
    gmm = True
    vis_data = True
    vis_loss = True
    manipulate_data = False
    k = 2
    rep = 1000

    # Load 2_gaussians dataset
    gaussians = np.load('../data/2_gaussians.npy')
    X = gaussians.T

    # Visualize  data
    if manipulate_data:
        indexes_up = np.argwhere(X[:, 1]>0.08)
        indexes_down = np.argwhere(X[:, 1] <= 0.08)
        X_up = np.squeeze(X[indexes_up, :])
        X_up[:,1] = X_up[:,1] + 0
        X_down = np.squeeze(X[indexes_down, :])

        mu_var = np.zeros((2,2))
        mu_up = np.mean(X_up, axis=0)
        mu_down = np.mean(X_down, axis=0)
        mu_var[0, :] = mu_up
        mu_var[1, :] = mu_down

        # Calculate loss

        loss = np.sum(cdist(XA=X_up,XB=mu_up[np.newaxis, :])**2) + np.sum(cdist(XA=X_down,XB=mu_down[np.newaxis, :])**2)
        print('Kmeans loss with a "perfect" separation: {}'.format(loss))

        # Plot "perfect" separation
        figp, axep = plt.subplots(figsize=(7, 5))
        axep.scatter(X_up[:, 0], X_up[:, 1], c='red')
        axep.scatter(X_down[:, 0], X_down[:, 1], c='blue')
        axep.scatter(mu_var[:, 0], mu_var[:, 1], marker='*', c='orange', s=200)
        axep.set_xlabel('x', fontsize=12)
        axep.set_ylabel('y', fontsize=12)
        plt.tick_params(axis='both', which='major', labelsize=11)
        plt.tick_params(axis='both', which='minor', labelsize=10)

        #X[indexes_up, 1] = X[indexes_up, 1]

    if vis_data:
        fig1, axe1 = plt.subplots(figsize=(7, 5))
        axe1.scatter(X[:, 0], X[:, 1])


    # KMEANS ANALYZE
    losses = np.zeros(rep)
    for rep_cur in range(rep):
        if k_means:
            mu, r, loss = kmeans(X, k=k, init_total_random=False)

        elif gmm:
            mpi, mu, sigma, loss, Iterations_kmeans = em_gmmv(X, k=k, init_kmeans=False, Iterations=True)

        else:
            print('ERROR')

        losses[rep_cur] = loss
        # plot min loss kmeans


        if loss > 7.32: # so cause we know the max is at 7.33
            if k_means:
                plt.figure(num='Global Minima kmeans', figsize=(5, 5))
                ax = plt.gca()
                ax.scatter(X[r == 1, 0], X[r == 1, 1], c='red')
                ax.scatter(X[r == 0, 0], X[r == 0, 1], c='blue')
                ax.scatter(mu[:, 0], mu[:, 1], marker='*', c='orange', s=200)
                break
            elif gmm:
                plot_gmm_solution(X=X, mu=mu, sigma=sigma, title='Global Maxima GMM', ax=None)
                break
            else:
                print('ERROR')

    if vis_loss:
        plt.figure(num='Loss kmeans', figsize=(7, 5))
        ax = plt.gca()
        ax.plot(np.arange(rep), np.sort(losses))
        ax.set_xlabel('Computation', fontsize=12)
        ax.set_ylabel('GMM Log-likelihood', fontsize=12)
        plt.tick_params(axis='both', which='major', labelsize=11)
        plt.tick_params(axis='both', which='minor', labelsize=10)

        if k_means:
            print('----')
            print('Min loss: {}'.format(np.min(losses)))
            print('----')

        elif gmm:
            print('----')
            print('Max log-likelihood: {}'.format(np.max(losses)))
            print('----')

        else:
            print('ERROR')
            
def gammaidx(X, k):
    '''  Compute the gamma values using the k nearest neighborus for the each data point of X
    Definition:  y = gammaidx(X, k)
    Input:       X        - DxN array of N data points with D features
                 k        - int, number of neighbours used
    Output:      y        - Nx1 array, contains the gamma value for each data point
    '''
    N = X.shape[0]
    y = np.zeros(N)
    # Find k nearest neighbour for every data sample and compute its correspondent gamma value
    for i in range(N):
        data = np.tile(X[i, :], (N, 1))
        distance = np.linalg.norm(x=(data - X), axis=1)
        idx = np.argpartition(distance, k)
        # Sort 4 largest distances and add them up
        y[i] = (1/k)*np.sum(distance[idx[0:k+1]])
    return y
Assignment8()
