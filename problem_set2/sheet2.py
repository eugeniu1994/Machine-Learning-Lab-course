from __future__ import division  # always use float division
import numpy as np
from scipy.spatial.distance import cdist  # fast distance matrices
from scipy.cluster.hierarchy import dendrogram  # you can use this
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D  # for when you create your own dendrogram
import scipy.io as spio  # used to lod .mat dataset
import random

# Assignment 1
def kmeans(X, k, max_iter=100, plotData=False):
    """ Performs k-means clustering
    Input:
    X: (n x d) data matrix with each datapoint in one column
    k: number of clusters
    max_iter: maximum number of iterations
    Output:
    mu: (k x d) matrix with each cluster center in one column
    r: assignment vector
    """
    n, d = np.shape(X)
    # Randomly initialize centroids as data points
    random_indexes = random.sample(range(n), k)
    mu = X[random_indexes, :]
    r = np.zeros(n)
    loss = 0
    for j in range(max_iter):
        # Compute distance to every centroid
        distances = cdist(X, mu, 'euclidean')
        # Assign data to closest centroid
        r_prim = r.copy()
        r = np.argmin(distances, axis=1)
        loss = np.sum(distances[np.arange(n), r] ** 2)
        if (r == r_prim).all():
            break
        # Compute new cluster center
        for i in range(k):  # Handle empty cluster by reinitializing them at a random data point if cluster is empty
            mu[i] = np.mean(X[r == i], axis=0) if i in r else X[np.random.randrange(n), :]

        print('Iterations: {},  Loss:{}'.format(j, loss))

    if plotData:
        plt.scatter(X[:, 0], X[:, 1], s=100, c=r)
        plt.scatter(mu[:, 0], mu[:, 1], marker='*', c='g', s=150)
        plt.show()

    return mu, r, loss

# Assignment 2
def kmeans_agglo(X, r, verbose=False):
    """ Performs agglomerative clustering with k-means criterion
    Input:
    X: (n x d) data matrix with each datapoint in one column
    r: assignment vector
    Output:
    R: (k-1) x n matrix that contains cluster memberships before each step
    kmloss: vector with loss after each step
    mergeidx: (k-1) x 2 matrix that contains merge idx for each step
    """
    n, d = np.shape(X)

    def kmeans_crit(X, r, mu):
        """ Computes k-means criterion
        Input:
        X: (n x d) data matrix with each datapoint in one column
        r: assignment vector
        Output:
        value: scalar for sum of euclidean distances to cluster centers
        """
        # Init loss value
        distances = cdist(X, mu, 'euclidean')
        clusters = np.unique(r)
        loss = 0.0
        for j in range(n):
            loss += np.squeeze(distances[j, np.argwhere(clusters == r[j])] ** 2)

        return loss

    def compute_New_Centroids(X, r):
        new_k = len(np.unique(r))  # number of different values  i.e. clusters
        j = 0
        new_mu = np.zeros((new_k, d))
        for i in np.unique(r):
            new_mu[j] = np.mean(X[r == i], axis=0) if i in r else X[random.randrange(n), :]
            j += 1
        return new_mu

    # r = [i for i in range(X.shape[0])] #this is just for test, comment this line, to get the r from parameters
    r = np.array(r)
    R, kmloss, mergeidx = [], [], []
    R.append(r.copy())
    centroids = compute_New_Centroids(X, r)
    kmloss.append(cdist(X, centroids, 'euclidean').sum())
    # kmloss.append(kmeans_crit(X=X, r=r, mu=centroids))
    m = len(centroids)
    while (m > 1):
        distance_matrix = cdist(centroids, centroids, 'euclidean')
        np.fill_diagonal(distance_matrix, np.inf)  # fill diagonal with infinity, in order to get the min value
        min_idx = np.where(distance_matrix == distance_matrix.min())[0]  # x index where min value

        u = np.unique(r)
        r[np.where(r == u[min_idx[1]])] = u[min_idx[0]]
        mergeidx.append([u[min_idx[0]], u[min_idx[1]]])  # save merged indexes
        R.append(r.copy())  # save current assignment vector
        centroids = compute_New_Centroids(X, r)
        kmloss.append(round(cdist(X, centroids, 'euclidean').sum(), 2))  # save distance loss
        # kmloss.append(np.round((kmeans_crit(X=X, r=r, mu=centroids)),2))
        m = len(centroids)

        # print('mergeidx ', mergeidx)
        # print(kmloss)
        # print('\n')

    if verbose:
        print('R  ', np.shape(R))
        print('kmloss  ', np.shape(kmloss))
        print('mergeidx  ', np.shape(mergeidx))

    return R, kmloss, mergeidx

# Assignment 3
def agglo_dendro(kmloss, mergeidx, verbose=False, title='', ax=None):
    """ Plots dendrogram for agglomerative clustering
    Input:
    kmloss: vector with loss after each step
    mergeidx: (k-1) x 2 matrix that contains merge idx for each step
    """
    if verbose:
        print(kmloss)
        print('level loses', np.shape(kmloss))
        print('merged indexes \n', mergeidx)

    def helper(x0, x1, y0, y1, next_loss):
        return ([[x0, x0, x1, x1], [y0, next_loss, next_loss, y1]]), [(x0 + x1) / 2.0, next_loss]

    kmloss = np.array(kmloss)
    MyDict = {}
    keys = range(len(kmloss))
    ps2_test = False
    if ax is None:
        ps2_test = True
        fig, ax = plt.subplots()

    UseSortedLabel = True  #
    if UseSortedLabel:
        m = np.ravel(np.array(mergeidx))
        idx = np.unique(m, return_index=True)[1]
        m = [m[index] for index in sorted(idx)]

    for i in keys:
        ax.scatter(i, 0, c='k')
        if UseSortedLabel:
            MyDict[m[i]] = dict({'label': str(m[i]), 'x': i, 'y': 0})
        else:
            MyDict[i] = dict({'label': str(i), 'x': i, 'y': 0})

    kmloss = 1.0 / kmloss
    kmloss = kmloss[1:]
    for i, (next_level, (c1, c2)) in enumerate(zip(kmloss, mergeidx)):
        x0, y0 = MyDict[c1]['x'], MyDict[c1]['y']
        x1, y1 = MyDict[c2]['x'], MyDict[c2]['y']

        if verbose:
            print('Iteration: {}, Clusters: ({},{}) current level loss:{}'.format(i, c1, c2, next_level))
            print('Merged clusters: c{}:({},{}) & c{}:({},{})'.format(c1, x0, y0, c2, x1, y1))

        line, center = helper(x0, x1, y0, y1, next_level)
        ax.plot(*line)
        ax.scatter(*center)

        if verbose:
            print('New centroid is {}, change the map[{}]:{}'.format(center, c1, MyDict[c1]))

        MyDict[c1]['x'] = center[0]
        MyDict[c1]['y'] = center[1]
        # MyDict[c1]['label'] = '({0},{1})'.format(MyDict[c1]['label'], MyDict[c2]['label'])
        MyDict[c1]['label'] = '{0}'.format(MyDict[c1]['label'])

        ax.text(*center, MyDict[c1]['label'])  # plot label to new cluster

    if UseSortedLabel:
        ax.set_xticks(np.arange(len(m)), [str(mi) for mi in m])
        plt.xticks(np.arange(len(m)), [str(mi) for mi in m])
    # ax.set_yticks([], [])
    ax.set_title(title)
    if ps2_test:
        plt.show()

# Assignment 4
def norm_pdf(X, mu, C):
    """ Computes probability density function for multivariate gaussian
    Input:
    X: (n x d) data matrix with each datapoint
    mu: vector for center
    C: covariance matrix
    Output:
    pdf value for each data point
    """

    d = np.shape(mu)[0]
    if np.linalg.det(C) == 0:
        #print('Matrix is not invertible =>', np.linalg.det(C))
        C = C + (np.eye(np.shape(C)[0])*0.05)
        #print('Matrix det after  =>', np.linalg.det(C))

    C_det = np.linalg.det(C)
    C_inv = np.linalg.inv(C)

    down = np.sqrt((2 * np.pi) ** d * C_det)  # , #down = ((2 * np.pi) ** d/2) * np.sqrt(C_det)
    up = -np.einsum('...i,ij,...j->...', X - mu, C_inv, X - mu) / 2  # einsum (x-mu)T.Sigma-1.(x-mu)
    #t = X-mu
    #print('t {}, C:{}'.format(np.shape(t), np.shape(C)))
    #s = np.linalg.solve(C,t)
    #print('s ', np.shape(s))
    pdf = np.exp(up) / down

    import scipy.sparse.linalg as spln
    import scipy.sparse as sp
    def lognormpdf(x, mu, S):
        """ Calculate gaussian probability density of x, when x ~ N(mu,sigma) """
        nx = len(S)
        norm_coeff = nx * np.math.log(2 * np.math.pi) + np.linalg.slogdet(S)[1]

        err = (x - mu).T
        print('S shape is:{}, err:{}'.format(np.shape(S), np.shape(err)))
        if (sp.issparse(S)):
            numerator = spln.spsolve(S, err).T.dot(err)
        else:
            numerator = np.linalg.solve(S, err).T.dot(err)

        return -0.5 * (norm_coeff + numerator)

    #pdf = (lognormpdf(X,mu,C))

    return pdf

# Assignment 5
def em_gmm(X, k, max_iter=100, init_kmeans=False, eps=1e-3, Iterations = False):
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
        print('Iter:{},  log-likelihood:{}, diff:{}'.format(iter,logLik,abs(logLik - prev_logLik)))
    print('Finished at {} iter, Log-likelihood:{}'.format(iter,logLik))
    if Iterations:
        return mpi, mu, sigma, logLik, iter
    return mpi, mu, sigma, logLik

# Assignment 6
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
    # ax.scatter(mu[:, 0].A1, mu[:, 1].A1, c='r', s=150, marker='x',lw=2)
    ax.scatter(np.ravel(mu[:, 0]), np.ravel(mu[:, 1]), c='r', s=150, marker='x', lw=2)
    t = np.linspace(0, 2 * np.pi, 100)
    for i in range(np.shape(mu)[0]):
        u = mu[i, 0]  # x-position center
        v = mu[i, 1]  # y-position center

        p = .9
        s = -2 * np.log(1 - p)
        #print(sigma)
        D, V = np.linalg.eig(sigma[i] * s)
        a = (V * np.sqrt(D)) @ [np.cos(t), np.sin(t)]
        ax.plot(a[0, :] + u, a[1, :] + v, c='g', lw=2)

    ax.set_title(title)
    ax.grid(color='lightgray', linestyle='--')
    custom_lines = [Line2D([0], [0], color='tab:blue', lw=1, marker='o'),
                    Line2D([0], [0], color='g', lw=4),
                    Line2D([0], [0], color='r', lw=1, marker='x')]
    ax.legend(custom_lines, ['Data points', 'GMM Covariance', 'Mean vectors'])
    if ps2_test:
        plt.show()

def Assignment7():
    # 1. Load the data set.
    _5gaussians = np.load('data/5_gaussians.npy')
    X = _5gaussians.T  # [:40,:]
    print(np.shape(X))

    # 2)Analyse with K-means & kmeans_agglo
    fig, ax = plt.subplots(figsize=(18, 14))
    kmloss_, mergeidx_ = [], []
    for i in range(2, 8):
        mu, r, loss = kmeans(X, k=i)
        ax = fig.add_subplot(3, 2, i - 1)
        plt.scatter(X[:, 0], X[:, 1], s=30, c=r, cmap=plt.cm.nipy_spectral)
        plt.title('K-means with k:{}'.format(i))
        plt.xticks([], [])
        plt.yticks([], [])
        plt.scatter(mu[:, 0], mu[:, 1], marker='*', c='r', zorder=2, s=200, lw=3, cmap=plt.cm.nipy_spectral)
        R, kmloss, mergeidx = kmeans_agglo(X, r)
        kmloss_.append(kmloss)
        mergeidx_.append(mergeidx)

    # kmeans_agglo plot here
    fig1, axe1 = plt.subplots(figsize=(18, 14))
    for i, (los, idx) in enumerate(zip(kmloss_, mergeidx_)):
        ax1 = fig1.add_subplot(3, 2, i + 1)
        agglo_dendro(kmloss=los, mergeidx=idx, verbose=False, title='Dendro from k={}'.format(i + 2), ax=ax1)
    plt.show()

    # 3)Analyse with GMM
    fig1, axe1 = plt.subplots(figsize=(18, 14))
    fig2, axe2 = plt.subplots(figsize=(18, 14))
    IterationGMM, IterationsGMM_kmeans, LogGMM, LogGMM_kmeans = [], [], [], []
    for i in range(2, 8):
        mpi, mu, sigma, logLik0, Iterations0 = em_gmm(X, k=i, Iterations=True)
        IterationGMM.append(Iterations0)
        LogGMM.append(logLik0)
        ax1 = fig1.add_subplot(3, 2, i - 1)
        plot_gmm_solution(X, mu, sigma, 'EM_GMM with k=' + str(i), ax=ax1)
        mpi, mu, sigma, logLik_kmeans, Iterations_kmeans = em_gmm(X, k=i, init_kmeans=True, Iterations=True)
        IterationsGMM_kmeans.append(Iterations_kmeans)
        LogGMM_kmeans.append(logLik_kmeans)
        ax2 = fig2.add_subplot(3, 2, i - 1)
        plot_gmm_solution(X, mu, sigma, 'EM_GMM with k=' + str(i) + ' init with k-means', ax=ax2)
    plt.show()

    plt.plot(range(2, 8), IterationGMM, label='Iterations')
    plt.plot(range(2, 8), IterationsGMM_kmeans, label='Iterations k-means')
    plt.plot(range(2, 8), LogGMM, label='LogGMM')
    plt.plot(range(2, 8), LogGMM_kmeans, label='LogGMM with kmeans')
    plt.legend()
    plt.xticks(range(2, 8), [str(mi) for mi in range(2, 8)])
    plt.xlabel('K')
    plt.show()

def Assignment8():
    # 1. Load the data set.
    _2gaussians = np.load('data/2_gaussians.npy')
    print(np.shape(_2gaussians))
    X = _2gaussians.T

    # 2)Analyse with K-means & kmeans_agglo
    fig, ax = plt.subplots(figsize=(18, 14))
    kmloss_, mergeidx_ = [], []
    for i in range(2, 4):
        mu, r, loss = kmeans(X, k=i)
        ax = fig.add_subplot(2, 1, i - 1)
        plt.scatter(X[:, 0], X[:, 1], s=30, c=r)
        plt.title('K-means with k:{}'.format(i))
        plt.xticks([], [])
        plt.yticks([], [])
        plt.scatter(mu[:, 0], mu[:, 1], marker='*', c='r', zorder=2, s=200, lw=3, cmap=plt.cm.nipy_spectral)
        R, kmloss, mergeidx = kmeans_agglo(X, r)
        kmloss_.append(kmloss)
        mergeidx_.append(mergeidx)

    # kmeans_agglo plot here
    fig1, axe1 = plt.subplots(figsize=(18, 14))
    for i, (los, idx) in enumerate(zip(kmloss_, mergeidx_)):
        ax1 = fig1.add_subplot(2, 1, i + 1)
        agglo_dendro(kmloss=los, mergeidx=idx, verbose=False, title='Dendro from k={}'.format(i + 2), ax=ax1)
    plt.show()

    # 3)Analyse with GMM
    fig1, axe1 = plt.subplots(figsize=(18, 14))
    fig2, axe2 = plt.subplots(figsize=(18, 14))
    IterationGMM, IterationsGMM_kmeans, LogGMM, LogGMM_kmeans = [], [], [], []
    for i in range(2, 4):
        mpi, mu, sigma, logLik0, Iterations0 = em_gmm(X, k=i, Iterations=True)
        IterationGMM.append(Iterations0)
        LogGMM.append(logLik0)
        ax1 = fig1.add_subplot(2, 1, i - 1)
        plot_gmm_solution(X, mu, sigma, 'EM_GMM with k=' + str(i), ax=ax1)
        mpi, mu, sigma, logLik_kmeans, Iterations_kmeans = em_gmm(X, k=i, init_kmeans=True, Iterations=True)
        IterationsGMM_kmeans.append(Iterations_kmeans)
        LogGMM_kmeans.append(logLik_kmeans)
        ax2 = fig2.add_subplot(2, 1, i - 1)
        plot_gmm_solution(X, mu, sigma, 'EM_GMM with k=' + str(i) + ' init with k-means', ax=ax2)
    plt.show()

def Assignment9():
    # 1. Load the usps data set.
    mat = spio.loadmat('data/usps.mat')
    X = mat['data_patterns'].T
    print('X:{}'.format(np.shape(X)))

    #plot some points from dataset
    fig, ax = plt.subplots(7, 7, figsize=(8, 8), subplot_kw=dict(xticks=[], yticks=[]))
    for i, axi in enumerate(ax.flat):
        im = axi.imshow(X[i].reshape(16, 16), cmap='binary')
    #plt.suptitle('Original dataset')
    plt.show()

    #K-means
    kmeans_mean_plot = None
    old_loss = np.inf
    assignments = None
    k=10
    for i in range(10):
        mu, r, loss = kmeans(X, k=k)
        print('K-means loss:{}'.format(loss))
        if loss < old_loss:
            kmeans_mean_plot = mu
            old_loss = loss
            assignments = r
    print('Kmean_loss: ',old_loss)
    for i in range(10):
        plt.subplot(2, 5, i+1)
        plt.imshow(kmeans_mean_plot[i,:].reshape(16, 16), cmap=plt.cm.Greys)
        plt.axis('off')
    plt.suptitle('K-means centroids')
    plt.show()

    #GMM k=10 with random init & GMM with kmeans init
    GMM_means_random = None
    GMM_log = -np.inf
    GMM_means_kmean = None
    GMM_log_kmean = -np.inf
    for i in range(5):
        mpi, mu, sigma, logLik_random = em_gmm(X, k=10, eps=1e-1, init_kmeans=False)
        print('Step {}, logLik_random:{}'.format(i, logLik_random))
        if logLik_random > GMM_log:
            GMM_log = logLik_random
            GMM_means_random = mu
        mpi, mu, sigma, logLik_kmeans = em_gmm(X, k=10,eps=1e-1, init_kmeans=True)
        print('Step {}, logLik_kmeans:{}'.format(i, logLik_kmeans))
        if logLik_kmeans > GMM_log_kmean:
            GMM_log_kmean = logLik_kmeans
            GMM_means_kmean = mu

    fig, ax = plt.subplots(2, 5, figsize=(8, 8), subplot_kw=dict(xticks=[], yticks=[]))
    for i, axi in enumerate(ax.flat):
        im = axi.imshow(GMM_means_random[i].reshape(16, 16), cmap='binary')
    plt.suptitle('GMM centroids with random init')
    plt.show()

    fig, ax = plt.subplots(2, 5, figsize=(8, 8), subplot_kw=dict(xticks=[], yticks=[]))
    for i, axi in enumerate(ax.flat):
        im = axi.imshow(GMM_means_kmean[i].reshape(16, 16), cmap='binary')
    plt.suptitle('GMM centroids with kmeans init')
    plt.show()

    #Dendro plot
    R, kmloss, mergeidx = kmeans_agglo(X, assignments)
    agglo_dendro(kmloss=kmloss, mergeidx=mergeidx, verbose=False, title='Dendro')
    #plot the cluster centroidsas a 16 × 16 image at every agglomerative step
    fig, ax = plt.subplots(9, 4, figsize=(8, 16), subplot_kw=dict(xticks=[], yticks=[]))
    for i in range(len(kmloss)-1):
        r = R[i]
        first = np.mean(X[r == mergeidx[i][0]], axis=0) #first cluster
        ax[i,1].imshow(first.reshape(16, 16), cmap=plt.cm.Greys)
        ax[i,1].set_title('First cluster--->')

        second = np.mean(X[r == mergeidx[i][1]], axis=0) #second cluster
        ax[i,3].imshow(second.reshape(16, 16), cmap=plt.cm.Greys)
        ax[i, 3].set_title('<---Second cluster')

        result = first + second #resulted cluster
        ax[i, 2].imshow(result.reshape(16, 16), cmap=plt.cm.Greys)
        ax[i, 2].set_title('>Resulted cluster<')

        ax[i, 0].text(0.25, 0.4, 'Step:{}'.format(i+1), dict(size=20))

    plt.show()

def Assignment10():
    def gammaidx(X, k):
        def euclid_dist(x1, x2):
            return np.sqrt(sum((x1 - x2) ** 2))

        dist_matrix = cdist(X, X, 'euclidean')
        near_points = np.argpartition(dist_matrix, k + 1)
        y = []
        for i in range(X.shape[0]):
            y.append(np.sum([euclid_dist(X[i, :], X[j, :]) for j in near_points[i, :k + 1]]))
        y = (1 / k) * np.array(y)

        return y

    def auc(y_true, y_pred, plot=False):
        fpr, tpr = [], []
        th = np.arange(0.0, 1.1, .01)  # thresholds
        P, N = 0, 0  # positive, negative
        for cls in y_true:
            if cls > 0:
                P += 1
        N = len(y_true) - P
        assert N > 0 and P > 0, 'N or P is zero, zero division exception (No positive or negative classes, Inbalanced data)'

        for thresh in th:
            FP, TP = 0, 0
            for i in range(len(y_pred)):
                if (y_pred[i] > thresh):
                    if y_true[i] == 1:
                        TP += 1
                    if y_true[i] == -1:
                        FP += 1
            fpr.append(FP / float(N))
            tpr.append(TP / float(P))

        c = np.abs(np.trapz(tpr, fpr))  # trapezoidal rule

        if plot:
            plt.plot(fpr, tpr, label="ROC, AUC score:" + str(c))
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.legend()
            plt.show()

        return c, tpr, fpr

    # 1. Load the data set.
    dataset = np.load('data/lab_data.npz')
    print(dataset.files)
    X = dataset['X']
    Y = dataset['Y']
    print('X:{},  Y:{}'.format(np.shape(X), np.shape(Y)))
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1, projection='3d', adjustable='box')
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=Y, s=20, edgecolor='k')

    # 2) em_gmm
    mu_best = None
    sigma_best = None
    logLik_best=-np.inf
    mpi_best = None
    for k in range(3):
        mpi, mu, sigma, logLik = em_gmm(X, k=3, init_kmeans=True)
        if logLik>logLik_best:
            logLik_best = logLik
            mu_best = mu
            sigma_best = sigma
            mpi_best = mpi

    fig = plt.figure(figsize=(8, 8))
    axes = fig.add_subplot(111, projection='3d')
    axes.scatter(X[:, 0], X[:, 1], X[:, 2], c=Y, s=20, edgecolor='k',alpha=0.1)
    axes.set_xlabel('X')
    axes.set_ylabel('Y')
    axes.set_zlabel('Z')

    def plotGMM_Means_And_Cov(center, Cov, ax, color):
        _, sig, rotation = np.linalg.svd(Cov)
        sca = 1.0 / np.sqrt(sig)
        phi = np.linspace(0.0, 2.0 * np.pi, 100)
        theta = np.linspace(0.0, np.pi, 100)
        x = sca[0] * np.outer(np.cos(phi), np.sin(theta))
        y = sca[1] * np.outer(np.sin(phi), np.sin(theta))
        z = sca[2] * np.outer(np.ones_like(phi), np.cos(theta))
        for i in range(len(x)):
            for j in range(len(x)):
                [x[i, j], y[i, j], z[i, j]] = np.dot([x[i, j], y[i, j], z[i, j]], rotation) + center
        ax.plot_wireframe(x, y, z, rstride=3, cstride=3, color=color, alpha=0.3)

    colors = ['b','y','g']
    for j in range(0,3):
       m = mu_best[j].T
       axes.scatter(m[0], m[1], m[2], c=colors[j], marker='o', s=200, edgecolor='k', alpha=1.0)
       plotGMM_Means_And_Cov(np.squeeze([i for i in m]), sigma_best[j], axes, colors[j])

    # 3) gammaidx
    y_prim = [-i for i in Y]
    for i in [3,6,9]:  # neighbors for gamma
        gamma = gammaidx(X, k=i)
        c, _, _ = auc(y_true=y_prim, y_pred=gamma)
        print('C:{},  k:{}'.format(c,i))
    
    # custom tailored outlier detection
    print('Best logLike: ',logLik_best)
    px=0
    for j in range(3):
        px += norm_pdf(X, np.ravel(mu_best[j, :]), sigma_best[j, :]) * mpi_best[j]
    print('px: ', np.shape(px))

    # 4)
    fig, axs = plt.subplots()
    gamma = gammaidx(X, k=9)
    c, tpr, fpr = auc(y_true=y_prim, y_pred=gamma)
    plt.plot(fpr, tpr, label="Gamma, ROC, AUC score:" + str(c))
    c, tpr, fpr = auc(y_true=Y, y_pred=px)
    plt.plot(fpr, tpr, label="Custom outlier Px, ROC, AUC score:" + str(c))
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()

    plt.show()

if __name__ == '__main__':
    print('Main')
    #Assignment7()
    #Assignment8()
    #Assignment9()
    Assignment10()

