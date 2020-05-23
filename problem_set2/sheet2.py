
from __future__ import division  # always use float division
import numpy as np
from scipy.spatial.distance import cdist  # fast distance matrices
from scipy.cluster.hierarchy import dendrogram  # you can use this
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D  # for when you create your own dendrogram
import scipy.io as spio          #used to lod .mat dataset

#np.random.seed(2020)

#Assignment 1
def kmeans(X, k, max_iter=100):
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
    mu = np.zeros((k,d))
    mu[:k,:] = X[:k,:] #initial centroids from data

    mu_old = np.zeros(mu.shape)  # store old centroids
    r = np.zeros(n)
    r_prim = np.zeros(n)
    loss = cdist(X, mu, 'euclidean').sum() # np.linalg.norm(mu - mu_old)

    plot = False
    for j in range(max_iter):
        #Compute distance to every centroid
        distances = cdist(X, mu, 'euclidean')

        # Assign data to closest centroid
        r_prim = r.copy()
        r = np.argmin(distances, axis=1)
        if (r == r_prim).all():
            break

        # Compute new cluster center
        mu_old = mu.copy()
        for i in range(k):
            mu[i] = np.mean(X[r == i], axis=0)

        loss = distances.sum() #for sum of euclidean distances to cluster centers
        #print('Iterations: {},  Loss:{}'.format(j, loss))

    if plot:
        for i in range(n):
            plt.scatter(X[i, 0], X[i, 1], s=100, c='b')
        plt.scatter(mu[:, 0], mu[:, 1], marker='*', c='g', s=150)
        plt.show()

    return mu, r, loss

#Assignment 2
def kmeans_agglo(X, r, verbose = False):
    """ Performs agglomerative clustering with k-means criterion
    Input:
    X: (n x d) data matrix with each datapoint in one column
    r: assignment vector
    Output:
    R: (k-1) x n matrix that contains cluster memberships before each step
    kmloss: vector with loss after each step
    mergeidx: (k-1) x 2 matrix that contains merge idx for each step
    """
    def kmeans_crit(X, r):
        """ Computes k-means criterion
        Input:
        X: (n x d) data
        r: assignment vector
        Output:
        value: scalar for sum of euclidean distances to cluster centers
        """
        local_k= len(np.unique(r)) #number of different values  i.e. clusters
        print('There are {} clusters'.format(local_k))
        local_mu = np.zeros((local_k, d))
        for i in range(local_k):
            local_mu[i] = np.mean(X[r == i], axis=0)
        return cdist(X, local_mu, 'euclidean').sum(-1)

    n, d = np.shape(X)
    labels = range(n)
    #plotting the data with labels
    if verbose:
        plt.figure(figsize=(6, 6))
        plt.scatter(X[:, 0], X[:, 1],)
        for label, x, y in zip(labels, X[:, 0], X[:, 1]):
            plt.annotate(label,xy=(x, y), xytext=(-3, 3),textcoords='offset points', ha='right', va='bottom')
        plt.show()

    if verbose:
        r = [i for i in range(X.shape[0])] #this is just for test, comment this line, to get the r from parameters
        print('r ',r)
    _k = len(np.unique(r))  # number of different values  i.e. clusters
    if verbose:
        print('There are {} clusters'.format(_k))
    centroids = np.zeros((_k, d))
    r = np.array(r)
    R, kmloss, mergeidx = [], [], []
    R.append(r)
    for i in range(_k):
        centroids[i] = np.mean(X[r == i], axis=0)
    if verbose:
        print('centroids ', np.shape(centroids))
    kmloss.append(cdist(X, centroids, 'euclidean').sum())
    m = len(centroids)
    while (m > 1):
        if verbose:
            print('Before clustering  {} clusters'.format(m))
            print('Centroids: ', np.shape(centroids))
        distance_matrix = cdist(centroids, centroids, 'euclidean')
        #print('distance_matrix ', np.shape(distance_matrix))
        np.fill_diagonal(distance_matrix, np.inf) #fill diagonal with infinity, in order to get the min value (before zero filling, diagonal is zero)
        min_idx = np.where(distance_matrix == distance_matrix.min())[0] #x index where min value
        if verbose:
            print('min_idx ', np.shape(min_idx))
            print('Indices that correspond to min value ',min_idx)
        u = np.unique(r)
        #for i in range(int(len(min_idx)/2)):
        i=0
        val1 = u[min_idx[int(2*i)]]
        val2 = u[min_idx[int(2 * i+1)]]
        r[np.where(r == val2)] = val1
        mergeidx.append([val1, val2]) #save merged indexes
        R.append(r) #save current assignment vector

        _k = len(np.unique(r))  # number of different values  i.e. clusters
        if verbose:
            print('There are {} clusters'.format(_k))
            print('unique ',np.unique(r))
        centroids = np.zeros((_k, d))

        j=0
        for i in np.unique(r):
            centroids[j] = np.mean(X[r == i], axis=0)
            j+=1
        if verbose:
            print('Centroids: ', np.shape(centroids))
        kmloss.append(cdist(X, centroids, 'euclidean').sum())  # save distance loss
        m = len(centroids)
        if verbose:
            print('Current r: ', r)
            print('Current m: ', m)
            print('\n')

    #this is just to check the scipy result
    '''import scipy.cluster.hierarchy as shc
    plt.figure(figsize=(8, 8))
    plt.title('Visualising the data')
    Z = shc.linkage(X, method='ward')
    print('Z shape is ', np.shape(Z))
    Dendrogram = dendrogram((Z))
    plt.show()'''

    if verbose:
        print('R  ', np.shape(R))
        print('kmloss  ',np.shape(kmloss))
        print('mergeidx  ', np.shape(mergeidx))

    return R, kmloss, mergeidx

#Assignment 3
def agglo_dendro(kmloss, mergeidx, verbose = True, title='', ax=None):
    """ Plots dendrogram for agglomerative clustering
    Input:
    kmloss: vector with loss after each step
    mergeidx: (k-1) x 2 matrix that contains merge idx for each step
    """
    kmloss = kmloss[::-1]
    if verbose:
        print(kmloss)
        print('level loses', np.shape(kmloss))
        print('merged indexes \n',mergeidx)

    def helper(x0,x1,y0,y1,next_loss):
        return ([[x0,x0,x1,x1],[y0,next_loss,next_loss,y1]]), [(x0+x1)/2.0,next_loss]

    MyDict = {}
    keys = range(len(kmloss))
    if ax==None:
        fig, ax = plt.subplots()
    for i in keys:
        plt.scatter(i, 0, c='k')
        MyDict[i] = dict({'label': str(i), 'x': i, 'y': 0})
    #print(MyDict)
    for i,(next_level,(c1,c2)) in enumerate(zip(kmloss,mergeidx)):
        x0, y0 = MyDict[c1]['x'], MyDict[c1]['y']
        x1, y1 = MyDict[c2]['x'], MyDict[c2]['y']

        if verbose:
            print('Iteration: {}, Clusters: ({},{}) current level loss:{}'.format(i,c1,c2,next_level))
            print('Merged clusters: c{}:({},{}) & c{}:({},{})'.format(c1,x0,y0,c2,x1,y1))

        line, center = helper(x0, x1, y0, y1, next_level)
        ax.plot(*line)
        ax.scatter(*center)

        if verbose:
            print('New centroid is {}, change the map[{}]:{}'.format(center,c1,MyDict[c1]))

        MyDict[c1]['x'] = center[0]
        MyDict[c1]['y'] = center[1]
        MyDict[c1]['label'] = '({0},{1})'.format(MyDict[c1]['label'], MyDict[c2]['label'])

        if verbose:
            ax.text(*center, MyDict[c1]['label']) #plot label to new cluster

    ax.set_ylim(-10, 1.2 * np.max(kmloss))
    plt.title(title)
    plt.show()

#Assignment 4
def norm_pdf(X, mu, C):
    """ Computes probability density function for multivariate gaussian
    Input:
    X: (n x d) data matrix with each datapoint
    mu: vector for center
    C: covariance matrix
    Output:
    pdf value for each data point
    """

    d = np.shape(X)[0]
    C = C + 1e-2 * np.eye(np.shape(C)[0]) # this is added for regularization - remove later (ask tutor)

    C_det = np.linalg.det(C)
    #assert C_det != 0, "Determinant is zero -> Matrix is not invertible "
    #C_inv = np.linalg.inv(C)
    C_inv = np.linalg.pinv(C)
    #C_inv = np.linalg.inv(C @ C.T + 1e-2 * np.eye(np.shape(C)[0])) #add regularization term because is not invertible

    down = 1.0 #  np.sqrt((2 * np.pi) ** d * C_det) #use this ,   remove 1.0 later
    #down = ((2 * np.pi) ** d/2) * np.sqrt(C_det)

    #  einsum (x-mu)T.Sigma-1.(x-mu)
    up = 1.0 # -np.einsum('...i,ij,...j->...', X - mu, C_inv, X - mu)/2   #use this

    #if down == 0 or C_det==0:
    #    print('C_det is {} and down is {}'.format(C_det, down))
    pdf = np.exp(up) / down

    from scipy.stats import multivariate_normal  # this is just for test now, remove it later and use the code provided above
    pdf = multivariate_normal.pdf(X, mean=mu, cov=C)

    return pdf

#Assignment 5
def em_gmm(X, k, max_iter=100, init_kmeans=False, eps=1e-3):
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
        sigma = np.array([np.cov(X.T) for _ in range(k)])
    else:
        print('Init random ')
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

    return mpi, mu, sigma, logLik

#Assignment 6
def plot_gmm_solution(X, mu, sigma, title):
    """ Plots covariance ellipses for GMM
    Input:
    X: (n x d) data matrix
    mu: (k x d) matrix
    sigma: list of d x d covariance matrices
    """
    fig, ax = plt.subplots(figsize=(6, 6))

    ax.scatter(X[:, 0], X[:, 1], s=50)
    t = np.linspace(0, 2 * np.pi, 100)
    for i in range(np.shape(mu)[0]):
        u = mu[i,0]  # x-position center
        v = mu[i,1] # y-position center
        r = sigma[i][0, 1] / np.sqrt(sigma[i][0, 0] * sigma[i][1, 1])
        a = np.sqrt(1 + r)  # radius on the x-axis
        b = np.sqrt(1 - r)  # radius on the y-axis

        plt.arrow(u, v, .5, .5, head_width=0.05, head_length=0.1, fc='b', ec='b')
        #plt.plot(u + a * np.cos(t), v + b * np.sin(t), c='r')

        p= .9
        s = -2 * np.log(1 - p)
        D,V = np.linalg.eig(sigma[i] * s)
        a = (V * np.sqrt(D)) @ [np.cos(t), np.sin(t)]
        plt.plot(a[0,:] + u, a[1,:] + v, c='g') #the correct ones

        s1 = np.sqrt(D[0])
        s2 = np.sqrt(D[1])
        plt.arrow(u, v, s1 * V[0,0], s1 * V[1,0], head_width=0.05, head_length=0.01, alpha=1., color='r')
        plt.arrow(u, v, s2 * V[0,1], s2 * V[1,1], head_width=0.05, head_length=0.01, alpha=1., color='g')

    ax.set_title(title)
    plt.grid(color='lightgray', linestyle='--')
    plt.legend()
    plt.show()

def Assignment7():
    # 1. Load the data set.
    _5gaussians = np.load('data/5_gaussians.npy')
    print(np.shape(_5gaussians))
    X = _5gaussians.T

    #2)Analyse with K-means & kmeans_agglo
    fig, ax = plt.subplots(figsize=(18, 14))
    kmloss_,mergeidx_ = [],[]
    for i in range(2,8):
        mu, r, loss = kmeans(X, k=i)
        ax = fig.add_subplot(3, 2, i-1)
        plt.scatter(X[:, 0], X[:, 1], s=30, c=r, cmap=plt.cm.nipy_spectral)
        plt.title('K-means with k:{}'.format(i))
        plt.xticks([], [])
        plt.yticks([], [])
        plt.scatter(mu[:, 0], mu[:, 1], marker='*', c='r', zorder=2, s=200, cmap=plt.cm.nipy_spectral)
        R, kmloss, mergeidx = kmeans_agglo(X, r)
        kmloss_.append(kmloss)
        mergeidx_.append(mergeidx)

    #kmeans_agglo plot here
    for i,(los, idx) in enumerate(zip(kmloss_, mergeidx_)):
        agglo_dendro(kmloss=los, mergeidx=idx, verbose=False, title='Dendro from k={}'.format(i+2))
    plt.show()

    #3)Analyse with GMM
    for i in range(2,8):
        mpi, mu, sigma, _ = em_gmm(X, k=i)
        plot_gmm_solution(X, mu, sigma, 'EM_GMM with k='+str(i))
    plt.show()

def Assignment8():
    # 1. Load the data set.
    _2gaussians = np.load('data/2_gaussians.npy')
    print(np.shape(_2gaussians))
    X = _2gaussians.T

    # 2)Analyse with K-means & kmeans_agglo
    fig, ax = plt.subplots(figsize=(16, 12))
    kmloss_, mergeidx_ = [], []
    for i in range(1, 5):
        mu, r, loss = kmeans(X, k=i)
        ax = fig.add_subplot(2, 2, i)
        plt.scatter(X[:, 0], X[:, 1], s=30, c=r, cmap=plt.cm.nipy_spectral)
        plt.title('K-means with k:{}'.format(i))
        plt.xticks([], [])
        plt.yticks([], [])
        plt.scatter(mu[:, 0], mu[:, 1], marker='*', c='r', zorder=2, s=200, cmap=plt.cm.nipy_spectral)
        R, kmloss, mergeidx = kmeans_agglo(X, r)
        kmloss_.append(kmloss)
        mergeidx_.append(mergeidx)

    #kmeans_agglo plot here
    for i,(los, idx) in enumerate(zip(kmloss_, mergeidx_)):
        agglo_dendro(kmloss=los, mergeidx=idx, verbose=False, title='Dendro from k={}'.format(i+1))
    plt.show()

    # 3)Analyse with GMM
    for i in range(1, 5):
        mpi, mu, sigma, _ = em_gmm(X, k=i)
        plot_gmm_solution(X, mu, sigma, 'EM_GMM with k=' + str(i))
    plt.show()

def Assignment9():
    # 1. Load the usps data set.
    mat = spio.loadmat('data/usps.mat')
    X = mat['data_patterns'].T
    print('X:{}'.format(np.shape(X)))

    mu, r, loss = kmeans(X, k=10)
    print('K-means result, mu:{}, loss:{}'.format(np.shape(mu), loss))
    R, kmloss, mergeidx = kmeans_agglo(X, r)
    agglo_dendro(kmloss=kmloss, mergeidx=mergeidx, verbose=False, title='Dendro')
    print('mergeidx: ',np.shape(mergeidx))
    print('kmloss: ',np.shape(kmloss))
    keys = range(len(kmloss))
    fig, ax = plt.subplots()
    for i in keys:
        print(i)
        #given X and r -> get the centroids
        #reshape to 16
        #plot them as images on  5x2 figure

    #mpi, mu, sigma, _ = em_gmm(X, k=10)  #returns problem, matrix is singular -> not invertible
    #print('EM_GMM result, mpi:{}, mu:{}'.format(np.shape(mpi), np.shape(mu)))

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
    th = np.arange(0.0, 1.1, .1)  # thresholds
    P, N = 0, 0  # positive, negative
    for cls in y_true:
        if cls > 0:
            P += 1
    N = len(y_true) - P
    assert N>0 and P>0, 'N or P is zero, zero division exception (No positive or negative classes, Inbalanced data)'

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

    c = np.abs(np.trapz(tpr, fpr)) # trapezoidal rule

    if plot:
        plt.plot(fpr, tpr, label="ROC, AUC score:" + str(c))
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend()
        plt.show()

    return c, tpr, fpr

def Assignment10():
    # 1. Load the data set.
    dataset = np.load('data/lab_data.npz')
    print(dataset.files)
    X = dataset['X'] #[:20,:]
    Y = dataset['Y']#[:20,]
    print('X:{},  Y:{}'.format(np.shape(X), np.shape(Y)))
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1, projection='3d', adjustable='box')
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=Y, s=20,edgecolor='k')

    #2) em_gmm
    mpi, mu, sigma, logLik = em_gmm(X, k=3, init_kmeans=False) #problem  with init_kmeans=True
    print('EM_GMM result, mpi:{}, mu:{}, sigma:{}, logLik:{}'.format(np.shape(mpi), np.shape(mu),np.shape(sigma),logLik))
    ax.scatter(mu[:, 0], mu[:, 1], mu[:, 2], c='r', marker='o', s=200, edgecolor='k')

    #3) gammaidx
    from sklearn.metrics import roc_auc_score #just for tests
    #I used sklearn just to compare with my own implementation, delete it later
    AUC_history = [] #AUC own
    AUC_history2 = [] #AUC with sklearn
    for i in range(1,15): #neighbors for gamma
        print(i)
        gamma = gammaidx(X, k=i)
        c, tpr, fpr = auc(y_true=Y, y_pred=gamma)
        AUC_history.append(c)
        AUC_history2.append(roc_auc_score(Y, gamma))

    fig, axs = plt.subplots()
    plt.plot(range(1,15),AUC_history, label='AUC')
    axs.set_title('gammaidx & AUC ')
    axs.set_xlabel(r'$k$')
    axs.set_ylabel(r'$AUC$')
    plt.legend()

    fig, axs = plt.subplots()
    plt.plot(range(1, 15), AUC_history2, label='AUC 2')
    axs.set_title('gammaidx & AUC using sklearn')
    axs.set_xlabel(r'$k$')
    axs.set_ylabel(r'$AUC $')
    plt.legend()

    #4)
    fig, axs = plt.subplots()
    gamma = gammaidx(X, k=2)
    c, tpr, fpr = auc(y_true=Y, y_pred=gamma)
    plt.plot(fpr, tpr, label="ROC, AUC score:" + str(c))
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()

    #------------------------------------------


    plt.show()

if __name__ == '__main__':
    print('Main')
    #Assignment7()
    #Assignment8()
    #Assignment9()
    Assignment10()
