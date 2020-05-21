
from __future__ import division  # always use float division
import numpy as np
from scipy.spatial.distance import cdist  # fast distance matrices
from scipy.cluster.hierarchy import dendrogram  # you can use this
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D  # for when you create your own dendrogram

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
    loss = np.linalg.norm(mu - mu_old)

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

        #loss = np.linalg.norm(mu - mu_old)
        loss = distances.sum()
        #print('Iterations: {},  Loss:{}'.format(j, loss))

    if plot:
        for i in range(n):
            plt.scatter(X[i, 0], X[i, 1], s=100, c='b')
        plt.scatter(mu[:, 0], mu[:, 1], marker='*', c='g', s=150)
        plt.show()

    #print('mu: ', np.shape(mu))
    #print('r: ',r)
    #print('loss  = {}'.format(loss))

    return mu, r, loss

#Assignment 2
def kmeans_agglo(X, r):
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
    plt.figure(figsize=(6, 6))
    plt.scatter(X[:, 0], X[:, 1],)
    for label, x, y in zip(labels, X[:, 0], X[:, 1]):
        plt.annotate(
            label,
            xy=(x, y), xytext=(-3, 3),
            textcoords='offset points', ha='right', va='bottom')
    plt.show()

    #r = [i for i in range(X.shape[0])]
    print('r ',r)
    _k = len(np.unique(r))  # number of different values  i.e. clusters
    print('There are {} clusters'.format(_k))
    centroids = np.zeros((_k, d))
    r = np.array(r)
    R, kmloss, mergeidx = [], [], []
    #R.append(r)
    for i in range(_k):
        #centroids[i] = np.mean(X[i], axis=0)
        centroids[i] = np.mean(X[r == i], axis=0)
    print('centroids ', np.shape(centroids))
    kmloss.append(cdist(X, centroids, 'euclidean').sum())
    m = len(centroids)
    print('m ', m)
    while (m > 1):
        print('Before clustering  ', m)
        print('centroids ', np.shape(centroids))
        distance_matrix = cdist(centroids, centroids, 'euclidean')
        print('distance_matrix ', np.shape(distance_matrix))
        np.fill_diagonal(distance_matrix, np.inf)
        #print(np.diag(distance_matrix))
        #print('Min value = ',distance_matrix.min())
        min_idx = np.where(distance_matrix == distance_matrix.min())[0] #x index where min value
        print('min_idx ', np.shape(min_idx))
        print(min_idx)
        u = np.unique(r)
        for i in range(int(len(min_idx)/2)):
            #i=0
            val1 = u[min_idx[int(2*i)]]
            val2 = u[min_idx[int(2 * i+1)]]
            r[np.where(r == val2)] = val1
            #mergeidx.append([min_idx[int(2*i)],min_idx[int(2 * i+1)]])
            mergeidx.append([val1, val2])
            print('mergeidx ', np.shape(mergeidx))
            kmloss.append(cdist(X, centroids, 'euclidean').sum())
            print('r ' ,r)
            R.append(r)
            print('R {}-'.format(np.shape(R)))
        _k = len(np.unique(r))  # number of different values  i.e. clusters
        print('There are {} clusters'.format(_k))
        centroids = np.zeros((_k, d))
        print('unique ',np.unique(r))
        j=0
        for i in np.unique(r):
            centroids[j] = np.mean(X[r == i], axis=0)
            j+=1
        print('centroids ', np.shape(centroids))
        m = len(centroids)

        print('Current r: ', r)
        print('Sample size ', m)
        print('\n')

    import scipy.cluster.hierarchy as shc
    plt.figure(figsize=(8, 8))
    plt.title('Visualising the data')
    Z = shc.linkage(X, method='ward')
    print('Z shape is ', np.shape(Z))
    Dendrogram = dendrogram((Z))
    plt.show()

    print('kmloss now ',np.shape(kmloss))
    return R, kmloss, mergeidx

#Assignment 3
def agglo_dendro(kmloss, mergeidx):
    """ Plots dendrogram for agglomerative clustering
    Input:
    kmloss: vector with loss after each step
    mergeidx: (k-1) x 2 matrix that contains merge idx for each step
    """
    print('----------------------------------------------------------')
    print('kmloss ', np.shape(kmloss))
    print('mergeidx ', np.shape(mergeidx))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i in range(len(mergeidx)):
        ax.scatter(i,0, c='k')
        kmloss[i]*=0.1

    kmloss = kmloss[::-1]
    print(kmloss)
    prevY1=prevY2=0
    tmp=[]
    mid1=mid2 = 0
    for i in range(3): #len(mergeidx)
        d = mergeidx[i]
        print(d)
        if d[0] in tmp:
            mid1 = (mergeidx[i][0] + mergeidx[i][1]) / 2.0
            mid2 = (mergeidx[i + 1][0] + mergeidx[i + 1][1]) / 2.0
            x = [mid1, mid1,mid2,mid2]
            print('Inside mid1 :{},  mid2: {}'.format(mid1,mid2))
        else:
            x = [d[0], d[0], d[1], d[1]]
            prevY2 = kmloss[i+1]
            prevY1 = 0



        y = [prevY1, prevY2,prevY2,prevY1]
        tmp.append(d[0])
        tmp.append(d[1])

        line = Line2D(x, y, c='r')
        ax.add_line(line)

    #print()
    ax.set_ylim(-5, 500)
    #ax.set_ylim(-10, max(kmloss))
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
    C_det = np.linalg.det(C)
    #print(C)
    assert C_det != 0, "Determinant is zero -> Matrix is not invertible "
    #C_inv = np.linalg.inv(C)
    C_inv = np.linalg.pinv(C)
    #C_inv = np.linalg.inv(C @ C.T + 1e-2 * np.eye(np.shape(C)[0])) #add regularization term because is not invertible

    down = np.sqrt((2 * np.pi) ** d * C_det)
    #down = ((2 * np.pi) ** d/2) * np.sqrt(C_det)

    #  einsum (x-mu)T.Sigma-1.(x-mu)
    up = -np.einsum('...i,ij,...j->...', X - mu, C_inv, X - mu)/2

    if down == 0 or C_det==0:
        print('C_det is {} and down is {}'.format(C_det, down))
    pdf = np.exp(up) / down

    #from scipy.stats import multivariate_normal
    #C = C + 1e-2 * np.eye(np.shape(C)[0])
    #pdf = multivariate_normal.pdf(X, mean=mu, cov=C)

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
def plot_gmm_solution(X, mu, sigma):
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

    plt.grid(color='lightgray', linestyle='--')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    #norm_pdf(0,0,0)
    print('main')

