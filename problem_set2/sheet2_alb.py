""" ps2_implementation.py

PUT YOUR NAME HERE:
<Alberto_Gonzalo><Rodriguez_Salgado>

Write the functions
- kmeans
- kmeans_agglo
- agglo_dendro
- norm_pdf
- em_gmm
- plot_gmm_solution

(c) Felix Brockherde, TU Berlin, 2013
    Translated to Python from Paul Buenau's Matlab scripts
"""

from __future__ import division  # always use float division
import numpy as np
from scipy.spatial.distance import cdist  # fast distance matrices
from scipy.cluster.hierarchy import dendrogram  # you can use this
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D  # for when you create your own dendrogram
import random
from itertools import combinations_with_replacement

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

    #Randomly initialize centroids as data points
    random_indexes = random.sample(range(n), k)
    mu = X[random_indexes, :]

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
########################################################################################################################
def new_centroids_agglo(X, r, k):
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

    for indx, i in enumerate(np.unique(r)):
        # In this case the cluster is not empty and we can compute the new centroid as usual
        mu[indx] = np.mean(X[r == i], axis=0)

    return mu


def kmeans_agglo(X, r):
    """ Performs agglomerative clustering with k-means criterion

    Input:
    X: (n x d) data matrix with each datapoint in one column
    r: assignment vector

    Output:
    R: (k-1) x n matrix that contains cluster memberships before each step
    kmloss: vector with loss after each step
    mergeidx: (k-2) x 2 matrix that contains merge idx for each step
    """
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
        loss = 0
        for j in range(n):
            loss += distances[j, np.argwhere(clusters == r[j])]**2

        return loss

    # Compute clusters set
    c = compute_C_set(r)
    k = len(c)
    n, d = X.shape
    # Init return arrays
    R = np.zeros(((k), n))
    kmloss = np.zeros(k)
    mergeidx = np.zeros((k-1, 2), dtype='int')
    # Compute centroids
    mu = new_centroids_agglo(X=X, r=r, k=k)

    # Compute L loss value for the initial clustering
    kmloss[0] = kmeans_crit(X=X, r=r, mu=mu)
    R[0, :] = r

    new_cluster_idx = n

    for j in range(k-1):
        # compute all possible 2 cluster merge combinations
        clust_comb = list(combinations_with_replacement(c, 2))

        #  Now we need to compute the loss value for every merge combination and take the max one
        loss_local = np.zeros(len(clust_comb))
        for index, merge_pair in enumerate(clust_comb):
            # Avoid (0,0), (1,1) etc merges
            if merge_pair[0] == merge_pair[1]:
                loss_local[index] = np.inf
            else:
                # Generate new merged cluster
                r_merged = r.copy()
                r_merged[np.argwhere(r == merge_pair[0])] = merge_pair[1]
                # Compute new centroids for the merged cluster
                mu_new = new_centroids_agglo(X=X, r=r_merged, k=(len(c)-1))
                # Compute loss
                loss_local[index] = kmeans_crit(X=X, r=r_merged, mu=mu_new)

        # Compute min L of all merge options
        L_min = np.min(loss_local)
        kmloss[j+1] = L_min
        # Get merge clusters min
        min_merge_clusters = clust_comb[int(np.argwhere(loss_local == L_min)[0])]
        r_min = r
        r_min[np.argwhere(r == min_merge_clusters[0])] = new_cluster_idx
        r_min[np.argwhere(r == min_merge_clusters[1])] = new_cluster_idx
        R[j+1, :] = r_min
        mergeidx[j, :] = np.asarray(min_merge_clusters)

        # Update
        r = r_min
        c = compute_C_set(r=r)

        new_cluster_idx += 1
    return R, kmloss, mergeidx

def compute_C_set(r):
    """  Computes the set of clusters

    Input:
    r: 1D array, assignment vector

    Output:
    c : 1D array, contains all possible clusters ordered
    """
    c = np.unique(r)
    return c


def agglo_dendro(kmloss, mergeidx):
    """ Plots dendrogram for agglomerative clustering

    Input:
    kmloss: vector with loss after each step
    mergeidx: (k-1) x 2 matrix that contains merge idx for each step
    """

    # Build Z for the dendogram
    Z = np.zeros((mergeidx.shape[0], 4))

    Z[:, 3] = 1
    Z[:, :2] = mergeidx
    Z[:, 2] = kmloss[1:]

    print(Z)
    plt.title('Hierarchical Clustering Dendrogram (truncated)')
    plt.xlabel('sample index or (cluster size)')
    plt.ylabel('distance')
    dendrogram(Z)
    plt.show()
#X = np.array([[0., 1., 1., 10., 10.25, 11., 10., 10.25, 11.], [0., 0., 1.,  0.,   0.5,  0.,  5.,   5.5,  5.]]).T
X = np.zeros((20, 2))

X[:10, :]     = 0.1*np.random.randn(10, 2) + np.array([1, 1])
X[10:15, :] = 0.1*np.random.randn(5, 2) + np.array([2, 1])
X[15:, :]    = 0.1*np.random.randn(5, 2) + np.array([1.5, 2])

#plt.scatter(X[:,0], X[:, 1])

k = 20
mu, r, loss = kmeans(X=X, k=k, max_iter=100)


R, kmloss, merge = kmeans_agglo(X=X, r=r)

agglo_dendro(mergeidx=merge, kmloss=kmloss)
