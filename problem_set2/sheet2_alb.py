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
        for i in range(k):
            if i in r:
                # In this case the cluster is not empty and we can compute the new centroid as usual
                mu[i] = np.mean(X[r == i], axis=0)
            else:
                # In this case the cluster is empty and we reinitialize the cluster at some random data point
                mu[i] = X[random.randrange(n), :]

        print('Iterations: {},  Loss:{}'.format(j, loss))


    if plot:
        for i in range(n):
            plt.scatter(X[i, 0], X[i, 1], s=100, c='b')
        plt.scatter(mu[:, 0], mu[:, 1], marker='*', c='g', s=150)
        plt.show()


    return mu, r, loss


''' little test

X = np.zeros((100, 2))

X[:25, :]     = 0.1*np.random.randn(25, 2) + np.array([1, 1])
X[25:50, :] = 0.1*np.random.randn(25, 2) + np.array([2, 1])
X[50:, :]    = 0.1*np.random.randn(50, 2) + np.array([1.5, 2])

k = 3
mu, r, loss = kmeans(X=X, k=k, max_iter=100)
print(mu)

#R, kmloss, mergeidx = kmeans_agglo(X=X.T, r=r)
#print(kmloss)
colors = ['blue', 'red', 'orange', 'black']

#for k_cur in range(k):
#    plt.scatter(X[np.argwhere(r==k_cur), 0], X[np.argwhere(r==k_cur), 1], c=colors[k_cur])

#plt.show()
'''