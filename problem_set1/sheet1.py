import numpy as np
import matplotlib.pyplot as plt
import scipy.io as spio          #used to lod .mat dataset
import scipy.sparse.csgraph as checkGraphConnected

class PCA():
    def __init__(self, Xtrain):
        self.Mean = np.mean(Xtrain, axis=0)
        Centered = Xtrain - self.Mean
        self.C = (Centered.T @ Centered) / (np.shape(Xtrain)[0] - 1)
        self.D, self.U = np.linalg.eig(self.C)

        Descending_index = self.D.argsort()[::-1] #Descending order
        self.D = self.D[Descending_index]
        self.U = self.U[:, Descending_index]

    def project(self, Xtest, m):
        Z = (Xtest - self.Mean) @ self.U[:, :m]
        return Z

    def denoise(self, Xtest, m):
        Y = self.Mean + (self.project(Xtest, m) @ self.U[:, :m].T)
        return Y

def gammaidx(X, k):
    #dist_matrix = np.sqrt(np.sum((X[None, :] - X[:, None])**2, -1))
    dist_matrix = distance_function(X, X)

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

    return c

def euclid_dist(x1, x2):
    return np.sqrt(sum((x1 - x2) ** 2))

def distance_function(x1, x2):
    x1 = np.asarray(x1)
    x2 = np.atleast_2d(x2)
    x1_dim = x1.ndim
    x2_dim = x2.ndim
    if x1_dim == 1:
        x1 = x1.reshape(1, 1, x1.shape[0])
    if x1_dim >= 2:
        x1 = x1.reshape(np.prod(x1.shape[:-1]), 1, x1.shape[-1])
    if x2_dim > 2:
        x2 = x2.reshape(np.prod(x2.shape[:-1]), x2.shape[-1])

    diff = x1 - x2
    arr_dist = np.sqrt(np.einsum('ijk,ijk->ij', diff, diff))
    return np.squeeze(arr_dist)

def checkIfGraphisConnected(graph,n):
    I = np.eye(n)
    for i in range(n):
        I[i, graph[i]] = 1
    n_components, labels = checkGraphConnected.connected_components(I)
    if n_components != 1:
        raise ValueError('Graph is not connected!')

def lle(X, m, tol, n_rule, k=None, epsilon=None, returnGraph=False):
    n = X.shape[0]
    # 1) Find neighbours
    distance_matrix = distance_function(X, X)
    neighbours = None
    if n_rule == 'knn':
        neighbours = np.argsort(distance_matrix, axis=1)[:, 1:k + 1]
    else:
        _neighbors = []
        for i in range(n):
            i_neighbors = np.flatnonzero((distance_matrix[i,] > 0) & (distance_matrix[i,] < epsilon))
            _neighbors.append(i_neighbors)
        neighbours = np.array(_neighbors)
    checkIfGraphisConnected(neighbours,n)

    # 2) Reconstruction weights W
    W = np.zeros((n, n))
    for i in range(n):
        j = neighbours[i,]
        Z = X[j, :] - X[i, :]  # matrix Z with all neighbours & substract Xi from it

        C = np.dot(Z, Z.T)
        #w = np.linalg.pinv(C) #
        #w = C.T @ np.linalg.inv(C @ C.T + tol * np.eye(np.shape(C)[0]))  # psuedoinverse with a regularization term
        w = np.linalg.inv(C + tol * np.eye(np.shape(C)[0]))

        scale = 2 / np.sum(w)
        #print('scale is: ', scale)
        W[i, j] = scale * np.sum(w, axis=1) / 2

    # 3) Embedding coordinates Y using weights W
    M = np.subtract(np.eye(n), W)
    eig_val, eig_vec = np.linalg.eigh(np.dot(np.transpose(M), M))
    ascending_index = eig_val.argsort()[::1]  # ascending order
    eig_vec = eig_vec[:, ascending_index]

    Y = eig_vec[:, 1:m + 1]
    if returnGraph:
        return Y, neighbours
    else:
        return Y

#Assignment 5 -> Load data, analyse PCA, Noise scenarios [&plots]
def usps():
    # 1. Load the usps data set.
    mat = spio.loadmat('data/usps.mat')
    # print(mat.items())
    data_patterns = mat['data_patterns']
    print('data_patterns:{}'.format(np.shape(data_patterns)))

    # 2. Analysis of PCA:
    X = data_patterns.T
    print('X shape is: {}'.format(np.shape(X)))

    pca = PCA(X)
    per_val = pca.D # np.round(pca.D, 1)
    plt.bar(x=range(1, len(pca.D) + 1), height=per_val)
    plt.xlabel('Principal values')
    plt.title('All principal values')
    plt.show()

    per_val = pca.D[:25]# np.round(pca.D[:25], 1)
    plt.bar(x=range(1, 26), height=per_val)
    plt.xlabel('Principal values')
    plt.title('Largest 25 principal values')
    plt.show()

    fig, axs = plt.subplots(2, 3)
    axs[0, 0].imshow(pca.U[:, 0].reshape(16, 16))
    axs[0, 0].set_title('comp 1')
    axs[0, 1].imshow(pca.U[:, 1].reshape(16, 16))
    axs[0, 1].set_title('comp 2')
    axs[0, 2].imshow(pca.U[:, 2].reshape(16, 16))
    axs[0, 2].set_title('comp 3')
    axs[1, 0].imshow(pca.U[:, 3].reshape(16, 16))
    axs[1, 0].set_title('comp 4')
    axs[1, 1].imshow(pca.U[:, 4].reshape(16, 16))
    axs[1, 1].set_title('comp 5')
    plt.axis('off')
    for ax in axs.flat:
        ax.label_outer()
    plt.show()

    # 3. Three noise scenarios
    d = np.shape(X)[1]
    row, col = 16, 16
    mean = 0
    sigma = [0.3, 0.8, 1.5]  # low gaussian, high gaussian, outliers
    scenarios_labels = ['Low', 'High', 'Outliers']
    for scenarios in range(len(sigma)):
        gauss = np.random.normal(mean, sigma[scenarios], (d))
        if scenarios < 2:
            noisy_X = X.copy() + gauss
        else:
            #add noise outlier to 5 images
            noisy_X = X.copy()
            noisy_X[:5,:] += gauss

        pca_noisy = PCA(noisy_X)
        per_val_noisy = pca_noisy.D

        fig, (ax1, ax2) = plt.subplots(2)
        ax1.bar(x=range(1, len(pca_noisy.D) + 1), height=per_val_noisy)
        ax1.set_title('All principal values {} Noisy case'.format(scenarios_labels[scenarios]))

        per_val_noisy = pca_noisy.D[:25] #np.round(pca_noisy.D[:25], 1)
        plt.bar(x=range(1, 26), height=per_val_noisy)
        ax2.set_title('Largest 25 principal values {} Noisy case'.format(scenarios_labels[scenarios]))

        plt.show()
        m = 10
        reconstructed = pca.denoise(noisy_X, m)

        difference = X - reconstructed
        err = np.sqrt(np.einsum('ij,ij->i', difference, difference).sum(-1))
        print('Reconstruction error: {}'.format(err))

        fig, axs = plt.subplots(3, 10)
        fig.suptitle('[Original > Noisy > Reconstruction]  -  {} Noisy case'.format(scenarios_labels[scenarios]))
        for j in range(10):
            axs[0, j].imshow(X[j, :].reshape(row, col))
            axs[1, j].imshow(noisy_X[j, :].reshape(row, col))
            axs[2, j].imshow(reconstructed[j, :].reshape(row, col))

        for ax in axs.flat:
            ax.label_outer()
        plt.show()

#This function is called in lle_noise() (for assignment 8), to plot the result
def plotData(Xt, Xp1_k1, Xp1_k2, Xp2_k1, Xp2_k2, n_rule, noise1, noise2, k1, k2, Xp_real, graph1,graph2,graph3,graph4):
    plt.figure(figsize=(14, 8))
    n = Xt.shape[0]

    plt.subplot(2, 4, 1)
    plt.scatter(Xp1_k1[:, 0], Xp_real[:, 0], 30, Xp1_k1[:, 0],zorder=2, cmap=plt.cm.nipy_spectral)
    plt.title('{} noise, k:{}'.format(noise1, k1))
    plt.xlabel(r'Embedding')
    plt.xticks([], [])
    plt.yticks([], [])

    plt.subplot(2, 4, 2)
    plt.scatter(Xp1_k2[:, 0], Xp_real[:, 0], 30, Xp1_k2[:, 0],zorder=2, cmap=plt.cm.nipy_spectral)
    plt.title(' {} noise, k:{}'.format(noise1, k2))
    plt.xlabel(r'Embedding')
    plt.xticks([], [])
    plt.yticks([], [])

    plt.subplot(2, 4, 3)
    plt.scatter(Xp2_k1[:, 0], Xp_real[:, 0], 30, Xp2_k1[:, 0],zorder=2, cmap=plt.cm.nipy_spectral)
    plt.title('${} noise, k:{}'.format(noise2, k1))
    plt.xlabel(r'Embedding')
    plt.xticks([], [])
    plt.yticks([], [])

    plt.subplot(2, 4, 4)
    plt.scatter(Xp2_k2[:, 0], Xp_real[:, 0], 30, Xp2_k2[:, 0],zorder=2, cmap=plt.cm.nipy_spectral)
    plt.title('${} noise, k:{}'.format(noise2, k2))
    plt.xlabel(r'Embedding')
    plt.xticks([], [])
    plt.yticks([], [])

    plt.subplot(2, 4, 5)
    plt.scatter(Xt[:, 0], Xt[:, 1], 5, c=Xp1_k1[:, 0],zorder=2, cmap=plt.cm.nipy_spectral)
    for i in range(n):
        Z = Xt[graph1[i,], :]
        plt.plot(Z[:, 0], Z[:, 1], color='black', zorder=0, lw=1)
    plt.title('k=%i,noise=0.2' % (k1), size=12)
    plt.axis('equal')
    plt.axis('off')

    plt.subplot(2, 4, 6)
    plt.scatter(Xt[:, 0], Xt[:, 1], 5, c=Xp1_k2[:, 0],zorder=2, cmap=plt.cm.nipy_spectral)
    for i in range(n):
        Z = Xt[graph2[i,], :]
        plt.plot(Z[:, 0], Z[:, 1], color='black', zorder=0, lw=1)
    plt.title('k=%i,noise=0.2' % (k2), size=12)
    plt.axis('equal')
    plt.axis('off')

    plt.subplot(2, 4, 7)
    plt.scatter(Xt[:, 0], Xt[:, 1], 5, c=Xp2_k1[:, 0],zorder=2, cmap=plt.cm.nipy_spectral)
    for i in range(n):
        Z = Xt[graph3[i,], :]
        plt.plot(Z[:, 0], Z[:, 1], color='black', zorder=0, lw=1)
    plt.title('k=%i,noise=1.8' % (k1), size=12)
    plt.axis('equal')
    plt.axis('off')

    plt.subplot(2, 4, 8)
    plt.scatter(Xt[:, 0], Xt[:, 1], 5,  c=Xp2_k2[:, 0],zorder=2, cmap=plt.cm.nipy_spectral)
    for i in range(n):
        Z = Xt[graph4[i,], :]
        plt.plot(Z[:, 0], Z[:, 1], color='black', zorder=0, lw=1)
    plt.title('k=%i,noise=1.8' % (k2), size=12)
    plt.axis('equal')
    plt.axis('off')

    plt.show()

#Assignment 8 -> Load data, Add noise, Apply LLE, Neighborhood graph, [&plots]
def lle_noise():
    # 1. Load the data set.
    dataset = np.load('data/flatroll_data.npz')
    print(dataset.files)
    Xt = dataset['Xflat'].T   # (1000 x 2)
    Xp_real = dataset['true_embedding'].T  # (1000 x 1)

    # 2. Add Gaussian noise
    d = np.shape(Xt)[1]
    gauss02 = np.random.normal(0, 0.2, (d))
    gauss18 = np.random.normal(0, 1.8, (d))

    dts1 = Xt + gauss02  # (1000 x 2)
    dts2 = Xt + gauss18  # (1000 x 2)

    # 3. Apply LLE
    m = 1
    k1 = 8
    k2 = 40

    Xp1_k1, graph1 = lle(dts1, m, n_rule='knn', k=k1, tol=1e-3, returnGraph=True)
    Xp1_k2, graph2 = lle(dts1, m, n_rule='knn', k=k2, tol=1e-3, returnGraph=True)

    Xp2_k1, graph3 = lle(dts2, m, n_rule='knn', k=k1, tol=1e-3, returnGraph=True)
    Xp2_k2, graph4 = lle(dts2, m, n_rule='knn', k=k2, tol=1e-3, returnGraph=True)

    plotData(Xt, Xp1_k1, Xp1_k2, Xp2_k1, Xp2_k2, 'knn', 0.2, 1.8, k1, k2, Xp_real, graph1,graph2,graph3,graph4)

#Assignment 7 -> Load data, Compute LLE, [&plots]
def lle_visualize():
    fig = plt.figure(figsize=(8, 8))

    fishbowl = np.load('data/fishbowl_dense.npz')
    Xt1 = fishbowl['X'].T
    referrence = Xt1[:,2]
    ax = fig.add_subplot(3, 2, 1, projection='3d',adjustable='box')
    ax.scatter(Xt1[:, 0], Xt1[:, 1], Xt1[:, 2], c=referrence, s=10, edgecolor='k')
    ax.set_title('fishbowl Original')
    ax.set_xlabel(r'$X$')
    ax.set_ylabel(r'$Y$')
    ax.set_zlabel(r'$Z$')
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    ax.get_zaxis().set_ticks([])

    k, n_rule = 30, 'knn'
    Embedding2D_1 = lle(Xt1, 2, n_rule='knn', k=k, tol=1e-3)
    plt.subplot(3, 2, 2)
    plt.scatter(Embedding2D_1[:, 0], Embedding2D_1[:, 1], s=10, c=referrence)
    plt.title('fishbowl embedding k:{},  n_rule:{}'.format(k,n_rule))
    plt.xticks([], [])
    plt.yticks([], [])

    #----------------------------------------------
    swissroll = np.load('data/swissroll_data.npz')
    Xt2 = swissroll['x_noisefree'].T
    referrence = swissroll['z'].T
    ax = fig.add_subplot(3, 2, 3, projection='3d', adjustable='box')
    ax.scatter(Xt2[:, 0], Xt2[:, 1], Xt2[:, 2], c=referrence[:,0],  s=10, edgecolor='k' )
    ax.set_title('swissroll Original')
    ax.set_xlabel(r'$X$')
    ax.set_ylabel(r'$Y$')
    ax.set_zlabel(r'$Z$')
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    ax.get_zaxis().set_ticks([])

    k, n_rule = 15, 'knn'
    Embedding2D_2 = lle(Xt2, 2, n_rule='knn', k=k, tol=1e-3)
    plt.subplot(3, 2, 4)
    plt.scatter(Embedding2D_2[:, 0], Embedding2D_2[:, 1], s=10, c=referrence[:,0])
    plt.title('swissroll embedding k:{},  n_rule:{}'.format(k, n_rule))
    plt.xticks([], [])
    plt.yticks([], [])

    #---------------------------------------------------
    flatroll = np.load('data/flatroll_data.npz')
    Xt3 = flatroll['Xflat'].T
    referrence = flatroll['true_embedding'].T
    plt.subplot(3, 2, 5)
    plt.scatter(Xt3[:, 0], Xt3[:, 1], s=10, edgecolor='k' )
    plt.title('flatroll Original')
    plt.ylabel(r'$Y$')
    plt.xlabel(r'$X$')
    plt.xticks([], [])
    plt.yticks([], [])

    k,n_rule = 10, 'knn'
    Embedding1D = lle(Xt3, 1, n_rule='knn', k=k, tol=1e-3)
    plt.subplot(3, 2, 6)
    plt.scatter(referrence[:, 0], Embedding1D[:, 0], s=10)
    plt.title('flatroll embedding k:{},  n_rule:{}'.format(k,n_rule))
    plt.ylabel(r'$Y$')
    plt.xlabel(r'$X$')
    plt.xticks([], [])
    plt.yticks([], [])

    plt.show()

#Assignment 6 -> Load data, Sample outliers, compute gamma & dist_mean, Compute AUC, exemplary run [&plots]
def outliers_calc():
    banana = np.load('data/banana.npz')
    print(banana.files)
    data=banana['data'].T
    label=banana['label'].T
    print('data:{},  label:{}'.format(np.shape(data), np.shape(label)))
    n = np.shape(data)[0]
    percentage = [1, 10, 50, 100]
    data_to_plot, labels = [],[]
    for p in percentage:
        sample_number = int((p*n)/100)
        AUC_K3_history, AUC_K10_history, AUC_dist_history = [],[],[]
        for i in range(10): #change here to 100
            #1. Sample a random set of outliers
            Xout = np.array([np.random.uniform(-4, 4, sample_number),
                             np.random.uniform(-4, 4, sample_number)]).T
            Yout = np.ones(sample_number)[..., np.newaxis] #outlier positive class
            #2. Add the outliers to the positive class
            X = np.append(data, Xout, axis=0)
            Y = np.append(label, Yout, axis=0) #y true

            gamma_k3 = gammaidx(X, k=3)
            gamma_k10 = gammaidx(X, k=10)
            dist_to_mean = np.sqrt(((X - np.mean(X, axis=0)) ** 2).sum(-1))

            #3. Compute the AUC
            AUC_K3_history.append(auc(y_true=Y, y_pred=gamma_k3))
            AUC_K10_history.append((auc(y_true=Y, y_pred=gamma_k10)))
            AUC_dist_history.append((auc(y_true=Y, y_pred=dist_to_mean)))

        auc_3 = np.array(AUC_K3_history).mean(axis=0)
        auc_10 = np.array(AUC_K10_history).mean(axis=0)
        auc_dist = np.array(AUC_dist_history).mean(axis=0)
        print('{} Percentage, AUC k3:{}, AUC k10:{}, AUC dist to mean :{}'.format(p,round(auc_3,2), round(auc_10,2), round(auc_dist,2)))

        data_to_plot.append(AUC_K3_history)
        data_to_plot.append(AUC_K10_history)
        data_to_plot.append(AUC_dist_history)
        labels.append('{}% AUC_k-3'.format(p))
        labels.append('{}% AUC_k-10'.format(p))
        labels.append('{}% AUC_mean'.format(p))

    fig = plt.figure(1, figsize=(16, 8))
    fig.suptitle('AUC (area under the ROC)')

    ax = fig.add_subplot(111)
    ax.boxplot(data_to_plot)
    ax.set_xlabel('Samples')
    ax.set_ylabel('AUC')
    ax.grid()
    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    plt.show()

    #exemplary run
    scaler = 20 #used to change the size of the points
    plt.figure(figsize=(12, 8))
    sample_number = int((10 * n) / 100) # 10 % of data
    sampled_outliers = np.array([np.random.uniform(-4, 4, sample_number),
                                 np.random.uniform(-4, 4, sample_number)]).T
    X = np.append(data, sampled_outliers, axis=0)

    plt.subplot(3, 1, 1)
    gamma_k3_score = gammaidx(X, k=3)
    plt.scatter(data[:, 0], data[:, 1], s=gamma_k3_score[:n,]*scaler, c='g', label='Original data')
    plt.scatter(sampled_outliers[:, 0], sampled_outliers[:, 1], s=gamma_k3_score[n:,]*scaler, c='r', label='Outliers')
    plt.title('Condition a) gamma with k-3')
    plt.legend()

    plt.subplot(3, 1, 2)
    gamma_k10_score = gammaidx(X, k=10)
    plt.scatter(data[:, 0], data[:, 1], s=gamma_k10_score[:n, ] * scaler, c='g', label='Original data')
    plt.scatter(sampled_outliers[:, 0], sampled_outliers[:, 1], s=gamma_k10_score[n:, ] * scaler, c='r',
                label='Outliers')
    plt.title('Condition b) gamma with k-10')
    plt.legend()

    plt.subplot(3, 1, 3)
    dist_to_mean_score = np.sqrt(((X - np.mean(X, axis=0)) ** 2).sum(axis=1))
    scaler = 5
    plt.scatter(data[:, 0], data[:, 1], s=dist_to_mean_score[:n, ] * scaler, c='g', label='Original data')
    plt.scatter(sampled_outliers[:, 0], sampled_outliers[:, 1], s=dist_to_mean_score[n:, ] * scaler, c='r',
                label='Outliers')
    plt.title('Condition c) dist. to mean score')
    plt.legend()

    plt.show()

if __name__ == '__main__':
    #usps()              # assignment 5
    #outliers_calc()     # assignment 6
    lle_visualize()     # assignment 7
    lle_noise()         # assignment 8


