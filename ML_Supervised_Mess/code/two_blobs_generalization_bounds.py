import numpy as np
from sklearn.datasets import make_blobs


def get_rademacher_bound(eerror, erc, n, delta):
    """

    :param eerror: empirical error
    :param erc: empirical rademacher complexity
    :param n: number of training samples
    :param delta: delta
    :return:
    """

    return eerror + erc + 3*np.sqrt(np.log(2/delta)/(2*n))

def get_vc_bound(eerror, vcd, n, delta):
    """
    :param eerror: empirical error
    :param vcd: the vc dimension
    :param n: number of training samples
    :param delta: delta
    :return:
    """

    from math import e

    # this version is the one in the lecture slides; with typo, used in quiz
    return eerror + np.sqrt((2*vcd*np.log((e*n)/vcd))/n) + np.sqrt((vcd*np.log(1/delta))/n)
    # # and this one is the one in Mohri book
    # return eerror + np.sqrt((2*vcd*np.log((e*n)/vcd))/n) + np.sqrt((np.log(1/delta))/(2*n))

# -----------
delta = 0.08
K = 100  # a lot of these to get a stable result over different random schemes
ns = np.array(range(20, 200, 5))
from sklearn.linear_model import Perceptron as binary_classifier
vcd = 3
# -----------

training_errors = []
test_errors = []
rademachers = []
vcdims = []

print("%3s - %5s - %5s" % ("n", "R.", "VC."))

for n_tot in ns:

    n = int(n_tot/2)  # will use half in training, half in testing

    X, y = make_blobs(n_tot, centers=2, cluster_std=4.0, random_state=1)

    # divide into training and testing
    np.random.seed(42)
    order = np.random.permutation(n_tot)
    train = order[:n]
    test = order[n:]

    bc = binary_classifier()
    bc.fit(X[train, :], y[train])
    training_error = np.count_nonzero(bc.predict(X[train, :]) - y[train])/n
    test_err = (y[train] != bc.predict(X[train, :])).sum() / n

    #print('training error {},  test_err:{}'.format(training_error, test_err))

    test_error = np.count_nonzero(bc.predict(X[test, :]) - y[test])/n

    # in order to calculate the rademacher error we need to randomize the labels:
    random_errors = []
    for k in range(K):
        new_labels = np.round(np.random.rand(n))
        bc.fit(X[train, :], new_labels)
        random_errors.append(np.count_nonzero(bc.predict(X[train, :]) - new_labels)/n)

    training_errors.append(training_error)
    test_errors.append(test_error)
    rademachers.append(get_rademacher_bound(training_error, 0.5-np.mean(random_errors), n, delta))
    vcdims.append(get_vc_bound(training_error, vcd, n, delta))

    print("%3d - %5.2f - %5.2f" % (n_tot, rademachers[-1], vcdims[-1]))


import matplotlib.pyplot as plt

plt.figure()
plt.plot(ns, training_errors, label="tr.err.")
plt.plot(ns, test_errors, label="ts.err.")
plt.plot(ns, rademachers, label="radem.")
plt.plot(ns, vcdims, label="VCdim")
plt.legend()
plt.xlabel("n")
plt.ylabel("err")

plt.show()
