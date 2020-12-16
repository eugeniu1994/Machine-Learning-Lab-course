import numpy as np
from sklearn.datasets import make_blobs
from sklearn.linear_model import Perceptron
import matplotlib.pyplot as plt

n_tot=200
n = int(n_tot/2)

X,y = make_blobs(n_tot, centers=2, cluster_std=4., random_state=1)
print('X:{}, y:{}'.format(np.shape(X), np.shape(y)))
plt.scatter(X[:,0], X[:,1], c=y)
plt.show()

delta = .08
d = 2
VC = d+1

np.random.seed(42)
order = np.random.permutation(n_tot)
train = order[:n]
test = order[n:]

my_classifier = Perceptron()

my_classifier.fit(X[train,:], y[train])

predictions = my_classifier.predict(X[test,:])

y_test = y[test]
y_train = y[train]

print('total test:{}, errors:{}'.format(len(y_test), (y_test != predictions).sum()/len(y_test)))

predictions = my_classifier.predict(X[train,:])
print('total train:{}, errors:{}'.format(len(y_train), (y_train != predictions).sum()/len(y_train)))

err_test, err_train = [],[]
V_dim, R_dim = [],[]

VC = 3
d = 2
for i in range(15,300):
    #print('i ',i)
    my_classifier = Perceptron()
    n_tot = i
    X, y = make_blobs(n_tot, centers=2, cluster_std=4., random_state=1)

    order = np.random.permutation(n_tot)
    n = int(n_tot / 2)

    train = order[:n]
    test = order[n:]
    my_classifier.fit(X[train, :], y[train])

    y_test = y[test]
    y_train = y[train]

    predictions = my_classifier.predict(X[test, :])
    test_err =  (y_test != predictions).sum()/len(y_test)

    predictions = my_classifier.predict(X[train, :])
    train_err =  (y_train != predictions).sum()/len(y_train)

    V = np.sqrt(2.*np.log(np.e*n/d)/(n/d)) + np.sqrt(np.log(1./delta)/(n/d))
    #V = np.sqrt(2.*np.log(np.e*n/d)/(n/d)) + np.sqrt(np.log(1./delta)/(2*n))

    #V = np.sqrt((2.*d*np.log(np.e*n/d))/n) + np.sqrt(np.log(1/delta)/(2*n))
    #V = np.sqrt((8.*d*(np.log(2*np.e*n/d) + 8*np.log(4/delta)))/n)

    #R = np.sqrt(np.log(1. / delta) / (2 * n))
    R = 3*np.sqrt(np.log(2/delta)/(2*n))

    diff = test_err - train_err

    err_test.append(test_err)
    err_train.append(train_err)
    V_dim.append(V)
    R_dim.append((R))
    if n_tot >= 99 and n_tot<=101:
        print('R: {}  V: {}, , diff:{}'.format(round(R,4), round(V,3), round(diff,4)))
        print('R_{},  V_{}'.format(R+diff, V-diff))
        R_bound = train_err + V
        print('Generalization bound ',R_bound)
        print()

plt.plot(err_test, label='test error')
plt.plot(err_train, label='train error')
plt.plot(V_dim, label='VC')
plt.plot(R_dim, label='R')

plt.legend()
plt.show()
