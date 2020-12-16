import numpy as np
import matplotlib.pyplot as plt

X = np.array([[-1,0],[2,1],[2,-1],[1,0],[-2,-1],[-2,1]])
y = np.array([1,1,1,-1,-1,-1])
print('X:{}, y:{}'.format(np.shape(X), np.shape(y)))

#plt.scatter(X[:,0],X[:,1], c=y)
#plt.show()

l = 1
lr = .1
w = np.array([0,0])
for i in range(len(X)):
    xi, yi =X[i], y[i]

    s = yi*(w.T@xi)   # 1 x  ([2*1] x [1*2])
    print('yi*(w.T@xi) ',s)

    gradient = 0.
    if s < 1:
        gradient = (-yi*xi) + (l*w)

    if s>=1:
        gradient = l*w
        
    print('gradient ',gradient)
    w = w - lr*gradient
    print('result ',w)
    print()

# Generate and plot a synthetic imbalanced classification dataset

'''from collections import Counter
from sklearn.datasets import make_classification
from matplotlib import pyplot
from numpy import where
# define dataset
X, y = make_classification(n_samples=50, n_features=2, n_redundant=0,
	n_clusters_per_class=1, weights=[0.999], flip_y=0, random_state=4)
# summarize class distribution
counter = Counter(y)
print(counter)
# scatter plot of examples by class label
for label, _ in counter.items():
	row_ix = where(y == label)[0]
	pyplot.scatter(X[row_ix, 0], X[row_ix, 1], label=str(label))
pyplot.legend()
pyplot.show()'''