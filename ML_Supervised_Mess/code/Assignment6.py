import numpy as np
import matplotlib.pyplot as plt
import itertools



AAW,DW,PP,D,F = 1,2,3,4,5
def kendall_tau_distance(order_a, order_b):
    pairs = itertools.combinations(range(1, len(order_a)+1), 2)
    distance = 0
    for x, y in pairs:
        a = order_a.index(x) - order_a.index(y)
        b = order_b.index(x) - order_b.index(y)
        if a * b < 0:
            distance += 1
    return distance

#print(kendall_tau_distance([3,1,2], [2,1,3]))
Alice = [AAW,DW,PP,D,F]
Bob = [PP,D,AAW,DW,F]

a = [PP,DW,D,AAW,F]
b = [AAW,PP,DW,D,F]
c = [PP,AAW,DW,D,F]

labels=['a','b','c']
data = [a,b,c]
d_max_array = []
for i, s in enumerate(data):
    d_Alice = kendall_tau_distance(Alice,s)
    d_Bob = kendall_tau_distance(Bob, s)
    d_max = max(d_Alice,d_Bob)
    d_max_array.append(d_max)
    print('Order:{},  d_Alice:{}, d_Bob:{}, d_max:{}'.format(labels[i], d_Alice,d_Bob,d_max))

print('small value:{} , corespond to order:{}'.format(min(d_max_array), labels[np.argmin(d_max_array)]))


'''from sklearn.linear_model import Perceptron
from sklearn.datasets import load_digits
from sklearn.metrics import pairwise_distances, accuracy_score
from sklearn.metrics import euclidean_distances, hamming_loss
from sklearn.multiclass import OutputCodeClassifier
from sklearn.metrics import plot_confusion_matrix

np.random.seed(2020)

# the constants for creating the data
n_tot = 200
ntr = 50
nts = n_tot - ntr
nc = 10

X, y = load_digits(n_class=nc, return_X_y=True)

# divide into training and testing
Xtr = X[:ntr, :]
ytr = y[:ntr]
Xts = X[ntr:(ntr + nts), :]
yts = y[ntr:(ntr + nts)]

c = 2  # 10# 500

X_train, X_test, y_train, y_test = Xtr, Xts, ytr, yts
print('Xtr:{},ytr:{},   Xts:{},yts:{}'.format(np.shape(Xtr), np.shape(ytr), np.shape(Xts), np.shape(yts)))

#plt.imshow(image, cmap=plt.)


classifier = Perceptron(random_state=2020)
ecoc_classifier = OutputCodeClassifier(classifier, code_size=c)
ecoc_classifier = ecoc_classifier.fit(X_train, y_train)

matrix = plot_confusion_matrix(ecoc_classifier, X_test, y_test,cmap=plt.cm.Blues,normalize='true')
plt.title('Confusion matrix for ECOC classifier')
plt.show()

y_predicted = ecoc_classifier.predict(X_test)
print('y_predicted ', np.shape(y_predicted))
print("---Accuracy: ", accuracy_score(y_test, y_predicted))

##---------------------------------------------------------

'''
'''code_size = 50  # c
classes_ = np.unique(y_train)  # [0 1 2 3 4 5 6 7 8 9]
n_classes = classes_.shape[0]
code_size_ = int(n_classes * code_size)
code_book_ = np.round(np.random.rand(n_classes, code_size_))
print('code_book_ ', np.shape(code_book_))
# code_book_[code_book_ > 0.5] = 1
# code_book_[code_book_ != 1] = -1
# print(code_book_)

classes_index = {c: i for i, c in enumerate(classes_)}
#print(classes_index)
Y = np.array([code_book_[classes_index[y_train[i]]] for i in range(X_train.shape[0])], dtype=np.int)
print('Y surrogate', np.shape(Y)) #surrogate
#print(Y)
print('=========== TRAINING =============')
clfs = []  # Training
for j in range(Y.shape[1]):
    X_, Y_ = X_train, Y[:, j]
    f_j = Perceptron(random_state=2020)

    f_j.fit(X=X_, y=Y_)
    clfs.append(f_j)



print('===========  TESTING  =============')
# Predict
def predict_function(estimator, data):
    score = estimator.predict(data)
    return score

Y_pred = np.array([predict_function(clf, X_test) for clf in clfs]).T
print('Y_pred ', np.shape(Y_pred))
pdists = pairwise_distances(Y_pred, code_book_, metric="hamming")*Y_pred.shape[1]
print('pdists', np.shape(pdists))
armin = pdists.argmin(axis=1)
print('armin ', np.shape(armin))
print(armin)

y_pred = classes_[armin]
print('y_pred:{}'.format(np.shape(y_pred)))
print("Accuracy: ", accuracy_score(y_test, y_pred))

min_Hdist_1 = np.min(pdists)
print('min_Hdist_1 ', min_Hdist_1)
print('pdists ', np.shape(pdists))
print('clfs ', np.shape(clfs))
for j in range(pdists.shape[1]):
    H_d = min(pdists[:,j])
    print('j:{},  H_d {}'.format(j,H_d))'''

