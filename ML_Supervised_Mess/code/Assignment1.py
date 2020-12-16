import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score

def Ex2():
    times = 10000
    n, p = 35, .2
    s = np.random.binomial(n, p, times)
    #print(s)
    print(np.shape(s))

    r = sum(s >= 10) / times
    print(r)

    runs = 10000
    prob_6 = sum([1 for i in np.random.binomial(n, p, size=runs) if i >= 10]) / runs
    print('The probability of 6 heads is: ' + str(prob_6))

def Ex4():
    X = np.genfromtxt('X.csv', delimiter=',')[1:]
    Y = np.genfromtxt('Y.csv', delimiter=',')[1:]
    print('X:{}, Y:{}'.format(np.shape(X), np.shape(Y)))

    X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.3, random_state=1)
    print('X_train:{}, X_test:{}, Y_train:{}, Y_test:{}'.format(np.shape(X_train), np.shape(X_test), np.shape(Y_train), np.shape(Y_test)))

    plt.scatter(X[:,0],X[:,1], label='points')
    plt.legend()
    plt.show()

    LR = LinearRegression(fit_intercept=False)
    LR.fit(X_train, Y_train)

    y_predicted =LR.predict(X_test)
    rmse = np.sqrt(mean_squared_error(Y_test, y_predicted))
    print('rmse ',rmse) #5.259964639481718

def Ex5():
    X = np.genfromtxt('X.csv', delimiter=',')[1:]
    Y = np.genfromtxt('Y.csv', delimiter=',')[1:]
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=1)

    for k in [2, 5]:
        kf = KFold(n_splits=k, random_state=1, shuffle=True)
        error = []
        for train_index, test_index in kf.split(X_train):

            X_tr, X_t = X_train[train_index,:], X_train[test_index,:]
            y_tr, y_t = Y_train[train_index], Y_train[test_index]

            LR = LinearRegression(fit_intercept=False)
            LR.fit(X_tr, y_tr)

            y_predicted = LR.predict(X_t)

            rmse = np.sqrt(mean_squared_error(y_t, y_predicted))
            error.append(rmse)

        average = np.average(error)
        variance = np.std(error)

        print('k:{},  average:{}, variance:{}'.format(k, round(average, 2), round(variance, 2)))

if __name__ == '__main__':
    print('Main')
    Ex2()
    #Ex4()
    #Ex5()