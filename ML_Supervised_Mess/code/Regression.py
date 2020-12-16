import numpy as np
import matplotlib.pyplot as plt

m0=100
m=2*m0+1

xx=2*np.arange(m+1)/m-1 ## input data is in the range [-1,+1], with steps 0.01
rr=xx**2 ## output values: a parabola

niteration=100 ## number of iterations
H=5 ## number of nodes in the hidden layer
eta=0.3 ## learning speed, step size

np.random.seed(12345)
W=2*(np.random.rand(H)-0.5)/100 ## random uniform in [-0.01,0.01]
V=2*(np.random.rand(H)-0.5)/100 ## random uniform in [-0.01,0.01]

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def transfer_derivative(output):
    return output * (1.0 - output)

print('xx:{},  rr:{}'.format(np.shape(xx), np.shape(rr)))
print('W:{}, V:{}'.format(np.shape(W), np.shape(V)))

for i in range(niteration): #training
    for random_idx in range(len(xx)):
        x,r = xx[random_idx],rr[random_idx]

        z = sigmoid(W.T*x)
        #print('z:{},  shape:{}'.format(z, np.shape(z)))
        y = V.T.dot(z.T)
        #print('y:{},  shape:{}'.format(y, np.shape(y)))

        error = (r-y)
        #error = (y - r)
        print('error:{}'.format(error))
        delta_V = np.dot(error,z)
        #print('error:{}, delta_V:{},  shape:{}'.format(error,delta_V, np.shape(delta_V)))
        V += eta*delta_V

        delta_W = (error*V) * (transfer_derivative(z)*x)
        #print('delta_W:{},  shape:{}'.format(delta_W, np.shape(delta_W)))
        #V += eta*delta_V
        W += eta*delta_W

yy = [] #prediction
for i in range(len(xx)):
    z = sigmoid(W.T * xx[i])
    y_pred = V.T.dot(z.T)
    yy.append(y_pred)

print('xx[m0+1] ',xx[m0+1])
err = rr[m0+1] - yy[m0+1]
print('rr[m0+1]:{},  yy[m0+1]:{},  err:{}'.format(rr[m0+1],yy[m0+1],err))

plt.plot(xx,rr, label='True data')
plt.plot(xx,yy,label='Predicted')
plt.legend()
plt.show()



