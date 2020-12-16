import numpy as np

eps = 0.08
delta = 0.04

d = 3
H = 3**d #works
#H = 8

m = (1/eps)*(np.log(H) + np.log(1/delta))
print(m)
print(np.ceil(m))


#b)
delta = 0.05

l1 = np.log(0.05/2)
print('l1 ',l1)

l2 = np.log(0.1/2)
print('l2 ',l2)

alpha = (l2/l1)
print('alpha ',alpha)

print(np.log(2/0.08)/np.log(2/0.04))

