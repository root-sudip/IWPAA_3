import math
from random import random

from sklearn.datasets import make_circles
import numpy as np

from sklearn import preprocessing
min_max = preprocessing.MinMaxScaler()
# min_max.fit_transform(a)

import matplotlib.pyplot as plt




# X = np.asarray([[15.8,3],[16.1,4],[15,5],[16.8,3.3],[17.1,2],[17.1,5.1],[15.5,3],[16.5,2.5],[16.8,3.8],[15.3,4.7],[0,5],[0.3,3.1],[0.7,4.6],[1.2,2.6],[2.7,3],[3.1,4],[2.6,4.9]])

x1 = np.random.uniform(low=0.3, high=0.4, size=(20,))
x2 = np.random.uniform(low=0.4, high=0.4, size=(20,))

# x = np.rot90(np.vstack((x1,x2)))
# x = min_max.fit_transform(x)


# np.savetxt('x1.out', x1, delimiter=',')
# np.savetxt('x2.out', x2)

y1 = np.random.uniform(low=0.7, high=1., size=(20,))
y2 = np.random.uniform(low=0.1, high=0.3, size=(20,))

# y = np.rot90(np.vstack((y1,y2)))
# y = min_max.fit_transform(y)


# np.savetxt('y1.out', y1, delimiter=',')
# np.savetxt('y2.out', y2)


a1 = np.random.uniform(low=0.1, high=0.3, size=(20,))
a2 = np.random.uniform(low=0.7, high=1., size=(20,))

# a = np.rot90(np.vstack((a1,a2)))
# a = min_max.fit_transform(a)

# np.savetxt('a1.out', a1, delimiter=',')
# np.savetxt('a2.out', a2)


b1 = np.random.uniform(low=0.7, high=1., size=(20,))
b2 = np.random.uniform(low=0.7, high=1., size=(20,))

# np.savetxt('b1.out', b1, delimiter=',')
# np.savetxt('b2.out', b2)

# b = np.rot90(np.vstack((b1,b2)))
# b = min_max.fit_transform(b)


# x = np.rot90(np.vstack((x1,x2,y1,y2,a1,a2,b1,b2)))
# print(x.shape)
# x = min_max.fit_transform(x)


# print(x1,x[:,1])




plt.scatter(x1,x2)
plt.scatter(y1,y2)
plt.scatter(a1,a2)
plt.scatter(b1,b2)

plt.show()




#import random
#from math import sin, cos, radians, pi, sqrt
#import matplotlib.pyplot as plt

#def meteorites(r):
#    angle = random.uniform(0, r * pi)  # in radians
#    distance = sqrt(random.uniform(4, 12.25))
#    return distance * cos(angle), distance * sin(angle)


#x = np.zeros(50)
#
#x1 = np.zeros(50)
#y1 = np.zeros(50)

#for i in range(50):
#	x[i], y[i] = meteorites(2)

#for i in range(50):
#	x1[i], y1[i] = meteorites(6)


#plt.scatter(x,y)
#plt.scatter(x1,y1)
#plt.show()