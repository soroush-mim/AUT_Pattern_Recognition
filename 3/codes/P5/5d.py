import numpy as np
from matplotlib import pyplot as plt

#setting mean vectors
m1 = np.array([-3,0])
m2 = np.array([3,0])
#setting covariance matrices
cov1 = np.array([[1.5,1],[1,1.5]])
cov2 = np.array([[1.5,-1],[-1,1.5]])
#generating samples
x1 = np.random.multivariate_normal(m1, cov1, 500)
x2 = np.random.multivariate_normal(m2, cov2, 500)
#plotting samples and seperating line
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.scatter(x1[:,0],x1[:,1] ,label='class 1')
ax1.scatter(x2[:,0],x2[:,1] , label = 'class 2')
ax1.axline((-7,0),(7,0) , label = 'projection line' , color = 'r')
ax1.axline((-0.3,5),(-0.3,-5) , label = 'seperating line' , color = 'g',linestyle='--')
ax1.legend()
fig.savefig('5d.png')