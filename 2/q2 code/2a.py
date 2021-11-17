import numpy as np
import matplotlib.pyplot as plt 
from scipy.stats import multivariate_normal

mu1 = [1, 2]
mu2 = [-1, -3]
cov1 = [[1.8, -0.7], [-0.7, 1.8]] 
cov2 = [[1.5, 0.3], [0.3, 1.5]] 

x1,y1 = np.random.multivariate_normal(mu1, cov1,1000).T
x2,y2 = np.random.multivariate_normal(mu2, cov2, 1000).T

xx1, yy1 = np.meshgrid(x1,y1)
pos1 = np.dstack((xx1,yy1))
rv1 = multivariate_normal(mean=mu1, cov=cov1)
fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
ax2.contourf(xx1, yy1, rv1.pdf(pos1))

fig2.savefig('2a1.png')

xx2, yy2 = np.meshgrid(x2,y2)
pos2 = np.dstack((xx2,yy2))
rv2 = multivariate_normal(mean=mu2, cov=cov2)
fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
ax2.contourf(xx2, yy2, rv2.pdf(pos2))

fig2.savefig('2a2.png')