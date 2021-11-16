import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd 
from scipy.stats import multivariate_normal

mu = [2,1]
sigma = np.array([[1.9934, 0.9611], [0.9611, 2.8049]] )
data = pd.read_csv('b.csv')

x1 = data['X'].to_numpy().reshape(len(data),1)
x2 = data['Y'].to_numpy().reshape(len(data),1)
x = np.concatenate([x1,x2]).reshape([500,2])

eigvals, eigvecs = np.linalg.eig(sigma)
v = np.power(eigvals , -1/2)
t = np.diag(v)
t = np.dot(eigvecs , t)
t = np.dot(t , eigvecs.T)

x = np.dot(x , t)


plt.plot(x[:,0],x[:,1], 'x')
plt.axis('equal')
plt.savefig('5h1.png')

plt.clf()
# contour plot
xx, yy = np.meshgrid(x[:,0],x[:,1])
pos = np.dstack((xx,yy))
rv = multivariate_normal(mean=mu, cov=list(sigma))
fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
ax2.contourf(xx, yy, rv.pdf(pos))
plt.savefig('5h2.png')



    