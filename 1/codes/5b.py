import numpy as np
import matplotlib.pyplot as plt 
from scipy.stats import multivariate_normal
import pandas as pd  

mu = [2, 1]
cov = [[2, 1], [1, 3]] 
#generating samples
x,y = np.random.multivariate_normal(mu, cov, 500).T
#saving samples
dict = {'X': x, 'Y': y}  
df = pd.DataFrame(dict) 
# saving the dataframe 
df.to_csv('b.csv') 
#plotting samples
plt.plot(x,y, 'x')
plt.axis('equal')
plt.savefig('5b1.png')

plt.clf()
# contour plot
xx, yy = np.meshgrid(x,y)
pos = np.dstack((xx,yy))
rv = multivariate_normal(mean=mu, cov=cov)
fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
ax2.contourf(xx, yy, rv.pdf(pos))
plt.savefig('5b2.png')