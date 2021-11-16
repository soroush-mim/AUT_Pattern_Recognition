import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd 
m = np.array([2,1])
sigma = np.array([[2, 1], [1, 3]] )
data = pd.read_csv('b.csv')
x1 = data['X'].to_numpy().reshape(len(data),1)
x2 = data['Y'].to_numpy().reshape(len(data),1)
x = np.concatenate([x1,x2]).reshape([500,2])
eigvals, eigvecs = np.linalg.eig(sigma)




p = np.array([[eigvecs[0][1] , eigvecs[0][0]],[eigvecs[1][1] , eigvecs[1][0]]])

y = np.dot((x-m) , p)

plt.plot(y[:,0],y[:,1], 'x')
plt.axis('equal')
plt.savefig('5j.png')



    