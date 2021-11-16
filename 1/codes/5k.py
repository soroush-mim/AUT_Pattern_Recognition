import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
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
 
m1 = np.sum(x[:,0]) / 500
m2 = np.sum(x[:,1]) / 500

print('m1= ',m1,' m2=' , m2)

cov11 = np.sum(np.power((x[:,0] - m1),2))/500
cov22 = np.sum(np.power((x[:,1] - m2),2))/500
cov12 = np.dot((x[:,0] - m1).T , (x[:,1] - m2))/500

print('cov11= ',cov11,' cov22 = ' , cov22 ,' cov12=cov21= ', float(cov12))

cov = np.array([[cov11 , cov12],[cov12 , cov22]])

eigvals, eigvecs = np.linalg.eig(cov)

print('eigenvalues: ' , eigvals)

print('eigenvectors: ',eigvecs)



    