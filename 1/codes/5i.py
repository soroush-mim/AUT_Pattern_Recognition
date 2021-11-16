import numpy as np
import matplotlib.pyplot as plt 

sigma = np.array([[2, 1], [1, 3]] )

eigvals, eigvecs = np.linalg.eig(sigma)

print('eigenvalues: ' , eigvals)

print('eigenvectors: ',eigvecs)


soa = np.array([[0, 0, eigvecs[0,0], eigvecs[1,0]], [0, 0, eigvecs[0,1], eigvecs[1,1]]])
X, Y, U, V = zip(*soa)
plt.figure()
ax = plt.gca()
ax.quiver(X, Y, U, V, angles='xy', scale_units='xy', scale=1)
ax.set_xlim([-2, 2])
ax.set_ylim([-2, 2])
plt.draw()
plt.savefig('5i.png')


    