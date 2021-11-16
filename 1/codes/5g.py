import numpy as np
import matplotlib.pyplot as plt 


sigma = np.array([[2, 1], [1, 3]])
sigmahat = np.array([[1.9934, 0.96116], [0.96116, 2.8049]])

eigvals, eigvecs = np.linalg.eig(sigma)
eigvals2, eigvecs2 = np.linalg.eig(sigmahat)
v = eigvals
vhat = eigvals2

print(v)
print(np.diag(v))
print(np.diag(vhat)) 



    