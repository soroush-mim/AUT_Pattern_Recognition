import numpy as np
import matplotlib.pyplot as plt 
mus = [[-3, 3],[0,0],[6,-6]]
covs =[[[1, 0], [0, 1]],[[1,1],[1,1]],[[6,0],[0,1]] ]
#generating samples
for i in range(3):
    mu = mus[i]
    cov = covs[i]
    x,y = np.random.multivariate_normal(mu, cov, 500).T
    #plotting samples
    plt.plot(x,y, 'x')
    plt.axis('equal')
    plt.savefig('5c'+ str(i+1)+'.png')
    plt.clf()
    