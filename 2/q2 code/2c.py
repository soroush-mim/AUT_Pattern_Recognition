import numpy as np
import matplotlib.pyplot as plt 


def MDC (x):
    return 2*x[0] + 5 * x[1] + 0.4 

def bayes(x):
    mu1 = np.array([1, 2])
    mu2 = np.array([-1, -3])
    cov1 = np.array([[1.8, -0.7], [-0.7, 1.8]] )
    cov2 = np.array([[1.5, 0.3], [0.3, 1.5]] )
    
    return -0.5 * ((x-mu1).T)@np.linalg.inv(cov1)@(x-mu1)-0.5*np.log(np.abs(np.linalg.det(cov1))) - (-0.5 * ((x-mu2).T)@np.linalg.inv(cov2)@(x-mu2)-0.5*np.log(np.abs(np.linalg.det(cov2))))

    
mu1 = [1, 2]
mu2 = [-1, -3]
cov1 = [[1.8, -0.7], [-0.7, 1.8]] 
cov2 = [[1.5, 0.3], [0.3, 1.5]] 

x0 = np.random.multivariate_normal(mu1, cov1,1000)
x1 = np.random.multivariate_normal(mu2, cov2, 1000)

mdc_err = 0
bayes_err = 0

for i in x0:
    if MDC(i)<0:
        mdc_err+=1
    
    if bayes(i)<0:
        bayes_err +=1
        
for i in x1:
    if MDC(i)>=0:
        mdc_err+=1
    
    if bayes(i)>=0:
        bayes_err +=1
        
print('MDC error: ',mdc_err/2000)
print('bayes error: ',bayes_err/2000)
    
#MDC error:  0.018
#bayes error:  0.012


