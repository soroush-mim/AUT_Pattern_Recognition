import numpy as np
import matplotlib.pyplot as plt 



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
#x0 is posetive

TP_bayes_cost = 0
TP_bayes = 0

for i in x0:
    if bayes(i)>=np.log(3):
        TP_bayes_cost +=1
    
    if bayes(i)>=0:
        TP_bayes +=1
        
        
recall_cost = TP_bayes_cost/2000
recall_bayes = TP_bayes/2000

FP_cost = 0
FP_bayes = 0

for i in x1:
    if bayes(i)>=np.log(3):
        FP_cost+=1
    
    if bayes(i)>=0:
        FP_bayes +=1
        
precision_cost = TP_bayes_cost / (TP_bayes_cost + FP_cost)
precision_bayes = TP_bayes/(TP_bayes + FP_bayes)

fscore_cost = 2*precision_cost*recall_cost/(precision_cost+recall_cost)
fscore_bayes = 2*precision_bayes*recall_bayes/(precision_bayes+recall_bayes)

print('bayes with cost F score: : ',fscore_cost)
print('bayes F score: ',fscore_bayes)
    

# bayes with cost F score: :  0.6552534407519301
# bayes F score:  0.6606666666666667

