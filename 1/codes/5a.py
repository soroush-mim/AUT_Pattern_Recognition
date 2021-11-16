import numpy as np
import matplotlib.pyplot as plt 


mu = 5
sigma = [1,2,3]

for i in sigma:
    s = np.random.normal(mu, i, 500) #generate samples
    plt.title('sigma= '+ str(i))
    plt.plot(s, np.zeros_like(s), 'o') #plotting samples
    plt.savefig('samples' + str(i) + '.png')
    plt.clf()
    
    plt.hist(s, 30, density=True) # plotting histogram
    plt.title('sigma= '+ str(i))
    plt.savefig('hist' + str(i) + '.png')
    plt.clf()