import numpy as np
import matplotlib.pyplot as plt 

p = [-0.99,-0.5,0.5,0.99]
d2 = [4,9,16]

x1 = np.linspace(-50, 50, 5000) 
x2 = np.linspace(-50, 50, 5000)

x_1, x_2 = np.meshgrid(x1, x2) 

for i in p:
    for d in d2:
        #contour plot
        d22 = (1/(4-4*i**2))*((4*x_1 - 2*i*x_2 * 6 * i - 8)*(x_1 - 2) + (-2*i*x_1+x_2-3+4*i)*(x_2 - 3)) - d
        plt.contourf(x_1, x_2, d22)  
        plt.title("d^2 = " + str(d) + ' p= ' + str(i))
        plt.savefig('5e'+str(i)+str(d)+'.png')
        plt.clf()

    



    