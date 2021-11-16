import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd 

data = pd.read_csv('b.csv')

x1 = data['X'].to_numpy().reshape(len(data),1)
x2 = data['Y'].to_numpy().reshape(len(data),1)


m1 = np.sum(x1) / 500
m2 = np.sum(x2) / 500

print('m1= ',m1,' m2=' , m2)

cov11 = np.sum(np.power((x1 - m1),2))/500
cov22 = np.sum(np.power((x2 - m2),2))/500
cov12 = np.dot((x1 - m1).T , (x2 - m2))/500

print('cov11= ',cov11,' cov22 = ' , cov22 ,' cov12=cov21= ', float(cov12))

    



    