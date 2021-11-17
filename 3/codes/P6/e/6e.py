import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

#reading data
data = pd.read_csv('penguins.csv')
#selecting columns that we want
data = data [['species' , 'bill_length_mm','bill_depth_mm']]
#shuffling data
data = data.sample(frac=1).reset_index(drop=True)

data['species']= data['species'].astype('category')
#dropping null values
data = data.dropna()
#changing categorical variable to numerical
data['species']= data['species'].cat.codes
#Ade = 0   Gentoo = 2  Chin = 1
#splitting target and features
target = data['species']
features = data[['bill_length_mm','bill_depth_mm']]
#changing terget and faetures type to numpy array
target = target.to_numpy()
features = features.to_numpy()
#splitting train and test sets
y_train = target[:300]
y_test = target[300:]
x_train = features[:300]
x_test = features[300:]
#selecting 2 cetegories(Gentoo and adelie)
x_train_a = x_train[y_train != 1]
y_train_a = y_train[y_train != 1]
x_test_a = x_test[y_test != 1]
y_test_a = y_test[y_test != 1]
#normalization
#add a column of ones to features
x_train_a = np.append(np.ones((x_train_a.shape[0],1)) , x_train_a , axis = 1)
x_test_a = np.append(np.ones((x_test_a.shape[0],1)) , x_test_a , axis = 1)
#featuers of samples that are from class 2 * -1
x_train_a[y_train_a == 2] = x_train_a[y_train_a == 2]*-1

#a is Weights matrix
a = np.zeros((3,1))
#perceptron algorithm
for i in range(2500):
    temp =  x_train_a @ a
    #Y_m is misclassified samples with weights a
    Y_m = x_train_a[(temp <= 0).ravel() ]
    #updating weights
    a = a + .000004 * (np.sum(Y_m , axis = 0).reshape(3,1))
    #check for convergence
    if -(np.sum(temp[temp < 0]))<0.2 and i!=0:
        print('number of iterations needed for convergence: ' , i)
        break
    


x_train_a[y_train_a == 2] = x_train_a[y_train_a == 2]*-1

#plotting data distribution and decision boundaries
#samples that are from class gentoo
Gentoo = features[target == 2]
#samples that are from class adelie
Adelie = features[target == 0]
#plotting samples
plt.scatter(Gentoo[:,0] , Gentoo[:,1] , label = 'Gentoo')
plt.scatter(Adelie[:,0] , Adelie[:,1] , label = 'Adelie')
#plotting decision boundary
x = np.linspace(30.,65.)
plt.plot(x,-a[0]/a[2] + x*(-a[1]/a[2]) , label = 'perceptron')
plt.legend()
plt.savefig('6e')
plt.clf()


#calculating results of classification with graadian decent
result_GD = x_test_a @ a
result_GD[result_GD>0] = 0
result_GD[result_GD<0] = 2

#calculating accuracy

y_test_a = y_test_a.reshape(y_test_a.shape[0] , 1)
acc_GD = 1 - np.sum(np.abs(y_test_a - result_GD))/result_GD.shape[0]

print('accuracy of perceptron: ' , acc_GD)