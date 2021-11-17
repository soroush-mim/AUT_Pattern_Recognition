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
#normalization
features = (features - features.min())/(features.max() - features.min())
#changing terget and faetures type to numpy array
target = target.to_numpy()
features = features.to_numpy()
#splitting train and test sets
y_train = target[:300]
y_test = target[300:]
x_train = features[:300]
x_test = features[300:]
#selecting 2 cetegories(Gentoo and Chinstrap)
x_train_a = x_train[y_train != 0]
y_train_a = y_train[y_train != 0]
x_test_a = x_test[y_test != 0]
y_test_a = y_test[y_test != 0]
#normalization
#add a column of ones to features
x_train_a = np.append(np.ones((x_train_a.shape[0],1)) , x_train_a , axis = 1)
x_test_a = np.append(np.ones((x_test_a.shape[0],1)) , x_test_a , axis = 1)
#featuers of samples that are from class 2 * -1
x_train_a[y_train_a == 2] = x_train_a[y_train_a == 2]*-1

b = np.ones((x_train_a.shape[0],1))
#gradian decent
#converge_time is a list that store differnt learning rates and their number of itration needed for convergence
converge_time = []
#we start with learning rate = 0.001 and in each step we add 0.0001 to it
learning_rate = .001
#first loop is for testing different learning rates
for j in range(80): 
    a = np.zeros((3,1))
    #this loop is for executing gradient decent
    for i in range(2000):
        a = a - learning_rate * (x_train_a.T @ (x_train_a @ a - b))
        #condition for convergence
        if np.sum(np.power(x_train_a @ a - b , 2)) <42 :
            converge_time.append([learning_rate,i])
            break
    #checking if this learning rate didnt converge
    if np.sum(np.power(x_train_a @ a - b , 2)) > 42:
        min_fail = learning_rate
        break
    learning_rate+=.0001
    
#plotting learning rate vs number of iterations needed for convergence
print('minimum learning rate that fails to lead convergences' , min_fail)
plt.plot([x[0] for x in converge_time ] , [x[1] for  x in converge_time])
plt.xlabel('learning rate')
plt.ylabel('number of iteretions for convergence')
plt.savefig('6d2.png')

