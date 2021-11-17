import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def KNN( X_train , Y_train , X_test , K = 1 , dist_type = 'Euc' ):
    """this functions execute KNN algorithm

    Args:
        X_train ([numpy array m*n]): [features for training]
        Y_train ([numpy array m*1]): [targets for training]
        X_test ([numpy array z*n]): [features for predicting]
        K (int, optional): [num of neighbours]. Defaults to 1.
        dist_type (str, optional): [type of distance, it should be one of these: 'Euc','Mnhtn','Cosin']. Defaults to 'Euc'.

    Returns:
        [numpy array z*1]: [predicted values for X_test]
    """
    
    #for all types of distances the dists is a z*m matrix which z is num of test points and m is num of train points
    #and in the ith row of dists we have distances for ith test point from all of train points
    if dist_type == 'Euc':
        #calculating L2 norm for each test point from each train point , L2 norm is the Euclidean distance
        dists = np.linalg.norm(X_test[:,np.newaxis]-X_train ,axis = 2)
    
    if dist_type == 'Mnhtn':
        #calculating manhatan distance for each test point from each train point
        dists = np.sum(np.absolute(X_test[:,np.newaxis]-X_train) , axis = 2)
        
    if dist_type == 'Cosin':
        #calculating cosin distance for each test point from each train point
        norm_xtrain = np.linalg.norm(X_train,axis = 1 ).reshape(X_train.shape[0],1)
        norm_xtest = np.linalg.norm(X_test,axis = 1 ).reshape(X_test.shape[0],1)
        norms = norm_xtest @ norm_xtrain.T
        dists = 1 - ((X_test @ X_train.T) / norms)
    
    #choosing min distances
    min_dists_indices = np.argpartition(dists,K,axis = 1)[:,:K]
    #calculating prediction labels
    Y_sum_min_distances = np.sum(Y_train[min_dists_indices] , axis = 1)
    y_pred = Y_sum_min_distances > (K-1)/2
    
    return y_pred.astype('int64')

#data preprocessing
data = pd.read_csv('data_banknote_authentication.txt', sep=",", header=None)
data.columns = ['f1' , 'f2' ,'f3'  , 'f4' ,'output']
#shuffling
data = data.sample(frac=1).reset_index(drop=True)
#splitting features and targets
X = data.drop('output' , axis = 1).to_numpy()
Y = data['output'].to_numpy()
#splitting train and test data
x_train = X[:500]
y_train = Y[:500]
x_test = X[500:]
y_test = Y[500:]

#seting up colors
cmap_light = ListedColormap(['#FFAAAA','#AAAAFF', '#AAFFAA'])

xx , yy = np.meshgrid(np.arange(-8, 8, .05),
                     np.arange(-15, 15, .05))

#executing 1NN
y_pred= KNN(x_train[: , [0,1]] , y_train ,np.c_[xx.ravel(), yy.ravel()] , K = 1)
y_pred=y_pred.reshape(xx.shape)

plt.figure(figsize=(10,8))
plt.pcolormesh(xx, yy, y_pred,cmap=cmap_light,shading='auto')
plt.savefig('3b.png')
plt.clf()

y_pred= KNN(x_train[: , [0,1]] , y_train ,np.c_[xx.ravel(), yy.ravel()] , K = 3)
y_pred=y_pred.reshape(xx.shape)
plt.figure(figsize=(10,8))
plt.pcolormesh(xx, yy, y_pred,cmap=cmap_light,shading='auto')
plt.savefig('3d.png')