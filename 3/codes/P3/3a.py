import pandas as pd
import numpy as np

def accuracy(y_true , y_predicted):
    """caculate num of correctly predicted samples / num of all samples
    
    class of samples should be 0 or 1
    Args:
        y_true ([numpy array n*1]): [true value of labels]
        y_predicted ([numpy array n*1]): [predicted value of labels]

    Returns:
        [float]: [accuracy of prediction]
    """
    
    difference = y_true - y_predicted
    return np.count_nonzero(difference == 0) / difference.shape[0]

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

#testing different features
for i,j in [(1,2),(1,3),(1,4),(2,3),(2,4),(3,4)]:
    x_train_new = x_train[: , [i-1,j-1]]
    x_test_new = x_test[: , [i-1,j-1]]
    y_pre = KNN(x_train_new , y_train ,x_test_new , K=1)
    acc = accuracy(y_test,y_pre)
    print('Features: ' ,i , ' and ' , j , '   accuracy: ' , acc)