from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import matplotlib.pyplot as plt
def prepare_data():
    
    vectorizer = CountVectorizer()
    
    #concating fake and real datas and save them in one file
    data = data2 = "" 
    # Reading data from file1 
    with open('clean_fake.txt') as fp: 
        data = fp.read() 
    # Reading data from file2 
    with open('clean_real.txt') as fp: 
        data2 = fp.read() 
    # Merging 2 files To add the data of file2 from next line 
    data += data2 
    #saving new file 
    with open ('clean.txt', 'w') as fp: 
        fp.write(data) 
        
    corpus = open('clean.txt')
    #vectorizing corpus
    X = vectorizer.fit_transform(corpus)
    #changing x type to numpy array
    X = np.array(X.toarray())
    #fake : class 1 , real : class 0 
    target = np.concatenate((np.ones((1298,1)) , np.zeros((1968,1))) , axis = 0)
    #concating targets to X for each row
    X = np.concatenate((X , target),axis = 1)
    #shuffling X
    np.random.shuffle(X)
    #splitting train , test and validation sets
    train = X[:int(len(X)*.7)]
    test = X[int(len(X)*.7) : int(len(X)*.85)]
    valid = X[int(len(X)*.85):]
    
    return train,test,valid

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

def knn_model_selection(dist_type):
    train , test , valid = prepare_data()
    train_acc = []
    valid_acc =[]
    val_acc = 0
    best_k = 0
    for i in range(1,21):
        print('k: ',i)
        y_pred = KNN(train[: , :-1], train[: , -1] ,train[: , :-1] , K = i , dist_type=dist_type )
        train_er = 1-accuracy(train[: , -1] , y_pred)
        print('train error : ',train_er)
        train_acc.append(1-train_er)
        y_pred = KNN(train[: , :-1], train[: , -1] ,valid[: , :-1] , K = i , dist_type=dist_type )
        val_er = 1-accuracy(valid[: , -1] , y_pred)
        print('validation error : ',val_er)
        valid_acc.append(1-val_er)
        if (1-val_er) > val_acc:
            val_acc = 1-val_er
            best_k = i
        
    print('Based on validation accuracy best k is : ' , best_k)
    return train_acc , valid_acc , best_k


train_acc , valid_acc , best_k = knn_model_selection('Cosin')
plt.plot(range(1,21),train_acc , label = 'train accuracy' )
plt.plot(range(1,21),valid_acc , label = 'validation accuracy')
plt.legend()
plt.xlabel('K')
plt.ylabel('accuracy')
plt.savefig('4c.png')