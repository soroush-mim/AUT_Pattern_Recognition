from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
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



def knn_model_selection(dist_type = 'minkowski'):
    train , test , valid = prepare_data()
    train_acc = []
    valid_acc =[]
    val_acc = 0
    best_k = 0
    for i in range(1,21):
        print('k: ',i)
        #initializing model
        neigh = KNeighborsClassifier(n_neighbors=i , metric=dist_type)
        #model training
        neigh.fit(train[: , :-1], train[: , -1])
        train_er = 1-neigh.score(train[: , :-1], train[: , -1])
        print('train error : ',train_er)
        train_acc.append(1-train_er)
        val_er = 1-neigh.score(valid[: , :-1], valid[: , -1])
        print('validation error : ',val_er)
        valid_acc.append(1-val_er)
        if (1-val_er) > val_acc:
            val_acc = 1-val_er
            best_k = i
        
    print('Based on validation accuracy best k is : ' , best_k)
    return train_acc , valid_acc , best_k


train_acc , valid_acc , best_k = knn_model_selection()
plt.plot(range(1,21),train_acc , label = 'train accuracy' )
plt.plot(range(1,21),valid_acc , label = 'validation accuracy')
plt.legend()
plt.xlabel('K')
plt.ylabel('accuracy')
plt.savefig('4b.png')