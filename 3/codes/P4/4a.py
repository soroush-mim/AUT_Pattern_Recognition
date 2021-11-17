from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

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