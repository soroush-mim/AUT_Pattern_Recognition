import pandas as pd
import numpy as np

def pca(data , n_component):
    """performs PCA on a given dataset and return reduced data

    Args:
        data (m*n numpy array): [m samples that each has n faetures]
        n_component ([int]): [num of pca components to keep]

    Returns:
        [m * n_component numpy array]: [reduced data]
    """    
    # computing mean of data set
    mu = data.mean(axis = 0)
    mu = mu.reshape(data.shape[1] , 1).T
    #subtracting mean from data samples
    data = data - mu
    #computing scatter matrix
    S = np.cov(data.T)*(data.shape[0]-1)
    #computing eigen values and eigen  vectors of scatter matrix
    eigen_values , eigen_vecs = np.linalg.eig(S)
    #selecting eigen vectors corresponding to first biggest n_component
    ind = eigen_values.argsort()[-n_component:][::-1]
    E = eigen_vecs[ : , ind ]
    #computing reduced data
    reduced_data = E.T @ data.T
    return reduced_data.T

#reading data
df = pd.read_csv('doughs.dat' , sep = ' ')
data = df.drop(['Restaurant'] , axis = 1).to_numpy()

reduced_data = pca(data , 3)

print(reduced_data)

