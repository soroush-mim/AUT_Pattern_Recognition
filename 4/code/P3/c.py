import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

#adding a column to dataset to check if sample is from naples or not
df['Naples'] = df.apply(lambda x: 0 if x['Restaurant'] in [1,2,3,4] else 1 , axis = 1)

reduced_data = pca(data , 3)
print(reduced_data)
#plotting 3d scatter plot for first 3 PCAs
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(reduced_data[: , 0], reduced_data[: , 1], reduced_data[: , 2], marker='o' , c=df['Naples'], cmap=plt.cm.Set1,edgecolor='k')
plt.xlabel('first pca')
plt.ylabel('sec pca')
ax.set_zlabel('third pca')
handles = scatter.legend_elements()[0]
labels = list(df['Naples'].unique())
legend1 = ax.legend(handles, labels, title="Naples")
plt.savefig('3c-3D.png')
plt.clf()

#plotting scatter plots for different pairs of first 3 PCAs
for i , j in [[0,1],[0,2],[1,2]]:
    fig, ax = plt.subplots()
    scatter = ax.scatter(reduced_data[: , i] ,reduced_data[: , j] ,c=df['Naples'], cmap=plt.cm.Set1,edgecolor='k')
    handles = scatter.legend_elements()[0]
    labels = list(df['Naples'].unique())
    legend1 = ax.legend(handles, labels,loc="upper left", title="Naples")
    plt.xlabel('pca num: '+str(i+1))
    plt.ylabel('pca num: '+str(j+1))
    ax.add_artist(legend1)
    plt.savefig('3c-2D' + str(i+1) + str(j+1)+'png')
    plt.clf()