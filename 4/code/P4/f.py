import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt

#reading dataset
df = pd.read_csv('1KGP.txt', delim_whitespace=True , header = None)
df[2] = df[2].astype("category")

#computing Y matrix
Y = df.drop([0,1,2] , axis = 1)
Y = Y.apply(lambda x: x!=x.mode()[0]).to_numpy()
Y = Y.astype('int')
#changing mean to zero
Y = Y - Y.mean(axis = 0).reshape(10101,1).T

#performing pca on Y
pca = PCA(n_components=3)
Y_reduced = pca.fit_transform(Y)

#plotting nucleobase index versus the absolute value of the third principal component
fig, ax = plt.subplots()
scatter = ax.scatter(range(10101) , np.abs(pca.components_[2]) )
plt.xlabel('index')
plt.ylabel('abs of third pca')
plt.savefig('4f')