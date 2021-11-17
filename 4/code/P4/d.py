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

#plotting data based on 2 first PCAs and population
fig, ax = plt.subplots()
scatter = ax.scatter(Y_reduced[:, 0], Y_reduced[:, 2] ,c=df[2].cat.codes, cmap=plt.cm.Set1,edgecolor='k')
handles = scatter.legend_elements()[0]
labels = list(df[2].cat.categories)
legend1 = ax.legend(handles, labels,loc="lower right", title="population")
plt.xlabel('first pca')
plt.ylabel('third pca')
ax.add_artist(legend1)
plt.savefig('4d-1.png')
plt.clf()

#plotting data based on 2 first PCAs and sex
fig, ax = plt.subplots()
scatter = ax.scatter(Y_reduced[:, 0], Y_reduced[:, 2] ,c=df[1], cmap=plt.cm.Set1,edgecolor='k')
handles = scatter.legend_elements()[0]
labels =  ['male' , 'female']
legend1 = ax.legend(handles, labels,loc="lower right", title="population")
plt.xlabel('first pca')
plt.ylabel('third pca')
ax.add_artist(legend1)
plt.savefig('4d-2.png')
plt.clf()