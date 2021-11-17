import numpy as np
import idx2numpy
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt

#reading data
images = idx2numpy.convert_from_file('t10k-images-idx3-ubyte')
labels = idx2numpy.convert_from_file('t10k-labels-idx1-ubyte')
labels = labels.reshape(10000,1)
images = images.reshape(10000,28*28)

#performing PCA
pca = PCA(n_components=20)
pca_reduced = pca.fit_transform(images)

print('eigenvalues: ' , pca.explained_variance_)

#plotting on first 2 and 3 PCs
fig, ax = plt.subplots()
scatter = ax.scatter(pca_reduced[:, 0], pca_reduced[:, 1] ,c=labels, cmap=plt.cm.tab10,edgecolor='k' , s = 30)
handles = scatter.legend_elements()[0]
labels_unique = np.unique(labels)
legend1 = ax.legend(handles, labels_unique,loc="upper right", title="category" , bbox_to_anchor=(1.15, 1))
plt.xlabel('first pca')
plt.ylabel('second pca')
ax.add_artist(legend1)
plt.savefig('a1.png')
plt.clf()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(pca_reduced[: , 0], pca_reduced[: , 1], pca_reduced[: , 2], marker='o' , c=labels, cmap=plt.cm.tab10,edgecolor='k'  ,s = 20)
handles = scatter.legend_elements()[0]
labels_unique = np.unique(labels)
legend1 = ax.legend(handles, labels_unique,loc="upper right", title="category" , bbox_to_anchor=(1.35, 1))
plt.xlabel('first pca')
plt.ylabel('second pca')
ax.set_zlabel('third pca')
ax.add_artist(legend1)
plt.savefig('a2.png')
plt.clf()