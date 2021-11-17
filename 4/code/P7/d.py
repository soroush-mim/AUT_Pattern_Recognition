import numpy as np
import idx2numpy
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

#reading data
images = idx2numpy.convert_from_file('t10k-images-idx3-ubyte')
labels = idx2numpy.convert_from_file('t10k-labels-idx1-ubyte')
images = images.reshape(10000,28*28)

#performing PCA
pca = PCA(n_components=2)
pca_reduced = pca.fit_transform(images)

k_4sets = [[0,2,4,6],[1,3],[5,7,9],[8]]
k_7sets = [[0,2,4],[1],[3],[5],[6],[8],[7,9]]
k_10sets = [[i] for i in range(10)]

sets = [k_4sets , k_7sets , k_10sets]
#performing kmeans and plotting clusters
#a loop for different values of k
for set in sets:
    k = len(set)
    #computing mean of each set
    means = [pca_reduced[np.isin(labels , i)].mean(axis=0) for i in set]
    means = np.array(means)
    #performing kmeans
    kmeans = KMeans(n_clusters=k,init = means).fit(pca_reduced)
    fig, ax = plt.subplots()
    scatter = ax.scatter(pca_reduced[:, 0], pca_reduced[:, 1] ,c=kmeans.labels_, cmap=plt.cm.tab10,edgecolor='k' , s = 30)
    handles = scatter.legend_elements()[0]
    plt.xlabel('first pca')
    plt.ylabel('second pca')
    plt.savefig('7d,K= '+str(k))
    plt.clf()
