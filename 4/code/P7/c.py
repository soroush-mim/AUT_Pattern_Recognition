import numpy as np
import idx2numpy
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

#reading data
images = idx2numpy.convert_from_file('t10k-images-idx3-ubyte')
labels = idx2numpy.convert_from_file('t10k-labels-idx1-ubyte')
labels = labels.reshape(10000,1)
images = images.reshape(10000,28*28)

#performing PCA
pca = PCA(n_components=2)
pca_reduced = pca.fit_transform(images)

#performing kmeans and plotting clusters
for k in [4,7,10]:
    kmeans = KMeans(n_clusters=k,init = 'random',n_init=1).fit(pca_reduced)
    fig, ax = plt.subplots()
    scatter = ax.scatter(pca_reduced[:, 0], pca_reduced[:, 1] ,c=kmeans.labels_, cmap=plt.cm.tab10,edgecolor='k' , s = 30)
    handles = scatter.legend_elements()[0]
    plt.xlabel('first pca')
    plt.ylabel('second pca')
    plt.savefig('7c,K= '+str(k))
    plt.clf()


