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
pca = PCA(n_components=185)
pca_reduced = pca.fit_transform(images)

#performing kmeans
kmeans = KMeans(n_clusters=10,init = 'k-means++').fit(pca_reduced)

#plotting 10 samples from each cluster randomly
fig, axs = plt.subplots(10, 10 , figsize=(14, 14))

for i in range(10):
    #select 10 indexes randomly from cluster i
    random_indxs = np.random.choice(np.arange(10000)[kmeans.labels_==i],10)
    
    for j in range(10):
        axs[i,j].imshow(images[random_indxs[j]].reshape(28,28), cmap='gray', vmin=0, vmax=255 )
        
for j in range(0,100,10):
    axs.flat[j].set( ylabel='C'+str(j//10))
    
for ax in fig.get_axes():
    ax.label_outer()
    
plt.savefig('7f.png')

