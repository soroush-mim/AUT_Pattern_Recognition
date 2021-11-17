import numpy as np
import idx2numpy
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
import cv2

#reading data
images = idx2numpy.convert_from_file('t10k-images-idx3-ubyte')
labels = idx2numpy.convert_from_file('t10k-labels-idx1-ubyte')
labels = labels.reshape(10000,1)
images = images.reshape(10000,28*28)

#performing PCA
pca = PCA(n_components=200)
pca_reduced = pca.fit_transform(images)

#calculating min number of PCs for 95% information
var = pca.explained_variance_ratio_[0]
i = 1
while(var < .95):
    var += pca.explained_variance_ratio_[i]
    i+=1

print('enough numbers of PCs for capturnig 95% of information: ',i+1)

#performing pca with i component
pca = PCA(n_components=i+1)
pca_reduced = pca.fit_transform(images)
#reversing reduced data to images
inverse_images = pca.inverse_transform(pca_reduced)

#saving 3 random image and their reduced versions
fig, axs = plt.subplots(3, 2 , figsize=(9, 9))
for i in range(3):
    #select 10 indexes randomly from cluster i
    random_indxs = np.random.choice(range(10000))
    axs[i,0].imshow(images[random_indxs].reshape(28,28), cmap='gray', vmin=0, vmax=255 )
    axs[i,1].imshow(inverse_images[random_indxs].reshape(28,28), cmap='gray', vmin=0, vmax=255 )
for j in range(3):
    axs[j,0].set( xlabel='original images')
    axs[j,1].set( xlabel='reduced images')
for ax in fig.get_axes():
    ax.label_outer()
plt.savefig('7e.png')
