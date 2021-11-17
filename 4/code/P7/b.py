import numpy as np
import idx2numpy
from matplotlib import pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

#reading data
images = idx2numpy.convert_from_file('t10k-images-idx3-ubyte')
labels = idx2numpy.convert_from_file('t10k-labels-idx1-ubyte')
labels = labels.reshape(10000,1)
images = images.reshape(10000,28*28)

#performing LDA
lda = LinearDiscriminantAnalysis(n_components=2)
lda_reduced = lda.fit(images, np.ravel(labels)).transform(images)

#plotting data on first 2 LDs
fig, ax = plt.subplots()
scatter = ax.scatter(lda_reduced[:, 0], lda_reduced[:, 1] ,c=labels, cmap=plt.cm.tab10,edgecolor='k' , s = 30)
handles = scatter.legend_elements()[0]
labels_unique = np.unique(labels)
legend1 = ax.legend(handles, labels_unique,loc="upper right", title="category", bbox_to_anchor=(1.13, 1))
plt.xlabel('first LD')
plt.ylabel('second LD')
ax.add_artist(legend1)
plt.savefig('7b.png')