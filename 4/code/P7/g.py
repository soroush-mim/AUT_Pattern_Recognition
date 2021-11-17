import numpy as np
import idx2numpy
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt

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

#caculating a dataframe that each row represent a cluster and each column
#reperesent a class and each cell tell us the percentage of class in cluster
percentage = {}
for i in range(10):
    percentage['cluster' + str(i)] = [int(np.count_nonzero(labels[kmeans.labels_ == i]==k)*100 / np.count_nonzero(kmeans.labels_ == i)) for k in range(10)]
df = pd.DataFrame.from_dict(percentage,orient='index')

#plotting bor plot
ax = df.plot(kind='bar', figsize=(20,7), width=0.8, edgecolor=None)

for p in ax.patches:
    width = p.get_width()
    height = p.get_height()
    x, y = p.get_xy() 
    if height >0:
        ax.annotate(f'{height}', (x + width/2, y + height*1.02), ha='center')
    
plt.legend(labels=df.columns , title = 'classes')
plt.savefig('7g')


    

    
        