import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

#reading data
data = pd.read_csv('ALS_train.csv')
data = data.set_index('ID')

#covariance matrix of data
cov = data.corr()
#selecting 10 columns with biggest value of covariance with ALSFRS_slope
cov = cov['ALSFRS_slope'].abs()
cov = cov.sort_values()
columns = cov[-11:].index
data = data[columns]
train_data = data.drop('ALSFRS_slope' , axis = 1)

#elbow method
sse = {}
for k in range(1, 15):
    kmeans = KMeans(n_clusters=k,init = 'random').fit(train_data)
    sse[k] = kmeans.inertia_ # Inertia: Sum of distances of samples to their closest cluster center
plt.figure()
plt.plot(list(sse.keys()), list(sse.values()))
plt.xlabel("Number of cluster")
plt.ylabel("SSE")
plt.savefig('6belbow')
plt.clf()

#silhouette method
sil = []
# dissimilarity would not be defined for a single cluster, thus, minimum number of clusters should be 2
for k in range(2, 15):
  kmeans = KMeans(n_clusters = k,init = 'random').fit(train_data)
  sil.append(silhouette_score(train_data, kmeans.labels_, metric = 'euclidean'))
  
plt.plot(range(2,15) , sil)
plt.xlabel("Number of cluster")
plt.ylabel("silhouette score")
plt.grid()
plt.xticks(range(1,15))
plt.savefig('6bsilhouette')