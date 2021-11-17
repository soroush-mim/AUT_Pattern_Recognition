import pandas as pd
import numpy as np
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

kmeans = KMeans(n_clusters = 3,init = 'k-means++').fit(train_data)

#calculating number of samples in each cluster
cluster1_num = np.count_nonzero(kmeans.labels_ == 0)
cluster2_num = np.count_nonzero(kmeans.labels_ == 1)
cluster3_num = np.count_nonzero(kmeans.labels_ == 2)

#plotting bar chart
fig, ax = plt.subplots()
plt.bar([0,1 , 2], [cluster1_num , cluster2_num , cluster3_num] , width = 0.3)
plt.xticks([0,1,2], ('cluster 0', 'cluster 1' , 'cluster 3'))
plt.ylabel('number of samples')
plt.savefig('6d')