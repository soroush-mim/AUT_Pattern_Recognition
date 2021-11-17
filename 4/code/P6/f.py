import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram

def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)


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

for linkage in ['complete', 'average', 'single']:
    # Create a subplot with 1 row and 2 columns
    fig,(ax1 ,ax2) = plt.subplots(1, 2)
    fig.set_size_inches(16, 5)

   
    ax1.set_xlim([-1, 1])

    ax1.set_ylim([0, len(train_data) + 4 * 10])


    clusterer = AgglomerativeClustering(linkage = linkage, n_clusters=3)
    cluster_labels = clusterer.fit_predict(train_data)

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(train_data, cluster_labels)
    print("For linkage =", linkage,
          "The average silhouette_score is :", silhouette_avg)

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(train_data, cluster_labels)

    y_lower = 10
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
    for i in range(3):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = \
        sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = plt.cm.nipy_spectral(float(i) / 3)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                        0, ith_cluster_silhouette_values,
                        facecolor=color, edgecolor=color, alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for 3 clusters and linkage: "+linkage)
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-1,-.8,-.6,-.4 , -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1])
    model = AgglomerativeClustering(linkage = linkage,distance_threshold=0, n_clusters=None).fit(train_data)
    model = model.fit(train_data)
    plt.title('Hierarchical Clustering Dendrogram for linkage: '+linkage)
    # plot the top three levels of the dendrogram
    ax2 = plot_dendrogram(model, truncate_mode='level', p=2)
    plt.xlabel("Number of points in node (or index of point if no parenthesis).")
    
    plt.savefig('6f'+linkage)
    plt.clf()
