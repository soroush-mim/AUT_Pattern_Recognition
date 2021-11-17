import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

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

#plotting heatmap of cov matrix
sns_plot = sns.heatmap(data.corr())
sns_plot.get_figure().savefig("6a1.png")

#plotting data in ALSFRS_slope vs other featurs spaces
fig, axs = plt.subplots(2, 5 , figsize=(12, 5))
temp = 0
for i in range(2):
    for j in range(5):
        axs[i,j].scatter(data['ALSFRS_slope'], data[columns[temp]],marker='.' )
        axs[i,j].set_title(columns[temp])
        temp+=1
        
for j in range(5,10):
    axs.flat[j].set( xlabel='ALSFRS_slope')
    
for ax in fig.get_axes():
    ax.label_outer()
    
plt.savefig('6a2')