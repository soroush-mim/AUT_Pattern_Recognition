import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#usefull functions
def jaccard_dist(a , b):
    """calculate jaccard dist between 2 sets

    Args:
        a ([set])
        b ([set]) 

    Returns:
        [int]: [jaccard dist between set a and set b]
    """    
    jaccard_index = len(a.intersection(b)) / len(a.union(b))
    return 1 - jaccard_index

def assign_cluster(data , centroids):
    """for point data calculate which centroid it belongs to

    Args:
        data ([set]): [set of words of a tweet]
        centroids ([dict]): [centriods id's and their set of words]

    Returns:
        [int]: [it's the id of centroid which data belongs to]
    """    
    min_dist = 1
    for key in centroids.keys():
        dist = jaccard_dist(data , centroids[key])
        if dist < min_dist:
            min_dist = dist
            assign = key
    return assign

def df_to_dict(df):
    """ a function for changing data to a dict with this format:
        every key is an id of a tweet and for every key we have
        a set which it has unique words of it's tweet

    Args:
        df ([dataframe]): [ids and tweets]

    Returns:
        [dict]
    """    
    df.set_index("id", drop=True, inplace=True)
    df = df.to_dict(orient="index")
    for i in df.keys():
        df[i] = set(df[i]['text'].split())
    return df

def new_centroid(cluster):
    """for a cluster calculates it's centriod

    Args:
        cluster ([dict]): [it's a dict of id's as keys and their set of words of corresponding tweet
                            which these id's belongs to a single cluster]

    Returns:
        [int]: [new centroid id]
    """    
    min_dist = 10**4
    for i in cluster.keys():
        dist = 0
        for j in cluster.keys():
            dist+=jaccard_dist(cluster[i] , cluster[j])
        if dist < min_dist:
            min_dist = dist
            centroid = i
    return centroid

def initial_seeds(data , k):
    """this function works based on initialization of kmeans++
        The first seed is randomly selected from the set of tweets.
        Then, for each tweet t, the distance between t and the nearest seed that has already
        been chosen is computed. Another seed is then chosen with probability of
        its distance squared among the sum of all distances squared.
        These steps are computed until k seeds have been selected.

    Args:
        data ([dict]): [a dict with tweet id's as keys and their words as values]
        k ([int]): [number of centroids that we want]

    Returns:
        [list]: [id's of tweets that the function choose to be centroid]
    """    
    centroids = []
    centroids.append(np.random.choice(list(data.keys())))
    
    while(len(centroids) < k):
        dists_from_near_seed = {}
        probs = {}
        for key in data.keys():
            dists_from_near_seed[key] = min([jaccard_dist(data[key] , data[centroid]) for centroid in centroids])
        for key in data.keys():
            probs[key] = dists_from_near_seed[key]**2 / (sum([dists_from_near_seed[i]**2 for i in dists_from_near_seed.keys()]))

        centroids.append(np.random.choice(a = list(probs.keys()) , p = [probs[key] for key in probs.keys()]))
    return centroids

#reading data and centroids
df = pd.read_json('tweets.json', lines=True)
data = df[['text' , 'id']]
data = df_to_dict(data)

centroids = initial_seeds(data , k = 25)


#clustring
converge = False
while(not converge):
    result = {}
    #assign each tweet to it's nearest centroid and saving them in result
    for i in data.keys():
        cluster = assign_cluster(data[i] , {centroid:data[centroid] for centroid in centroids})
        
        if cluster not in result.keys():
            result[cluster] = []
            
        result[cluster].append(i)
    
    centroids_new = []
    
    #calculating new centroid for each cluster
    for i in result.keys():
        cluster_ids = result[i]
        cluster_ids.append(i)
        
        centroids_new.append(new_centroid({j:data[j] for j in cluster_ids}))
    
    #checking converge 
    if set(centroids_new) == set(centroids):
        converge = True
    centroids = centroids_new
    
#writing clusters in a file
with open("result_b.txt", "w") as file:
    for i in result.keys():
        file.write(str(i) + ' : ')
        for j in result[i]:
            file.write(str(j) + ' ')
        file.write('\n')