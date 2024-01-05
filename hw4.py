import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram


def load_data(filepath):
    dictionaries = []
    with open(filepath, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        # create a dictionary for each row of the dataset
        for row in reader:
            dictionaries.append(dict(row))
    return dictionaries


def calc_features(row):
    feature_order = ['Population', 'Net migration', 'GDP ($ per capita)', 'Literacy (%)', 'Phones (per 1000)', 'Infant mortality (per 1000 births)']
    
    # get the features in the required order and convert them to float
    features = [float(row[feature]) for feature in feature_order]
    return np.array(features, dtype=np.float64)

def hac(features):
    # assign each of the features one cluster number
    clusters = {i:[features[i]] for i in range(len(features))}

    # create the initial distance matrix (twice the size of features, so we can store the new clusters' distances)
    distances = np.zeros((2*len(features),2*len(features)))

    for i in range(len(features)):
        for j in range(i+1,len(features)):
            distances[i][j] = np.linalg.norm(clusters[i][0]-clusters[j][0])
    
    # make the result matrix
    res = np.zeros((len(features)-1,4))

    old_clusters = set()
    # repeat the process n-1 times
    for row in range(len(features)-1):
        # find the smallest (complete-linkage) distance
        mins = (float('inf'),float('inf'),float('inf'))
        for i in range(len(features)+row):
            for j in range(i+1,len(features)+row):
                if i not in old_clusters and j not in old_clusters:
                    mins = min(mins,(distances[i][j],i,j))

        i,j = mins[1],mins[2]

        # remove the old clusters 
        old_clusters.add(i)
        old_clusters.add(j)

        # merge all of clusters i and j items into the new cluster 
        new_cluster = len(features)+row
        clusters.update({new_cluster:clusters[i] + clusters[j]})

        # update the distances in the matrix
        for x in clusters:        
            if x!=new_cluster:
                distances[x][new_cluster] = max(distances[min(i,x)][max(i,x)],distances[min(j,x)][max(j,x)])
        
        # remove the old clusters from dict to save memory
        clusters.pop(i)
        clusters.pop(j)

        # add the indices to result
        res[row][0] = i
        res[row][1] = j 
        res[row][2] = mins[0]
        res[row][3] = len(clusters[new_cluster])

    return res


def fig_hac(Z, names):  
    fig = plt.figure()
    
    # generate the dendrogram
    dendrogram(Z, labels=names, leaf_rotation=90)
    plt.tight_layout()
    plt.show()
    
    return fig

def normalize_features(features):  
    feature_matrix = np.array(features)
    
    # compute mean and standard deviation for each statistic
    means = np.mean(feature_matrix, axis=0)
    std_devs = np.std(feature_matrix, axis=0)
    normalized_features = (feature_matrix - means) / std_devs
    
    # convert the normalized feature matrix back to a list of arrays
    return [np.array(row, dtype=np.float64) for row in normalized_features]


data = load_data("countries.csv")
country_names = [row["Country"] for row in data] 
features = [calc_features(row) for row in data] 
features_normalized = normalize_features(features) 
n = 50
Z_raw = hac(features[:n])
Z_normalized = hac(features_normalized[:n]) 
print(Z_normalized)
fig = fig_hac(Z_normalized, country_names[:n]) 
plt.show()
