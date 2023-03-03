import pandas as pd
from sklearn.cluster import KMeans

# Load the data from the PCA-transformed CSV file
data_pca = pd.read_csv("mfcc_data_pca.csv")

# Separate the features (principal components) from the labels
X = data_pca.iloc[:, :-1].values

# Perform K-Means clustering with 2 clusters
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(X)

# Add the cluster labels to the data frame
data_pca["cluster"] = kmeans.labels_

# Save the data frame to a new CSV file
data_pca.to_csv("mfcc_data_clusters_k2.csv", index=False)
