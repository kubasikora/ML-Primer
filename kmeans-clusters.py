import random
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pandas as pd 
from sklearn.preprocessing import StandardScaler

cust_df = pd.read_csv("./data/customer_segmentation.csv")
print(cust_df.head())

df = cust_df.drop('Address', axis=1)

X = df.values[:, 1:]
X = np.nan_to_num(X)
Clus_dataSet = StandardScaler().fit_transform(X)

clusterNum = 3
k_means = KMeans(init = "k-means++", n_clusters = clusterNum, n_init = 12)
k_means.fit(X)
labels = k_means.labels_
print(labels)

df["Clus_km"] = labels
print(df.groupby('Clus_km').mean())

area = np.pi * ( X[:, 1])**2  
plt.scatter(X[:, 0], X[:, 3], s=area, c=labels.astype(np.float), alpha=0.5)
plt.xlabel('Age', fontsize=18)
plt.ylabel('Income', fontsize=16)

plt.show()