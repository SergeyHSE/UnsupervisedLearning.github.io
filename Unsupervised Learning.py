# -*- coding: utf-8 -*-
"""
Created on Sat Aug  5 18:49:29 2023

@author: User
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

path = r"C:\Users\User\Documents\pyton-projects\spider\Машинное обучение\Learning without teacher\data_Mar_64.txt"
path = path.replace('\\', '/')
data = pd.read_csv(path, header=None)

X, y_name = np.array(data.iloc[:, 1:]), data.iloc[:, 0]

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
le. fit(y_name)
y = le.transform(y_name)

y.shape
X.shape
unique, value_counts = np.unique(y, return_counts=True)
concat_counts = pd.DataFrame({'unique' : unique, 'counts' : value_counts})
concat_counts
num_rows = 16*15
num_rows

X = X[:240, :]
X.shape
y = y[:240]
y.shape

from sklearn.decomposition import PCA

pca = PCA(n_components=2, random_state=0)
pca_fit = pca.fit_transform(X)

plt.figure(figsize=(16, 16))
plt.scatter(pca_fit[:, 0], pca_fit[:, 1], c=y[y<15])
plt.colorbar()
plt.title('PCA CLASTERS')
plt.show()

pca_fit[0]

from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, random_state=0)
tsne_fit = tsne.fit_transform(X)

plt.figure(figsize=(16, 16))
plt.scatter(tsne_fit[:, 0], tsne_fit[:, 1], c=y[y<15])
plt.title('t-SNE Clasters')
plt.colorbar()
plt.show()

tsne_fit[0]

tsne3 = TSNE(n_components=3, random_state=0)
tsne_fit = tsne3.fit_transform(X)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
plt.scatter(tsne_fit[:, 0], tsne_fit[:, 1], tsne_fit[:, 2], c=y)
plt.show()

##########################################################################
#                             K-means                                   ##
##########################################################################

from sklearn.metrics import pairwise_distances_argmin
  
class MyKMeans():
    def __init__(self, n_clusters=3, n_iters=100):
        self.n_clusters = n_clusters
        self.n_iters = n_iters
    
    def fit(self, X):
        count = 0 
        np.random.seed(0)
        self.centers = np.random.uniform(low=X.min(axis=0),
                                         high=X.max(axis=0),
                                         size=(self.n_clusters, X.shape[1]))

        for it in range(self.n_iters):
            count += 1
            labels = pairwise_distances_argmin(X, self.centers)
            new_centers = np.array([X[labels == i].mean(axis=0) for i in range(self.n_clusters)])

            if np.all(self.centers == new_centers):
                break

            self.centers = new_centers
        print(count)

    def predict(self, X):
        labels = pairwise_distances_argmin(X, self.centers)
        return labels

from sklearn import datasets
n_samples = 1000

noisy_blobs = datasets.make_blobs(n_samples=n_samples,
                             cluster_std=[1.0, 3.0, 0.5],
                             random_state=0)
X, y = noisy_blobs

kmeans1 = MyKMeans()
kmeans_fit1 = kmeans1.fit(X)
kmean_pred1 = kmeans1.predict(X)

kmean_pred1[1]

plt.scatter(X[:, 0], X[:, 1], c=kmean_pred1)
plt.title('K-means')
plt.show()

kmeans = MyKMeans(n_iters=5)
kmeans_fit = kmeans.fit(X)
kmean_pred = kmeans.predict(X)
kmean_pred[1]

plt.scatter(X[:, 0], X[:, 1], c=kmean_pred)
plt.title('K-means')
plt.show()

kmean_pred[1]

np.array_equal(kmean_pred1, kmean_pred)
array_nonequal = kmean_pred != kmean_pred1
count_mismatches = np.count_nonzero(array_nonequal)
count_mismatches

array_nonequal1 = kmean_pred != kmean_pred1
mismathes = np.count_nonzero(array_nonequal1)
mismathes

from sklearn.cluster import DBSCAN

dbscan = DBSCAN(eps=0.5)
dbscan_fit = dbscan.fit(X) 
dbscan_pred = dbscan.fit_predict(X)
dbscan_pred[1]
labels = dbscan_fit.labels_
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)
n_clusters_
n_noise_

#plt.scatter(X[:, 0], X[:, 1], c=dbscan_fit)
#plt.title('DBSCAN')
