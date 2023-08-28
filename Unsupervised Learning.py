# -*- coding: utf-8 -*-
"""
Created on Sat Aug  5 18:49:29 2023

@author: SergeyHSE
"""

#########################################################################################
#  Here we will use different algorithms of the "Unsupervised Learning" method on       #
#  the classification data of plant leaves, as well as on the data from sklearn library.#
#########################################################################################

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

path = r"your path"
path = path.replace('\\', '/')
data = pd.read_csv(path, header=None)

#The first column is the answer, let's put it in a separate variable

X, y_name = np.array(data.iloc[:, 1:]), data.iloc[:, 0]

#The target variable takes a text value.
#Use the Lame Encoder from sklearn to encode the text variable y_name
#and save the resulting values to the variable y

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

#Using the PCA method, reduce the dimension of the feature space to two.

from sklearn.decomposition import PCA

pca = PCA(n_components=2, random_state=0)
pca_fit = pca.fit_transform(X)

plt.figure(figsize=(16, 16))
plt.scatter(pca_fit[:, 0], pca_fit[:, 1], c=y[y<15])
plt.colorbar()
plt.title('PCA CLASTERS')
plt.show()

pca_fit[0]

#Select objects that correspond to values from 0 to 14 of the target variable y.
#Draw the selected objects in a two-dimensional feature space using the scatter
#method from matplotlib.pyplot. To display objects of different classes in
#different colors, pass c = y[y<15] to the scatter method.

from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, random_state=0)
tsne_fit = tsne.fit_transform(X)

plt.figure(figsize=(16, 16))
plt.scatter(tsne_fit[:, 0], tsne_fit[:, 1], c=y[y<15])
plt.title('t-SNE Clasters')
plt.colorbar()
plt.show()

#we are gonna make the same operations by TNSE

tsne_fit[0]

tsne3 = TSNE(n_components=3, random_state=0)
tsne_fit = tsne3.fit_transform(X)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
plt.scatter(tsne_fit[:, 0], tsne_fit[:, 1], tsne_fit[:, 2], c=y)
plt.show()

##########################################################################
#                   Write class for K-means and realize it              ##
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

#Import dataset

from sklearn import datasets
n_samples = 1000

noisy_blobs = datasets.make_blobs(n_samples=n_samples,
                             cluster_std=[1.0, 3.0, 0.5],
                             random_state=0)
X, y = noisy_blobs

#Cluster noisy_blobs objects using MyKMeans, use hyperparameters n_clusters=3, n_iters=100.

kmeans1 = MyKMeans()
kmeans_fit1 = kmeans1.fit(X)
kmean_pred1 = kmeans1.predict(X)

kmean_pred1[1]

plt.scatter(X[:, 0], X[:, 1], c=kmean_pred1)
plt.title('K-means')
plt.show()

#Cluster noisy_blobs objects, use hyperparameters n_clusters=3, n_items = 5.

kmeans = MyKMeans(n_iters=5)
kmeans_fit = kmeans.fit(X)
kmean_pred = kmeans.predict(X)
kmean_pred[1]

plt.scatter(X[:, 0], X[:, 1], c=kmean_pred)
plt.title('K-means')
plt.show()

#Calculate how many objects have the label of the predicted cluster
#changed when changing the n_iters hyperparameter from 5 to 100

kmean_pred[1]

np.array_equal(kmean_pred1, kmean_pred)
array_nonequal = kmean_pred != kmean_pred1
count_mismatches = np.count_nonzero(array_nonequal)
count_mismatches

#Determine how many iterations the algorithm converged on objects objects noisy_blobs

array_nonequal1 = kmean_pred != kmean_pred1
mismathes = np.count_nonzero(array_nonequal1)
mismathes

#Cluster noisy_blobs objects using HDDSCAN
#Calculate the resulting number of clusters
#Calculate objects classified as noise

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
