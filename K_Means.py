import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

data = pd.read_csv('wine-clustering.csv')

X = data.iloc[:, :].values
km = KMeans(n_clusters=6)
y_means = km.fit_predict(X)

plt.title('Clustering')

plt.scatter(X[y_means == 0, 0], X[y_means == 0, 1], color='black')
plt.scatter(X[y_means == 1, 0], X[y_means == 1, 1], color='orange')
plt.scatter(X[y_means == 2, 0], X[y_means == 2, 1], color='purple')
plt.scatter(X[y_means == 3, 0], X[y_means == 3, 1], color='cyan')
plt.scatter(X[y_means == 4, 0], X[y_means == 4, 1], color='gray')
plt.scatter(X[y_means == 5, 0], X[y_means == 5, 1], color='brown')