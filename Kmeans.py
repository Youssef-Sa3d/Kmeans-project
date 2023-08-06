import matplotlib.pyplot as plt
from sklearn.cluster import KMeans



x = [3,3,3,4,4,5,5,7,7,8,9,1,2,6,7,8,-9,2,-6,2]
y = [4,6,8,5,7,1,5,3,5,5,-5,3,-7,-9,-1,5,-4,0,9,-1]
p = list(zip(x, y))


k = int(input("K = "))

plt.scatter(x, y)
plt.show()
kmeans = KMeans(n_clusters=k)
kmeans.fit_predict(p)
clusters = kmeans.cluster_centers_
print(clusters)
plt.scatter(x, y , c = kmeans.labels_)
plt.title('Clustering')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend()
plt.show()

    


