import time
import sys
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt

def unit_norm(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)

gt = time.time
np.random.seed(123)

print('======== Building Dataset ==========')
N = 100
K = 5
D = 784

means = 20 * np.random.rand(K, D) - 10
x = np.vstack([np.random.randn(N, D) + means[i] for i in range(K)])
np.random.shuffle(x)
x = unit_norm(x)
x = x.astype(np.float32)
x = np.require(x, requirements=['A', 'C', 'O', 'W'])
print(x)

print('======== Hierarchical Clustering ==========')
t = gt()

# Compute condensed pairwise cosine distance matrix
distance_matrix = pdist(x, metric='cosine')

# Perform hierarchical clustering using average linkage
Z = linkage(distance_matrix, method='average')
print(list(Z))
# labels = fcluster(Z, K, criterion='maxclust')

b_t = gt() - t
print("Clustering time:", b_t, "seconds")

plt.figure()
dn = dendrogram(Z)
plt.show()

sys.stdout.flush()
