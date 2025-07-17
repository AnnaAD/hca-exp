import numpy as np
from scipy.cluster import hierarchy
import matplotlib.pyplot as plt

ytdist = np.array([662., 877., 255., 412., 996., 295., 468., 268.,
                   400., 754., 564., 138., 219., 869., 669.])
Z = hierarchy.linkage(ytdist, 'single')
print(Z)
plt.figure()
dn = hierarchy.dendrogram(Z)
plt.show()

#https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html
# for notes on the linkage matrix output