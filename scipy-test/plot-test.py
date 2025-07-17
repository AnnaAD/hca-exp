import numpy as np
from scipy.cluster import hierarchy
import matplotlib.pyplot as plt
import random

n = 1_000_000 # number of observations
rows = n - 1

# Preallocate an empty matrix
Z = np.zeros((rows, 4), dtype=np.float64)


unused = 0

for i in range(n - 1):
    
    Z[i, 0] = unused
    Z[i, 1] = unused + 1
    Z[i, 2] = i*random.randint(1,9)  # fake increasing distance
    Z[i, 3] = 2 if i == 0 else Z[i - 1, 3] + 1  # fake size growing
    unused += 2
    

print("generated")
# Save to disk (optional)
# np.save("fake_linkage_1M.npy", Z)

# Optional: test if dendrogram plotting works — WARNING: This will be *very* slow or crash!
# Comment this out if you're just testing matrix creation
plt.figure()
dn = hierarchy.dendrogram(Z, p = 3, truncate_mode = "level")
plt.show()

print("Fake linkage matrix for 1M observations created. Shape:", Z.shape)
