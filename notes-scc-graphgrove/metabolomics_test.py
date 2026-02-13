from test_cluster_recovery import get_cluster_assignments

import time
import sys

import numpy as np
import matplotlib.pyplot as plt

from graphgrove.vec_scc import Cosine_SCC
from graphgrove.graph_builder import unit_norm

from plot_scc_dendrogram_fixed import custom_sparse_distance_tree_plot

gt = time.time

np.random.seed(123)
cores = 12

print('======== Loading Dataset ==========')
with open('anna_full_metabolomics_data.csv') as f:
    ncols = len(f.readline().split(','))

x = np.loadtxt('anna_full_metabolomics_data.csv', delimiter=',', skiprows=1, usecols=range(1,ncols))
print("loaded")

print(x.shape)
print(x.mean())
print(x[:5][:5])

x = x.astype(np.float32)
#x = unit_norm(x)  # Normalize for cosine similarity
x = np.require(x, requirements=['A', 'C', 'O', 'W'])
print(x)
print(f"Data shape: {x.shape}")
print(f"Data range: [{x.min():.3f}, {x.max():.3f}]")
print(f"Norm of first vector: {np.linalg.norm(x[0]):.3f}")


print('======== SCC ==========')
t = gt()
num_rounds = 100
thresholds = np.geomspace(1.0, 0.001, num_rounds).astype(np.float32)
scc = Cosine_SCC(k=5, num_rounds=num_rounds, thresholds=thresholds, index_name='cosine_sgtree', cores=cores, verbosity=1)

out = scc.partial_fit(x)
b_t = gt() - t
print("Clustering time:", b_t, "seconds")
sys.stdout.flush()

X_2d = x[:, :2]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Plot ground truth
# axes[0].scatter(X_2d[:, 0], X_2d[:, 1], c=y_true, cmap='tab10', alpha=0.6)
# axes[0].set_title('Ground Truth Clusters')

# Plot fine-grained clustering (early level)
level_fine = min(2, len(scc.scc.levels)-1)
assignments_fine = get_cluster_assignments(scc, level_fine)
axes[1].scatter(X_2d[:, 0], X_2d[:, 1], c=assignments_fine, cmap='tab10', alpha=0.6)
axes[1].set_title(f'SCC Level {level_fine} ({len(np.unique(assignments_fine))} clusters)')

# Plot coarse clustering (later level)
level_coarse = min(len(scc.scc.levels)-2, len(scc.scc.levels)-1)
assignments_coarse = get_cluster_assignments(scc, level_coarse)
axes[2].scatter(X_2d[:, 0], X_2d[:, 1], c=assignments_coarse, cmap='tab10', alpha=0.6)
axes[2].set_title(f'SCC Level {level_coarse} ({len(np.unique(assignments_coarse))} clusters)')

plt.tight_layout()
plt.savefig('mb.png', dpi=150)
print("Saved visualization to 'mb.png'")

custom_sparse_distance_tree_plot(scc, x, 20, leaves=True)