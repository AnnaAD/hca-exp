#!/usr/bin/env python3
"""
Test script to understand SCC output format
"""

import numpy as np
import sys
sys.path.append('graphgrove')

from graphgrove.vec_scc import Cosine_SCC
from graphgrove.graph_builder import unit_norm

# Create a small test dataset
np.random.seed(42)
N = 20  # small number of points for testing
K = 3   # 3 clusters
D = 10  # 10 dimensions

# Create distinct clusters
means = 10 * np.random.rand(K, D)
x = np.vstack([np.random.randn(N//K, D) + means[i] for i in range(K)])
# Add remaining points to last cluster
remaining = N - (N//K) * K
if remaining > 0:
    x = np.vstack([x, np.random.randn(remaining, D) + means[-1]])

# Normalize
x = unit_norm(x)
x = x.astype(np.float32)

print(x)

print(f"Test data shape: {x.shape}")
print("Creating SCC clustering...")

# Create SCC with a few thresholds
num_rounds = 5
thresholds = np.geomspace(1.0, 0.1, num_rounds).astype(np.float32)
print(f"Thresholds: {thresholds}")

scc = Cosine_SCC(k=5, num_rounds=num_rounds, thresholds=thresholds, 
                 index_name='cosine_sgtree', cores=1, verbosity=1)

# Fit the data
scc.partial_fit(x)

print("\n=== SCC Structure ===")
print(f"Number of levels: {len(scc.scc.levels)}")

# Explore each level
for i, level in enumerate(scc.scc.levels):
    print(f"\nLevel {i} (height={level.height}):")
    print(f"  Number of nodes: {len(level.nodes)}")
    #print(f"  Threshold: {level.threshold}")
    
    # Show first few nodes
    for j, node in enumerate(level.nodes[:5]):
        print(f"  Node {j}: id={node.uid}, height={node.height}")
        if hasattr(node, 'descendants'):
            desc = node.descendants()
            print(f"    Descendants: {desc}")
        if hasattr(node, 'children'):
            children = node.children
            print(f"    Children: {[c.uid for c in children]}")

# Get roots (top level clusters)
print("\n=== Root Nodes (Top Level Clusters) ===")
roots = scc.scc.roots()
print(f"Number of root clusters: {len(roots)}")

for i, root in enumerate(roots):
    print(f"\nRoot cluster {i}:")
    print(f"  ID: {root.uid}")
    print(f"  Height: {root.height}")
    descendants = root.descendants()
    print(f"  Descendants (leaf nodes): {descendants}")
    print(f"  Number of points in cluster: {len(descendants)}")

# Clean up
del scc