"""
Copyright (c) 2021 The authors of SCC All rights reserved.

Initially modified from CoverTree
https://github.com/manzilzaheer/CoverTree
Copyright (c) 2017 Manzil Zaheer All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import time
import sys

import numpy as np

from graphgrove.vec_scc import Cosine_SCC
from graphgrove.graph_builder import unit_norm

#!/usr/bin/env python3
"""
Fixed SCC dendrogram plotting with three different approaches
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from scipy.spatial.distance import pdist, squareform
import sys
sys.path.append('graphgrove')

from graphgrove.vec_scc import Cosine_SCC
from graphgrove.graph_builder import unit_norm


def approach3_custom_tree_plot(scc, figsize=(12, 8), top_n_levels=5):
    """
    Approach 3: Custom visualization that shows the actual SCC structure
    This doesn't use scipy's dendrogram but creates a custom plot
    
    Args:
        scc: Fitted SCC object
        figsize: Figure size tuple
        top_n_levels: Number of top levels to plot (default 5)
    """
    import matplotlib.patches as patches
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Determine which levels to plot (top N levels)
    total_levels = len(scc.scc.levels)
    start_level = max(0, total_levels - top_n_levels)
    levels_to_plot = list(range(start_level, total_levels))
    
    print(f"Plotting levels {start_level} to {total_levels-1} (top {len(levels_to_plot)} levels)")
    
    # Collect all nodes and their positions
    node_positions = {}
    level_nodes = []
    
    # Group nodes by level (only for levels we're plotting)
    for plot_idx, level_idx in enumerate(levels_to_plot):
        level = scc.scc.levels[level_idx]
        level_nodes.append([])
        for node in level.nodes:
            level_nodes[plot_idx].append(node)

    # Calculate positions
    max_width = max(len(nodes) for nodes in level_nodes)
    
    # Plot nodes and edges
    for plot_idx, nodes in enumerate(level_nodes):
        level_idx = levels_to_plot[plot_idx]
        print(f"Level {level_idx}: {len(nodes)} nodes")
        
        y = plot_idx  # Use plot index for y-coordinate
        width = len(nodes)
        x_start = (max_width - width) / 2
        
        for i, node in enumerate(nodes):
            x = x_start + i
            node_positions[node.uid] = (x, y)
            
            # Draw node
            descendants = node.descendants()
            size = len(descendants)
            
            # Scale node size based on number of descendants
            radius = min(0.4, 0.1 + 0.3 * (size / scc.point_counter))
            circle = plt.Circle((x, y), radius, color='lightblue', 
                              edgecolor='darkblue', linewidth=1, zorder=2)
            ax.add_patch(circle)
            
            # Add label
            label = f"{node.uid}\n({size})"
            ax.text(x, y, label, ha='center', va='center', 
                   fontsize=max(6, min(10, 200/max_width)), zorder=3)
    
    # Draw edges between levels
    for plot_idx in range(len(level_nodes) - 1):
        level_idx = levels_to_plot[plot_idx]
        level = scc.scc.levels[level_idx]
        
        for node in level.nodes:
            if node.uid in node_positions:
                x1, y1 = node_positions[node.uid]
                
                # Find parent in next level
                if plot_idx + 1 < len(level_nodes):
                    next_level = scc.scc.levels[levels_to_plot[plot_idx + 1]]
                    for next_node in next_level.nodes:
                        # Check if this node is a child of next_node
                        if node.uid in [child.uid for child in next_node.children]:
                            if next_node.uid in node_positions:
                                x2, y2 = node_positions[next_node.uid]
                                ax.plot([x1, x2], [y1, y2], 'k-', alpha=0.5, 
                                       linewidth=0.5, zorder=1)
    
    # Set plot limits and labels
    ax.set_xlim(-1, max_width)
    ax.set_ylim(-0.5, len(level_nodes) - 0.5)
    ax.set_xlabel('Node Position')
    ax.set_ylabel('Level (relative)')
    ax.set_title(f'SCC Hierarchical Structure (Top {len(levels_to_plot)} Levels)')
    ax.invert_yaxis()  # Top level at top
    
    # Add level information on the side
    for plot_idx, level_idx in enumerate(levels_to_plot):
        level = scc.scc.levels[level_idx]
        ax.text(-0.5, plot_idx, f"L{level_idx}\nh={level.height:.3f}", 
               ha='right', va='center', fontsize=8)
    
    plt.tight_layout()
    plt.show()

def plot_scc_dendrogram(scc, X=None, method='binary', **kwargs):
    """
    Main function to plot SCC dendrogram
    
    Args:
        scc: Fitted SCC object
        X: Original data (required for method='distance')
        method: 'binary', 'distance', or 'custom'
        **kwargs: Additional arguments for plotting
    """
    
    if method == 'custom':
        approach3_custom_tree_plot(scc, figsize=kwargs.get('figsize', (12, 8)))
    
    else:
        print(f"Unknown method: {method}")

gt = time.time

np.random.seed(123)
cores = 12

print('======== Building Dataset ==========')
# N=1000
# K=5
# D=784
# means = 20*np.random.rand(K,D) - 10
# x = np.vstack([np.random.randn(N,D) + means[i] for i in range(K)])
# np.random.shuffle(x)
# x = unit_norm(x)

x = np.loadtxt('../../anna_full_metabolomics_data.csv', delimiter=',',skiprows=1)
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
num_rounds = 50
thresholds = np.geomspace(1.0, .1, num_rounds).astype(np.float32)
print(thresholds)
scc = Cosine_SCC(k=10, num_rounds=num_rounds, thresholds=thresholds, index_name='cosine_sgtree', cores=cores, verbosity=1)
scc.partial_fit(x)
b_t = gt() - t
print("Clustering time:", b_t, "seconds")

print(thresholds)
print("\nLevel summary:")
for i, level in enumerate(scc.scc.levels):
    print(f"Level {i}: {len(level.nodes)} nodes at height {level.height:.4f}")
    if i == len(scc.scc.levels) - 1:  # Final level
        print("Final clusters:")
        for j, node in enumerate(level.nodes[:10]):  # Show first 10 clusters
            desc_count = len(node.descendants())
            print(f"  Cluster {j}: {desc_count} points")
sys.stdout.flush()

#plot_scc_dendrogram(scc, method="custom", top_n_levels=1)

# print('======== MB-SCC ==========')
# t = gt()
# num_rounds = 50
# thresholds = np.geomspace(1.0, 0.001, num_rounds).astype(np.float32)
# scc = Cosine_SCC(k=5, num_rounds=num_rounds, thresholds=thresholds, index_name='cosine_sgtree', cores=cores, verbosity=0)
# bs = 1
# for i in range(0, x.shape[0], bs):
#     # print(i)
#     scc.partial_fit(x[i:i+bs])
# b_t = gt() - t
# print("Clustering time:", b_t, "seconds")
del scc
sys.stdout.flush()
