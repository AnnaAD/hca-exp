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


def custom_tree_plot(scc, figsize=(12, 8), top_n_levels=5):
    """
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

def approach1_binary_tree_conversion(scc):
    """
    Approach 1: Convert n-ary merges to binary merges
    This maintains the hierarchical structure but creates intermediate nodes
    """
    linkage_rows = []
    cluster_map = {}
    next_idx = 0
    
    # Get all unique leaf nodes
    leaf_set = set()
    for level in scc.scc.levels:
        for node in level.nodes:
            desc = node.descendants()
            if len(desc) == 1 and len(node.children) == 0:
                leaf_set.add(int(desc.flatten()[0]))
    
    n_leaves = len(leaf_set)
    leaf_to_idx = {leaf: i for i, leaf in enumerate(sorted(leaf_set))}
    next_idx = n_leaves
    
    # Process each level
    processed = set()
    
    for level in scc.scc.levels:
        for node in level.nodes:
            if node.uid in processed:
                continue
                
            children = node.children
            if len(children) <= 1:
                # Leaf or single child - just map it
                desc = node.descendants()
                if len(desc) == 1:
                    cluster_map[node.uid] = leaf_to_idx[int(desc.flatten()[0])]
            else:
                # Multiple children - convert to binary merges
                child_indices = []
                for child in children:
                    if child.uid in cluster_map:
                        child_indices.append(cluster_map[child.uid])
                    else:
                        # Find child's cluster index
                        child_desc = child.descendants()
                        if len(child_desc) == 1:
                            idx = leaf_to_idx[int(child_desc.flatten()[0])]
                            child_indices.append(idx)
                            cluster_map[child.uid] = idx
                
                if len(child_indices) >= 2:
                    # Sort for consistency
                    child_indices.sort()
                    
                    # Binary merges
                    current = child_indices[0]
                    for i in range(1, len(child_indices)):
                        linkage_rows.append([
                            float(min(current, child_indices[i])),
                            float(max(current, child_indices[i])),
                            float(level.height),
                            float(i + 1)  # Approximate count
                        ])
                        current = next_idx
                        next_idx += 1
                    
                    cluster_map[node.uid] = current - 1
                    processed.add(node.uid)
    
    return np.array(linkage_rows, dtype=np.float64) if linkage_rows else None

def approach2_distance_matrix(X, scc):
    """
    Approach 2: Build linkage from pairwise distances and cluster assignments
    This uses the actual data distances
    """
    from scipy.cluster.hierarchy import linkage
    
    # Compute pairwise distances
    distances = pdist(X, metric='cosine')
    
    # Use average linkage (you can also try 'single', 'complete', 'ward')
    Z = linkage(distances, method='average')
    
    return Z

def approach3_custom_tree_plot(scc, figsize=(12, 8)):
    """
    Approach 3: Custom visualization that shows the actual SCC structure
    This doesn't use scipy's dendrogram but creates a custom plot
    """
    import matplotlib.patches as patches
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Collect all nodes and their positions
    node_positions = {}
    level_nodes = {}
    
    # Group nodes by level
    for i, level in enumerate(scc.scc.levels):
        level_nodes[i] = []
        for node in level.nodes:
            level_nodes[i].append(node)
    
    # Calculate positions
    max_width = max(len(nodes) for nodes in level_nodes.values())
    
    for level_idx, nodes in level_nodes.items():
        y = level_idx
        width = len(nodes)
        x_start = (max_width - width) / 2
        
        for i, node in enumerate(nodes):
            x = x_start + i
            node_positions[node.uid] = (x, y)
            
            # Draw node
            descendants = node.descendants()
            size = len(descendants)
            circle = plt.Circle((x, y), 0.2, color='lightblue', zorder=2)
            ax.add_patch(circle)
            
            # Add label
            label = f"{node.uid}\n({size})"
            ax.text(x, y, label, ha='center', va='center', fontsize=8, zorder=3)
            
            # Draw edges to children
            for child in node.children:
                if child.uid in node_positions:
                    x1, y1 = node_positions[node.uid]
                    x2, y2 = node_positions[child.uid]
                    ax.plot([x1, x2], [y1, y2], 'k-', alpha=0.5, zorder=1)
    
    ax.set_xlim(-1, max_width)
    ax.set_ylim(-1, len(level_nodes))
    ax.set_xlabel('Node Position')
    ax.set_ylabel('Level')
    ax.set_title('SCC Hierarchical Structure')
    ax.invert_yaxis()  # Top level at top
    
    # Add height labels
    for i, level in enumerate(scc.scc.levels):
        ax.text(-0.5, i, f"h={level.height:.2f}", ha='right', va='center')
    
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
    if method == 'binary':
        Z = approach1_binary_tree_conversion(scc)
        if Z is not None:
            plt.figure(figsize=kwargs.get('figsize', (10, 7)))
            dendrogram(Z, **{k: v for k, v in kwargs.items() if k != 'figsize'})
            plt.title('SCC Dendrogram (Binary Conversion)')
            plt.xlabel('Sample Index')
            plt.ylabel('Height')
            plt.tight_layout()
            plt.show()
        else:
            print("Could not create binary linkage matrix")
    
    elif method == 'distance':
        if X is None:
            print("Original data X is required for distance method")
            return
        Z = approach2_distance_matrix(X, scc)
        plt.figure(figsize=kwargs.get('figsize', (10, 7)))
        dendrogram(Z, **{k: v for k, v in kwargs.items() if k != 'figsize'})
        plt.title('Dendrogram from Distance Matrix')
        plt.xlabel('Sample Index')
        plt.ylabel('Distance')
        plt.tight_layout()
        plt.show()
    
    elif method == 'custom':
        approach3_custom_tree_plot(scc, figsize=kwargs.get('figsize', (12, 8)))
    
    else:
        print(f"Unknown method: {method}")

if __name__ == "__main__":
    # Test data
    np.random.seed(42)
    N = 3000
    K = 3
    D = 5
    
    # Create clusters
    means = 10 * np.random.rand(K, D)
    x = np.vstack([np.random.randn(N//K, D) + means[i] for i in range(K)])
    x = unit_norm(x)
    x = x.astype(np.float32)
    
    # Fit SCC
    num_rounds = 100
    thresholds = np.geomspace(1.0, 0.001, num_rounds).astype(np.float32)
    scc = Cosine_SCC(k=5, num_rounds=num_rounds, thresholds=thresholds, 
                     index_name='cosine_sgtree', cores=1, verbosity=0)
    scc.partial_fit(x)
    
    print("Testing different dendrogram approaches:\n")
    
    # Test all three approaches
    # print("1. Binary tree conversion approach:")
    # plot_scc_dendrogram(scc, method='binary')
    
    # print("\n2. Distance matrix approach:")
    # plot_scc_dendrogram(scc, X=x, method='distance')
    
    print("\n3. Custom tree visualization:")
    custom_tree_plot(scc, top_n_levels = 5)