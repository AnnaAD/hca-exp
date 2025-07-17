# GraphGrove SCC Output Structure Documentation

## Overview

GraphGrove's SCC (Sub-Cluster Component) algorithm produces a hierarchical clustering structure represented as a tree with multiple levels. This document explains how to interpret the output and map original vectors to their clusters at any level of the hierarchy.

## Key Concepts

### Hierarchical Structure

The SCC algorithm produces a hierarchy with the following structure:
- **Level 0 (Bottom)**: Each node represents a single data point
- **Level 1, 2, ...**: Progressively coarser clusters
- **Level N (Top)**: Usually contains very few clusters (often just 1-3)

### Node Structure

Each node in the hierarchy contains:
- `uid`: Unique identifier for the node
- `descendants()`: Method that returns the set of original vector indices contained in this cluster
- `children`: Direct child nodes from the level below
- `mean`: Cluster centroid (if computed)
- Other metadata about the cluster

## Level Structure Details

### Level 0 - Individual Data Points

At level 0, each node represents exactly one input vector:
- Number of nodes = Number of input vectors
- Each node's `uid` corresponds to the vector's index (0, 1, 2, ...)
- `descendants()` returns a set containing only the node's own uid
- This is the finest granularity possible

Example:
```python
level_0 = scc.scc.levels[0]
for node in level_0.nodes:
    print(f"Node {node.uid} represents vector {node.uid}")
    print(f"Descendants: {node.descendants()}")  # Will print {uid}
```

### Levels 1+ - Cluster Hierarchy

At higher levels, nodes represent clusters of multiple vectors:
- Each node may contain multiple data points
- The `descendants()` method returns ALL original vector indices in the cluster
- Nodes at level i+1 are formed by merging nodes from level i

Example progression:
- Level 0: 150 nodes (one per vector)
- Level 1: 150 nodes (some initial merging based on k-NN graph)
- Level 2: 34 nodes (more aggressive clustering)
- Level 3: 8 nodes (coarser clusters)
- ...
- Level 10: 3 nodes (very coarse clusters)

## Mapping Vectors to Clusters

### Basic Mapping Function

```python
def get_cluster_assignments(scc, level_idx):
    """
    Returns array where element i is the cluster ID of vector i.
    
    Args:
        scc: Trained SCC object
        level_idx: Index of the level (0 = finest, higher = coarser)
    
    Returns:
        numpy array where assignments[i] = cluster_id for vector i
    """
    level = scc.scc.levels[level_idx]
    n_vectors = scc.point_counter
    
    assignments = np.zeros(n_vectors, dtype=int)
    
    for cluster_id, node in enumerate(level.nodes):
        # Get all original vector indices in this cluster
        for vec_idx in node.descendants():
            assignments[vec_idx] = cluster_id
    
    return assignments
```

### Understanding the Mapping

The key insight is that `descendants()` always returns the original vector indices (from level 0), regardless of which level you're examining. This makes it straightforward to determine which cluster any vector belongs to at any level of granularity.

## Practical Examples

### Example 1: Get All Vectors in a Specific Cluster

```python
# Get all vectors in cluster 2 at level 5
level_5 = scc.scc.levels[5]
cluster_node = level_5.nodes[2]
vector_indices = list(cluster_node.descendants())
print(f"Cluster 2 contains vectors: {vector_indices}")

# Retrieve the actual vector data
vectors_in_cluster = original_data[vector_indices]
```

### Example 2: Find Which Cluster a Specific Vector Belongs To

```python
def find_vector_cluster(scc, vector_idx, level_idx):
    """Find which cluster a specific vector belongs to at a given level."""
    level = scc.scc.levels[level_idx]
    
    for cluster_id, node in enumerate(level.nodes):
        if vector_idx in node.descendants():
            return cluster_id
    
    return -1  # Vector not found

# Example: Find cluster for vector 42 at level 3
cluster_id = find_vector_cluster(scc, 42, 3)
```

### Example 3: Track Cluster Evolution Across Levels

```python
def track_vector_clusters(scc, vector_idx):
    """Track which clusters a vector belongs to across all levels."""
    clusters_by_level = {}
    
    for level_idx in range(len(scc.scc.levels)):
        cluster_id = find_vector_cluster(scc, vector_idx, level_idx)
        clusters_by_level[level_idx] = cluster_id
    
    return clusters_by_level

# Example: Track vector 0 through the hierarchy
history = track_vector_clusters(scc, 0)
print(f"Vector 0 cluster assignments: {history}")
```

## Important Notes

1. **Sequential Vector IDs**: The implementation assumes vectors are provided sequentially (0, 1, 2, ...). The vector at position i in your input array will have uid=i at level 0.

2. **Hierarchy Consistency**: The hierarchy is consistent - if two vectors are in the same cluster at level i, they will remain in the same cluster at all levels > i.

3. **Memory Efficiency**: The `descendants()` method computes the descendant set on-demand by traversing the tree, so it may be expensive for very large clusters. Consider caching results if you need to access them repeatedly.

4. **Deleted Nodes**: Some nodes may be marked as deleted during the clustering process. The `level.nodes` list only contains active (non-deleted) nodes.

## Common Use Cases

1. **Multi-scale Analysis**: Examine clusters at different granularities by accessing different levels
2. **Cluster Stability**: Check how clusters merge across levels to understand stability
3. **Outlier Detection**: Identify small clusters that persist at higher levels
4. **Dendrogram Construction**: Use the parent-child relationships to build a dendrogram
5. **Cut-tree at Threshold**: Select a specific level based on the number of desired clusters

## Performance Considerations

- Accessing `descendants()` for nodes near the top of the hierarchy can be expensive as it needs to traverse many child nodes
- For large-scale analysis, consider pre-computing and caching cluster assignments for frequently accessed levels
- The level 0 access is always O(1) since each node has only itself as a descendant