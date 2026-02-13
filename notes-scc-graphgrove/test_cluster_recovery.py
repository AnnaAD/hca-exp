#!/usr/bin/env python3
"""
Test script to verify that we can recover cluster assignments from SCC hierarchical clustering.
This demonstrates how to map each original vector back to its cluster at any level.
"""

import numpy as np
from graphgrove.vec_scc import Cosine_SCC
import matplotlib.pyplot as plt


def get_cluster_assignments(scc, level_idx):
    """
    Returns array where element i is the cluster ID of vector i at the specified level.
    
    Args:
        scc: Trained SCC clustering object
        level_idx: Index of the level (0 is bottom/finest, higher is coarser)
    
    Returns:
        numpy array of cluster assignments
    """
    level = scc.scc.levels[level_idx]
    n_vectors = scc.point_counter

    print("LEVEL", level.__dict__)

    
    # Initialize with -1 to detect any unmapped vectors
    assignments = np.full(n_vectors, -1, dtype=int)
    
    for cluster_id, node in enumerate(level.nodes):
        # Get all original vector indices in this cluster
        vector_indices = node.descendants()
        for vec_idx in vector_indices:
            assignments[vec_idx] = cluster_id
    
    return assignments


def make_synthetic_clusters(n_samples=150, n_features=10, n_clusters=3, cluster_std=0.5, random_state=42):
    """Create synthetic clustered data without sklearn."""
    np.random.seed(random_state)
    
    # Generate cluster centers
    centers = np.random.randn(n_clusters, n_features) * 3
    
    # Generate samples
    samples_per_cluster = n_samples // n_clusters
    X = []
    y = []
    
    for i in range(n_clusters):
        # Generate samples around each center
        cluster_samples = centers[i] + np.random.randn(samples_per_cluster, n_features) * cluster_std
        X.append(cluster_samples)
        y.extend([i] * samples_per_cluster)
    
    # Add remaining samples to last cluster if needed
    remaining = n_samples - len(y)
    if remaining > 0:
        cluster_samples = centers[-1] + np.random.randn(remaining, n_features) * cluster_std
        X.append(cluster_samples)
        y.extend([n_clusters-1] * remaining)
    
    return np.vstack(X), np.array(y)


def test_cluster_recovery():
    """
    Test function to verify the cluster recovery hypothesis.
    Creates synthetic data with known clusters and verifies we can recover assignments.
    """
    print("=== Testing SCC Cluster Recovery ===\n")
    
    # Generate synthetic data with 3 clear clusters
    n_samples = 150
    n_features = 10
    n_clusters = 3
    
    print(f"Generating {n_samples} samples with {n_clusters} ground truth clusters...")
    X, y_true = make_synthetic_clusters(n_samples=n_samples, n_features=n_features,
                                       n_clusters=n_clusters, cluster_std=0.5, random_state=42)
    
    # Normalize to unit vectors for cosine similarity
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    X = X.astype(np.float32)
    
    # Run SCC clustering
    print("\nRunning SCC clustering...")
    num_rounds = 10
    thresholds = np.geomspace(1.0, 0.001, num_rounds).astype(np.float32)
    scc = Cosine_SCC(k=15, num_rounds=num_rounds, thresholds=thresholds, 
                     index_name='cosine_sgtree', cores=4, verbosity=0)
    scc.partial_fit(X)
    
    print(f"SCC completed with {len(scc.scc.levels)} levels")
    
    # Test 1: Verify all vectors are assigned at each level
    print("\n--- Test 1: Checking vector coverage ---")
    all_vectors_assigned = True
    
    for level_idx in range(len(scc.scc.levels)):
        assignments = get_cluster_assignments(scc, level_idx)
        n_clusters_at_level = len(scc.scc.levels[level_idx].nodes)
        
        # Check no vector is unassigned
        unassigned = np.sum(assignments == -1)
        if unassigned > 0:
            print(f"ERROR: Level {level_idx} has {unassigned} unassigned vectors!")
            all_vectors_assigned = False
        
        # Count unique clusters
        unique_clusters = len(np.unique(assignments[assignments >= 0]))
        print(f"Level {level_idx}: {unique_clusters} clusters, "
              f"{n_clusters_at_level} nodes (all vectors assigned: {unassigned == 0})")
    
    print(f"\nAll vectors assigned test: {'PASSED' if all_vectors_assigned else 'FAILED'}")
    
    # Test 2: Verify hierarchy consistency
    print("\n--- Test 2: Checking hierarchy consistency ---")
    hierarchy_consistent = True
    
    for level_idx in range(len(scc.scc.levels) - 1):
        assignments_fine = get_cluster_assignments(scc, level_idx)
        assignments_coarse = get_cluster_assignments(scc, level_idx + 1)
        
        # For each fine cluster, all its vectors should map to the same coarse cluster
        for fine_cluster in np.unique(assignments_fine):
            vectors_in_cluster = np.where(assignments_fine == fine_cluster)[0]
            coarse_clusters = np.unique(assignments_coarse[vectors_in_cluster])
            
            if len(coarse_clusters) > 1:
                print(f"ERROR: Fine cluster {fine_cluster} at level {level_idx} "
                      f"maps to multiple coarse clusters: {coarse_clusters}")
                hierarchy_consistent = False
    
    print(f"\nHierarchy consistency test: {'PASSED' if hierarchy_consistent else 'FAILED'}")
    
    # Test 3: Verify descendants() returns correct indices
    print("\n--- Test 3: Checking descendants() method ---")
    descendants_correct = True
    
    # Check a few nodes at different levels
    for level_idx in [0, len(scc.scc.levels)//2, len(scc.scc.levels)-1]:
        if level_idx >= len(scc.scc.levels):
            continue
            
        level = scc.scc.levels[level_idx]
        for node_idx, node in enumerate(level.nodes[:min(3, len(level.nodes))]):
            descendants = list(node.descendants())
            
            # At level 0, each node should have exactly one descendant (itself)
            if level_idx == 0:
                if len(descendants) != 1 or descendants[0] != node.uid:
                    print(f"ERROR: Level 0 node {node.uid} has descendants {descendants}")
                    descendants_correct = False
            
            # All descendants should be valid indices
            if any(d < 0 or d >= n_samples for d in descendants):
                print(f"ERROR: Invalid descendant indices at level {level_idx}")
                descendants_correct = False
    
    print(f"\nDescendants test: {'PASSED' if descendants_correct else 'FAILED'}")
    
    # Test 4: Visualize clustering at different levels
    print("\n--- Test 4: Visualizing clusters at different levels ---")
    
    # Simple 2D projection using first two dimensions
    X_2d = X[:, :2]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot ground truth
    axes[0].scatter(X_2d[:, 0], X_2d[:, 1], c=y_true, cmap='tab10', alpha=0.6)
    axes[0].set_title('Ground Truth Clusters')
    
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
    plt.savefig('scc_cluster_recovery_test.png', dpi=150)
    print("Saved visualization to 'scc_cluster_recovery_test.png'")
    
    # Summary
    print("\n=== Test Summary ===")
    all_passed = all_vectors_assigned and hierarchy_consistent and descendants_correct
    print(f"Overall result: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    
    return all_passed


if __name__ == "__main__":
    test_cluster_recovery()