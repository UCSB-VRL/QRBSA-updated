#!/usr/bin/env python3
"""
Classify quaternions from Fibonacci sampling into 48 zones based on cubic group symmetries.
"""

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

def load_fibonacci_quaternions(filepath):
    """Load quaternions from text file."""
    print(f"Loading Fibonacci quaternions from: {filepath}")
    quats = np.loadtxt(filepath)
    print(f"Loaded {len(quats)} quaternions with shape {quats.shape}")
    
    # Normalize quaternions
    norms = np.linalg.norm(quats, axis=1)
    quats = quats / norms[:, np.newaxis]
    
    # Canonicalize to w >= 0 (same convention as cubic group)
    mask = quats[:, 0] < 0
    quats[mask] = -quats[mask]
    
    return quats

def load_cubic_group_quaternions(filepath):
    """Load the 48 cubic group quaternions."""
    print(f"Loading cubic group quaternions from: {filepath}")
    df = pd.read_csv(filepath)
    
    # Extract quaternions (w, x, y, z)
    cubic_quats = df[['w', 'x', 'y', 'z']].values
    print(f"Loaded {len(cubic_quats)} cubic group quaternions")
    
    return cubic_quats, df

def quaternion_distance(q1, q2):
    """
    Compute the distance between quaternions considering q and -q represent the same rotation.
    Returns the minimum of |q1 - q2| and |q1 + q2|.
    """
    # Both q1 and q2 should be arrays of quaternions
    if q1.ndim == 1:
        q1 = q1.reshape(1, -1)
    if q2.ndim == 1:
        q2 = q2.reshape(1, -1)
    
    # Compute distances to both q2 and -q2
    dist_pos = cdist(q1, q2, metric='euclidean')
    dist_neg = cdist(q1, -q2, metric='euclidean')
    
    # Take minimum distance (accounting for q and -q equivalence)
    return np.minimum(dist_pos, dist_neg)

def classify_quaternions_to_zones(fib_quats, cubic_quats):
    """
    Classify each Fibonacci quaternion to the nearest cubic group element (zone).
    """
    print("Classifying quaternions into zones...")
    
    n_fib = len(fib_quats)
    n_cubic = len(cubic_quats)
    
    # Find nearest cubic group element for each Fibonacci quaternion
    zones = np.zeros(n_fib, dtype=int)
    min_distances = np.zeros(n_fib)
    
    # Process in batches to manage memory
    batch_size = 10000
    
    for i in range(0, n_fib, batch_size):
        end_idx = min(i + batch_size, n_fib)
        batch_quats = fib_quats[i:end_idx]
        
        # Compute distances to all cubic group elements
        distances = quaternion_distance(batch_quats, cubic_quats)
        
        # Find closest zone for each quaternion in batch
        batch_zones = np.argmin(distances, axis=1)
        batch_min_dist = np.min(distances, axis=1)
        
        zones[i:end_idx] = batch_zones
        min_distances[i:end_idx] = batch_min_dist
        
        if (i // batch_size) % 10 == 0:
            print(f"Processed {end_idx}/{n_fib} quaternions...")
    
    return zones, min_distances

def analyze_zone_distribution(zones, cubic_df):
    """Analyze the distribution of quaternions across zones."""
    print("\n=== Zone Distribution Analysis ===")
    
    unique_zones, counts = np.unique(zones, return_counts=True)
    
    print(f"Number of zones with quaternions: {len(unique_zones)}/48")
    print(f"Total quaternions classified: {np.sum(counts)}")
    
    # Create distribution DataFrame
    zone_dist = pd.DataFrame({
        'zone_id': unique_zones,
        'count': counts,
        'percentage': counts / len(zones) * 100
    })
    
    # Add cubic group information
    zone_dist = zone_dist.merge(
        cubic_df[['mat_index', 'w', 'x', 'y', 'z', 'angle_deg']].reset_index().rename(columns={'index': 'zone_id'}),
        on='zone_id',
        how='left'
    )
    
    # Sort by count (descending)
    zone_dist = zone_dist.sort_values('count', ascending=False)
    
    print(f"\nTop 10 most populated zones:")
    print(zone_dist.head(10)[['zone_id', 'count', 'percentage', 'angle_deg']].to_string(index=False))
    
    print(f"\nBottom 10 least populated zones:")
    print(zone_dist.tail(10)[['zone_id', 'count', 'percentage', 'angle_deg']].to_string(index=False))
    
    # Check for empty zones
    empty_zones = set(range(48)) - set(unique_zones)
    if empty_zones:
        print(f"\nEmpty zones: {sorted(empty_zones)}")
    else:
        print(f"\nAll 48 zones have at least one quaternion!")
    
    return zone_dist

def save_results(fib_quats, zones, min_distances, zone_dist, output_prefix="fibonacci_zone_classification"):
    """Save classification results."""
    
    # Save classified quaternions with zone assignments
    classified_data = np.column_stack([fib_quats, zones, min_distances])
    classified_df = pd.DataFrame(
        classified_data, 
        columns=['w', 'x', 'y', 'z', 'zone_id', 'min_distance']
    )
    
    classified_file = f"{output_prefix}.csv"
    classified_df.to_csv(classified_file, index=False)
    print(f"\nSaved classified quaternions to: {classified_file}")
    
    # Save zone distribution
    dist_file = f"{output_prefix}_distribution.csv"
    zone_dist.to_csv(dist_file, index=False)
    print(f"Saved zone distribution to: {dist_file}")
    
    return classified_file, dist_file

def main():
    # File paths
    fib_file = "/data/home/umang/Materials/QRBSA-data-augmentation/quaternions_fibonacci.txt"
    cubic_file = "/data/home/umang/Materials/QRBSA-data-augmentation/cubic_group_quaternions_48.csv"
    
    # Load data
    fib_quats = load_fibonacci_quaternions(fib_file)
    cubic_quats, cubic_df = load_cubic_group_quaternions(cubic_file)
    
    # Classify quaternions
    zones, min_distances = classify_quaternions_to_zones(fib_quats, cubic_quats)
    
    # Analyze distribution
    zone_dist = analyze_zone_distribution(zones, cubic_df)
    
    # Save results
    classified_file, dist_file = save_results(fib_quats, zones, min_distances, zone_dist)
    
    print(f"\n=== Summary ===")
    print(f"Processed {len(fib_quats)} Fibonacci quaternions")
    print(f"Classified into {len(np.unique(zones))} out of 48 possible zones")
    print(f"Average distance to nearest zone: {np.mean(min_distances):.6f}")
    print(f"Max distance to nearest zone: {np.max(min_distances):.6f}")
    
    print(f"\nOutput files:")
    print(f"  - {classified_file}")
    print(f"  - {dist_file}")

if __name__ == "__main__":
    main()
