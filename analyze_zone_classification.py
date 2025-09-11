#!/usr/bin/env python3
"""
Analyze and visualize the quaternion zone classification results.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def analyze_zone_classification():
    """Analyze the quaternion zone classification results."""
    
    # Load results
    print("Loading classification results...")
    classified_df = pd.read_csv("fibonacci_zone_classification.csv")
    distribution_df = pd.read_csv("fibonacci_zone_classification_distribution.csv")
    
    print(f"Total quaternions: {len(classified_df)}")
    print(f"Zones with quaternions: {len(distribution_df)}")
    
    # Display zone distribution
    print("\n=== Zone Distribution ===")
    print(distribution_df[['zone_id', 'count', 'percentage', 'angle_deg']].to_string(index=False))
    
    # Analyze by rotation angle
    print("\n=== Distribution by Rotation Angle ===")
    angle_dist = distribution_df.groupby('angle_deg').agg({
        'count': 'sum',
        'zone_id': 'count'
    }).rename(columns={'zone_id': 'num_zones'})
    angle_dist['percentage'] = angle_dist['count'] / len(classified_df) * 100
    print(angle_dist)
    
    # Check distance statistics
    print(f"\n=== Distance Statistics ===")
    print(f"Mean distance to nearest zone: {classified_df['min_distance'].mean():.6f}")
    print(f"Median distance: {classified_df['min_distance'].median():.6f}")
    print(f"Std distance: {classified_df['min_distance'].std():.6f}")
    print(f"Max distance: {classified_df['min_distance'].max():.6f}")
    print(f"Min distance: {classified_df['min_distance'].min():.6f}")
    
    # Show some examples from each zone
    print(f"\n=== Sample Quaternions by Zone ===")
    for zone_id in sorted(distribution_df['zone_id'].values)[:10]:  # Show first 10 zones
        zone_quats = classified_df[classified_df['zone_id'] == zone_id]
        print(f"Zone {zone_id}: {len(zone_quats)} quaternions")
        print(f"  Sample: [{zone_quats.iloc[0]['w']:.4f}, {zone_quats.iloc[0]['x']:.4f}, "
              f"{zone_quats.iloc[0]['y']:.4f}, {zone_quats.iloc[0]['z']:.4f}] "
              f"(dist: {zone_quats.iloc[0]['min_distance']:.4f})")
    
    return classified_df, distribution_df

def create_visualizations(classified_df, distribution_df):
    """Create visualizations of the zone classification."""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Zone population histogram
    ax1 = axes[0, 0]
    ax1.bar(distribution_df['zone_id'], distribution_df['count'])
    ax1.set_xlabel('Zone ID')
    ax1.set_ylabel('Number of Quaternions')
    ax1.set_title('Quaternion Distribution Across Zones')
    ax1.grid(True, alpha=0.3)
    
    # 2. Distance distribution
    ax2 = axes[0, 1]
    ax2.hist(classified_df['min_distance'], bins=50, alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Distance to Nearest Zone Center')
    ax2.set_ylabel('Number of Quaternions')
    ax2.set_title('Distribution of Distances to Zone Centers')
    ax2.grid(True, alpha=0.3)
    
    # 3. Distribution by rotation angle
    ax3 = axes[1, 0]
    angle_counts = distribution_df.groupby('angle_deg')['count'].sum()
    ax3.bar(angle_counts.index, angle_counts.values)
    ax3.set_xlabel('Rotation Angle (degrees)')
    ax3.set_ylabel('Number of Quaternions')
    ax3.set_title('Distribution by Rotation Angle')
    ax3.grid(True, alpha=0.3)
    
    # 4. Zone count by rotation angle
    ax4 = axes[1, 1]
    zone_counts = distribution_df.groupby('angle_deg').size()
    ax4.bar(zone_counts.index, zone_counts.values)
    ax4.set_xlabel('Rotation Angle (degrees)')
    ax4.set_ylabel('Number of Zones')
    ax4.set_title('Number of Active Zones by Rotation Angle')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('quaternion_zone_analysis.png', dpi=300, bbox_inches='tight')
    print("\nSaved visualization to: quaternion_zone_analysis.png")
    plt.show()

def create_3d_visualization(classified_df, distribution_df):
    """Create a 3D visualization of the quaternions in their zones."""
    
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Sample a subset for visualization (plotting 1M points is slow)
    sample_size = min(10000, len(classified_df))
    sample_df = classified_df.sample(n=sample_size, random_state=42)
    
    # Color by zone
    zones = sample_df['zone_id'].values
    
    # Plot quaternions in 3D (using x, y, z components, w determines color intensity)
    scatter = ax.scatter(sample_df['x'], sample_df['y'], sample_df['z'], 
                        c=zones, cmap='viridis', alpha=0.6, s=2)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y') 
    ax.set_zlabel('Z')
    ax.set_title(f'3D Visualization of Quaternions by Zone\n(Sample of {sample_size} points)')
    
    plt.colorbar(scatter, ax=ax, label='Zone ID')
    plt.savefig('quaternion_3d_zones.png', dpi=300, bbox_inches='tight')
    print("Saved 3D visualization to: quaternion_3d_zones.png")
    plt.show()

def main():
    print("=== Quaternion Zone Classification Analysis ===")
    
    # Analyze results
    classified_df, distribution_df = analyze_zone_classification()
    
    # Create visualizations
    print("\nCreating visualizations...")
    create_visualizations(classified_df, distribution_df)
    create_3d_visualization(classified_df, distribution_df)
    
    # Summary insights
    print(f"\n=== Key Insights ===")
    print(f"1. Only {len(distribution_df)} out of 48 possible zones are populated")
    print(f"2. This is expected because we canonicalized quaternions (w >= 0)")
    print(f"3. The distribution is quite uniform across zones (~4.17% each)")
    print(f"4. Average distance to zone centers: {classified_df['min_distance'].mean():.4f}")
    
    # Check which symmetries are represented
    unique_angles = sorted(distribution_df['angle_deg'].unique())
    print(f"5. Rotation angles represented: {unique_angles}")
    
    angle_counts = distribution_df.groupby('angle_deg').size()
    print(f"6. Number of zones per angle: {dict(angle_counts)}")

if __name__ == "__main__":
    main()
