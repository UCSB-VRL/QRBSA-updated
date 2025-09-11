#!/usr/bin/env python3
"""
Visualize quaternion zones with separate colors for each of the 24 active zones.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches

def load_classification_data():
    """Load the zone classification data."""
    print("Loading classification data...")
    classified_df = pd.read_csv("fibonacci_zone_classification.csv")
    distribution_df = pd.read_csv("fibonacci_zone_classification_distribution.csv")
    cubic_df = pd.read_csv("cubic_group_quaternions_48.csv")
    
    return classified_df, distribution_df, cubic_df

def create_zone_color_mapping():
    """Create a consistent color mapping for the 24 active zones."""
    
    # Load data to get active zones
    _, distribution_df, _ = load_classification_data()
    active_zones = sorted(distribution_df['zone_id'].values)
    
    # Create a color mapping using matplotlib colormaps
    # Use different colormaps for different rotation angles for better distinction
    color_mapping = {}
    
    # Group zones by rotation angle
    angle_groups = distribution_df.groupby('angle_deg')['zone_id'].apply(list).to_dict()
    
    # Color schemes for different angles
    color_schemes = {
        0.0: ['red'],  # Identity - single red
        90.0: plt.cm.Blues(np.linspace(0.3, 0.9, 6)),  # 6 zones - blues
        120.0: plt.cm.Greens(np.linspace(0.3, 0.9, 8)),  # 8 zones - greens  
        180.0: plt.cm.Oranges(np.linspace(0.3, 0.9, 9))  # 9 zones - oranges
    }
    
    # Assign colors to zones
    for angle, zones in angle_groups.items():
        colors = color_schemes[angle]
        for i, zone in enumerate(sorted(zones)):
            color_mapping[int(zone)] = colors[i]
    
    print(f"Created color mapping for {len(color_mapping)} zones")
    return color_mapping, angle_groups

def create_3d_zone_visualization(sample_size=50000):
    """Create a 3D visualization showing each zone in different colors."""
    
    print(f"Creating 3D visualization with {sample_size} sample points...")
    
    # Load data
    classified_df, distribution_df, cubic_df = load_classification_data()
    color_mapping, angle_groups = create_zone_color_mapping()
    
    # Sample points for visualization
    if sample_size < len(classified_df):
        sample_df = classified_df.sample(n=sample_size, random_state=42)
    else:
        sample_df = classified_df.copy()
    
    # Create the 3D plot
    fig = plt.figure(figsize=(16, 12))
    
    # Main 3D plot
    ax1 = fig.add_subplot(221, projection='3d')
    
    # Plot each zone with its assigned color
    for angle, zones in angle_groups.items():
        for zone_id in zones:
            zone_data = sample_df[sample_df['zone_id'] == zone_id]
            if len(zone_data) > 0:
                color = color_mapping[zone_id]
                ax1.scatter(zone_data['x'], zone_data['y'], zone_data['z'], 
                           c=[color], alpha=0.6, s=1, label=f'Zone {zone_id}')
    
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title(f'3D Quaternion Zones\n({len(sample_df):,} points sampled)')
    
    # 2D projections
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)
    
    # XY projection
    for zone_id, color in color_mapping.items():
        zone_data = sample_df[sample_df['zone_id'] == zone_id]
        if len(zone_data) > 0:
            ax2.scatter(zone_data['x'], zone_data['y'], c=[color], alpha=0.6, s=0.5)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_title('XY Projection')
    ax2.grid(True, alpha=0.3)
    
    # XZ projection  
    for zone_id, color in color_mapping.items():
        zone_data = sample_df[sample_df['zone_id'] == zone_id]
        if len(zone_data) > 0:
            ax3.scatter(zone_data['x'], zone_data['z'], c=[color], alpha=0.6, s=0.5)
    ax3.set_xlabel('X')
    ax3.set_ylabel('Z')
    ax3.set_title('XZ Projection')
    ax3.grid(True, alpha=0.3)
    
    # YZ projection
    for zone_id, color in color_mapping.items():
        zone_data = sample_df[sample_df['zone_id'] == zone_id]
        if len(zone_data) > 0:
            ax4.scatter(zone_data['y'], zone_data['z'], c=[color], alpha=0.6, s=0.5)
    ax4.set_xlabel('Y')
    ax4.set_ylabel('Z')
    ax4.set_title('YZ Projection')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('quaternion_zones_colored_3d.png', dpi=300, bbox_inches='tight')
    print("Saved 3D visualization to: quaternion_zones_colored_3d.png")
    plt.show()

def create_zone_centers_visualization():
    """Visualize the zone centers (cubic group quaternions) with colors."""
    
    print("Creating zone centers visualization...")
    
    # Load data
    _, distribution_df, cubic_df = load_classification_data()
    color_mapping, angle_groups = create_zone_color_mapping()
    
    # Get active zone centers
    active_zones = sorted(distribution_df['zone_id'].values)
    active_centers = cubic_df.iloc[active_zones]
    
    # Create visualization
    fig = plt.figure(figsize=(15, 10))
    
    # 3D plot of zone centers
    ax1 = fig.add_subplot(221, projection='3d')
    
    for zone_id in active_zones:
        center = cubic_df.iloc[zone_id]
        color = color_mapping[zone_id]
        ax1.scatter([center['x']], [center['y']], [center['z']], 
                   c=[color], s=100, alpha=0.8, edgecolors='black', linewidth=1)
        
        # Add zone ID labels
        ax1.text(center['x'], center['y'], center['z'], f'{zone_id}', 
                fontsize=8, ha='center', va='center')
    
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y') 
    ax1.set_zlabel('Z')
    ax1.set_title('Zone Centers (Cubic Group Elements)')
    
    # 2D projections of zone centers
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)
    
    for zone_id in active_zones:
        center = cubic_df.iloc[zone_id]
        color = color_mapping[zone_id]
        
        # XY projection
        ax2.scatter([center['x']], [center['y']], c=[color], s=50, alpha=0.8, edgecolors='black')
        ax2.text(center['x'], center['y'], f'{zone_id}', fontsize=6, ha='center', va='center')
        
        # XZ projection
        ax3.scatter([center['x']], [center['z']], c=[color], s=50, alpha=0.8, edgecolors='black')
        ax3.text(center['x'], center['z'], f'{zone_id}', fontsize=6, ha='center', va='center')
        
        # YZ projection  
        ax4.scatter([center['y']], [center['z']], c=[color], s=50, alpha=0.8, edgecolors='black')
        ax4.text(center['y'], center['z'], f'{zone_id}', fontsize=6, ha='center', va='center')
    
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_title('XY Projection')
    ax2.grid(True, alpha=0.3)
    
    ax3.set_xlabel('X')
    ax3.set_ylabel('Z')
    ax3.set_title('XZ Projection')
    ax3.grid(True, alpha=0.3)
    
    ax4.set_xlabel('Y')
    ax4.set_ylabel('Z')
    ax4.set_title('YZ Projection')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('quaternion_zone_centers.png', dpi=300, bbox_inches='tight')
    print("Saved zone centers visualization to: quaternion_zone_centers.png")
    plt.show()

def create_legend_and_summary():
    """Create a legend showing the color coding and zone information."""
    
    print("Creating legend and summary...")
    
    # Load data
    classified_df, distribution_df, cubic_df = load_classification_data()
    color_mapping, angle_groups = create_zone_color_mapping()
    
    # Create figure with legend
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 10))
    
    # Left plot: Color legend by rotation angle
    y_pos = 0
    legend_elements = []
    
    for angle, zones in angle_groups.items():
        ax1.text(0, y_pos, f"{angle:.0f}° Rotations:", fontsize=14, fontweight='bold')
        y_pos -= 0.5
        
        for zone_id in sorted(zones):
            zone_info = distribution_df[distribution_df['zone_id'] == zone_id].iloc[0]
            cubic_info = cubic_df.iloc[zone_id]
            color = color_mapping[zone_id]
            
            # Create colored rectangle
            rect = plt.Rectangle((0.5, y_pos-0.2), 0.3, 0.3, 
                               facecolor=color, edgecolor='black', alpha=0.8)
            ax1.add_patch(rect)
            
            # Add zone information
            ax1.text(1.0, y_pos, f"Zone {zone_id:2d}: {zone_info['count']:6,} quaternions", 
                    fontsize=10, va='center')
            ax1.text(3.0, y_pos, f"[{cubic_info['w']:6.3f}, {cubic_info['x']:6.3f}, {cubic_info['y']:6.3f}, {cubic_info['z']:6.3f}]", 
                    fontsize=9, va='center', family='monospace')
            
            y_pos -= 0.4
        
        y_pos -= 0.5
    
    ax1.set_xlim(0, 6)
    ax1.set_ylim(y_pos, 1)
    ax1.set_title('Zone Color Legend', fontsize=16, fontweight='bold')
    ax1.axis('off')
    
    # Right plot: Summary statistics
    ax2.text(0.05, 0.95, 'Zone Classification Summary', fontsize=16, fontweight='bold', 
             transform=ax2.transAxes)
    
    summary_text = f"""
    Total Quaternions: {len(classified_df):,}
    Active Zones: {len(distribution_df)}/48
    
    Rotation Type Distribution:
    • Identity (0°): {len(angle_groups[0.0])} zone, {sum(distribution_df[distribution_df['angle_deg'] == 0.0]['count']):,} quaternions
    • Face (90°): {len(angle_groups[90.0])} zones, {sum(distribution_df[distribution_df['angle_deg'] == 90.0]['count']):,} quaternions  
    • Edge (120°): {len(angle_groups[120.0])} zones, {sum(distribution_df[distribution_df['angle_deg'] == 120.0]['count']):,} quaternions
    • Vertex (180°): {len(angle_groups[180.0])} zones, {sum(distribution_df[distribution_df['angle_deg'] == 180.0]['count']):,} quaternions
    
    Classification Quality:
    • Mean distance to center: {classified_df['min_distance'].mean():.6f}
    • Max distance to center: {classified_df['min_distance'].max():.6f}
    • Std deviation of population: {distribution_df['count'].std():.1f}
    
    Color Scheme:
    • Red: Identity rotation
    • Blues: 90° face rotations
    • Greens: 120° edge rotations
    • Oranges: 180° vertex rotations
    """
    
    ax2.text(0.05, 0.85, summary_text, transform=ax2.transAxes, fontsize=11, 
             verticalalignment='top', fontfamily='monospace')
    ax2.axis('off')
    
    plt.tight_layout()
    plt.savefig('quaternion_zones_legend.png', dpi=300, bbox_inches='tight')
    print("Saved legend and summary to: quaternion_zones_legend.png")
    plt.show()

def create_interactive_zone_browser():
    """Create individual plots for each zone for detailed inspection."""
    
    print("Creating individual zone visualizations...")
    
    # Load data
    classified_df, distribution_df, cubic_df = load_classification_data()
    color_mapping, angle_groups = create_zone_color_mapping()
    
    # Create a grid of subplots for each zone
    n_zones = len(distribution_df)
    n_cols = 6
    n_rows = (n_zones + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, n_rows*3))
    axes = axes.flatten() if n_zones > 1 else [axes]
    
    for i, zone_id in enumerate(sorted(distribution_df['zone_id'].values)):
        ax = axes[i]
        
        # Get zone data
        zone_data = classified_df[classified_df['zone_id'] == zone_id]
        zone_info = distribution_df[distribution_df['zone_id'] == zone_id].iloc[0]
        cubic_info = cubic_df.iloc[zone_id]
        color = color_mapping[zone_id]
        
        # Sample points for this zone if too many
        if len(zone_data) > 1000:
            zone_sample = zone_data.sample(n=1000, random_state=42)
        else:
            zone_sample = zone_data
        
        # Plot XY projection
        ax.scatter(zone_sample['x'], zone_sample['y'], c=[color], alpha=0.6, s=1)
        
        # Plot zone center
        ax.scatter([cubic_info['x']], [cubic_info['y']], c='black', s=50, marker='x', linewidth=3)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title(f'Zone {zone_id}\n{zone_info["angle_deg"]:.0f}° rot, {zone_info["count"]:,} points')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')
    
    # Hide unused subplots
    for i in range(len(distribution_df), len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('individual_zone_plots.png', dpi=300, bbox_inches='tight')
    print("Saved individual zone plots to: individual_zone_plots.png")
    plt.show()

def main():
    """Main function to create all visualizations."""
    
    print("=== QUATERNION ZONE VISUALIZATION ===")
    print("Creating comprehensive visualizations with separate colors for each zone...")
    
    try:
        # Create all visualizations
        create_3d_zone_visualization(sample_size=30000)
        create_zone_centers_visualization()
        create_legend_and_summary()
        create_interactive_zone_browser()
        
        print("\n✅ All visualizations created successfully!")
        print("\nGenerated files:")
        print("  • quaternion_zones_colored_3d.png - 3D and 2D projections")
        print("  • quaternion_zone_centers.png - Zone center locations")
        print("  • quaternion_zones_legend.png - Color legend and summary")
        print("  • individual_zone_plots.png - Individual zone details")
        
    except Exception as e:
        print(f"❌ Error creating visualizations: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
