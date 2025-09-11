#!/usr/bin/env python3
"""
Simple script to create a clear quaternion zone visualization with distinct colors.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def create_simple_colored_visualization():
    """Create a simple but effective colored visualization of quaternion zones."""
    
    print("Creating simple colored visualization...")
    
    # Load data
    classified_df = pd.read_csv("fibonacci_zone_classification.csv")
    distribution_df = pd.read_csv("fibonacci_zone_classification_distribution.csv") 
    cubic_df = pd.read_csv("cubic_group_quaternions_48.csv")
    
    # Sample data for visualization (50k points for good resolution but manageable rendering)
    sample_size = 50000
    sample_df = classified_df.sample(n=sample_size, random_state=42)
    
    # Get active zones and create a simple color palette
    active_zones = sorted(distribution_df['zone_id'].astype(int).values)
    n_zones = len(active_zones)
    
    # Use a colormap that provides good contrast
    colors = plt.cm.tab20(np.linspace(0, 1, n_zones))
    zone_colors = {zone_id: colors[i] for i, zone_id in enumerate(active_zones)}
    
    # Create the main visualization
    fig = plt.figure(figsize=(20, 15))
    
    # 3D scatter plot
    ax1 = fig.add_subplot(221, projection='3d')
    
    for i, zone_id in enumerate(active_zones):
        zone_data = sample_df[sample_df['zone_id'] == zone_id]
        if len(zone_data) > 0:
            ax1.scatter(zone_data['x'], zone_data['y'], zone_data['z'],
                       c=[zone_colors[zone_id]], s=3, alpha=0.7, 
                       label=f'Zone {zone_id}')
    
    ax1.set_xlabel('Quaternion X', fontsize=12)
    ax1.set_ylabel('Quaternion Y', fontsize=12)
    ax1.set_zlabel('Quaternion Z', fontsize=12)
    ax1.set_title(f'3D Quaternion Space\n24 Zones Colored ({sample_size:,} points)', fontsize=14, pad=20)
    
    # XY projection
    ax2 = fig.add_subplot(222)
    for zone_id in active_zones:
        zone_data = sample_df[sample_df['zone_id'] == zone_id]
        if len(zone_data) > 0:
            ax2.scatter(zone_data['x'], zone_data['y'], 
                       c=[zone_colors[zone_id]], s=2, alpha=0.7)
    ax2.set_xlabel('Quaternion X', fontsize=12)
    ax2.set_ylabel('Quaternion Y', fontsize=12)
    ax2.set_title('XY Projection', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    # XZ projection
    ax3 = fig.add_subplot(223)
    for zone_id in active_zones:
        zone_data = sample_df[sample_df['zone_id'] == zone_id]
        if len(zone_data) > 0:
            ax3.scatter(zone_data['x'], zone_data['z'],
                       c=[zone_colors[zone_id]], s=2, alpha=0.7)
    ax3.set_xlabel('Quaternion X', fontsize=12)
    ax3.set_ylabel('Quaternion Z', fontsize=12)
    ax3.set_title('XZ Projection', fontsize=14)
    ax3.grid(True, alpha=0.3)
    
    # YZ projection  
    ax4 = fig.add_subplot(224)
    for zone_id in active_zones:
        zone_data = sample_df[sample_df['zone_id'] == zone_id]
        if len(zone_data) > 0:
            ax4.scatter(zone_data['y'], zone_data['z'],
                       c=[zone_colors[zone_id]], s=2, alpha=0.7)
    ax4.set_xlabel('Quaternion Y', fontsize=12)
    ax4.set_ylabel('Quaternion Z', fontsize=12) 
    ax4.set_title('YZ Projection', fontsize=14)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('simple_quaternion_zones_colored.png', dpi=300, bbox_inches='tight')
    print("Saved simple colored visualization to: simple_quaternion_zones_colored.png")
    plt.show()
    
    return zone_colors

def create_zone_info_table():
    """Create a detailed table showing zone information with colors."""
    
    print("Creating zone information table...")
    
    # Load data
    distribution_df = pd.read_csv("fibonacci_zone_classification_distribution.csv")
    cubic_df = pd.read_csv("cubic_group_quaternions_48.csv")
    
    # Create a comprehensive table
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare table data
    table_data = []
    active_zones = sorted(distribution_df['zone_id'].astype(int).values)
    colors = plt.cm.tab20(np.linspace(0, 1, len(active_zones)))
    
    # Group by rotation angle for better organization
    angle_groups = distribution_df.groupby('angle_deg')
    
    current_row = 0
    for angle, group in angle_groups:
        # Add angle header
        table_data.append([f"{angle:.0f}° ROTATIONS", "", "", "", "", "", ""])
        
        for _, row in group.iterrows():
            zone_id = int(row['zone_id'])
            cubic_info = cubic_df.iloc[zone_id]
            
            table_data.append([
                f"Zone {zone_id:2d}",
                f"{row['count']:7,}",
                f"{row['percentage']:5.2f}%",
                f"{cubic_info['w']:7.4f}",
                f"{cubic_info['x']:7.4f}", 
                f"{cubic_info['y']:7.4f}",
                f"{cubic_info['z']:7.4f}"
            ])
    
    # Create table
    table = ax.table(cellText=table_data,
                    colLabels=['Zone', 'Count', '%', 'W', 'X', 'Y', 'Z'],
                    cellLoc='center',
                    loc='center')
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    
    # Color code the zones
    zone_idx = 0
    for i, row_data in enumerate(table_data):
        if "ROTATIONS" in row_data[0]:
            # Header row - make it bold
            for j in range(7):
                table[(i+1, j)].set_facecolor('#E0E0E0')
                table[(i+1, j)].set_text_props(weight='bold')
        elif row_data[0].startswith("Zone"):
            # Zone data row - color it
            color = colors[zone_idx % len(colors)]
            for j in range(7):
                table[(i+1, j)].set_facecolor(color)
                table[(i+1, j)].set_alpha(0.3)
            zone_idx += 1
    
    plt.title('Quaternion Zone Information Table\n24 Active Zones from Cubic Group Symmetries', 
              fontsize=16, fontweight='bold', pad=20)
    
    plt.savefig('quaternion_zone_table.png', dpi=300, bbox_inches='tight')
    print("Saved zone information table to: quaternion_zone_table.png")
    plt.show()

def create_rotation_type_visualization():
    """Create a visualization showing the different rotation types."""
    
    print("Creating rotation type visualization...")
    
    # Load data
    classified_df = pd.read_csv("fibonacci_zone_classification.csv")
    distribution_df = pd.read_csv("fibonacci_zone_classification_distribution.csv")
    
    # Sample data
    sample_df = classified_df.sample(n=30000, random_state=42)
    
    # Group zones by rotation angle
    angle_groups = distribution_df.groupby('angle_deg')['zone_id'].apply(list).to_dict()
    
    # Create subplots for each rotation type
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    rotation_colors = {
        0.0: 'red',
        90.0: 'blue', 
        120.0: 'green',
        180.0: 'orange'
    }
    
    for i, (angle, zones) in enumerate(sorted(angle_groups.items())):
        ax = axes[i]
        
        # Get all quaternions for this rotation angle
        angle_data = sample_df[sample_df['zone_id'].isin(zones)]
        
        # Create different colors for zones within this angle
        zone_colors = plt.cm.get_cmap('Set3')(np.linspace(0, 1, len(zones)))
        
        for j, zone_id in enumerate(zones):
            zone_data = angle_data[angle_data['zone_id'] == zone_id]
            if len(zone_data) > 0:
                ax.scatter(zone_data['x'], zone_data['y'], 
                          c=[zone_colors[j]], s=3, alpha=0.7, 
                          label=f'Zone {zone_id}')
        
        ax.set_xlabel('Quaternion X')
        ax.set_ylabel('Quaternion Y')
        ax.set_title(f'{angle:.0f}° Rotations\n{len(zones)} zones, {len(angle_data):,} points')
        ax.grid(True, alpha=0.3)
        
        # Add legend if not too many zones
        if len(zones) <= 8:
            ax.legend(markerscale=2, fontsize=8)
    
    plt.tight_layout()
    plt.savefig('quaternion_rotation_types.png', dpi=300, bbox_inches='tight')
    print("Saved rotation type visualization to: quaternion_rotation_types.png")
    plt.show()

def main():
    """Main function to create all simple visualizations."""
    
    print("=== SIMPLE QUATERNION ZONE VISUALIZATION ===")
    
    try:
        # Create visualizations
        zone_colors = create_simple_colored_visualization()
        create_zone_info_table()
        create_rotation_type_visualization()
        
        print("\n✅ Simple visualizations created successfully!")
        print("\nGenerated files:")
        print("  • simple_quaternion_zones_colored.png - Main 3D and 2D views")
        print("  • quaternion_zone_table.png - Detailed zone information")
        print("  • quaternion_rotation_types.png - Grouped by rotation angle")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
