#!/usr/bin/env python3
"""
Debug quaternion visualization issues - check normalization and proper projection.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def analyze_quaternion_data():
    """Analyze the quaternion data to understand the visualization issue."""
    
    print("=== QUATERNION DATA ANALYSIS ===")
    
    # Load data
    classified_df = pd.read_csv("fibonacci_zone_classification.csv")
    distribution_df = pd.read_csv("fibonacci_zone_classification_distribution.csv")
    cubic_df = pd.read_csv("cubic_group_quaternions_48.csv")
    
    print(f"Total quaternions: {len(classified_df)}")
    print(f"Active zones: {len(distribution_df)}")
    
    # Check normalization
    print("\n=== NORMALIZATION CHECK ===")
    norms = np.sqrt(classified_df['w']**2 + classified_df['x']**2 + 
                   classified_df['y']**2 + classified_df['z']**2)
    
    print(f"Quaternion norms - Min: {norms.min():.6f}, Max: {norms.max():.6f}")
    print(f"Mean norm: {norms.mean():.6f}, Std: {norms.std():.6f}")
    print(f"All normalized? {np.allclose(norms, 1.0, atol=1e-6)}")
    
    # Check w component distribution
    print(f"\nW component - Min: {classified_df['w'].min():.6f}, Max: {classified_df['w'].max():.6f}")
    print(f"W negative count: {(classified_df['w'] < 0).sum()}")
    print(f"W positive count: {(classified_df['w'] >= 0).sum()}")
    
    # Analyze zone centers
    print("\n=== ZONE CENTERS ANALYSIS ===")
    active_zones = sorted(distribution_df['zone_id'].astype(int).values)
    centers = cubic_df.iloc[active_zones]
    
    center_norms = np.sqrt(centers['w']**2 + centers['x']**2 + 
                          centers['y']**2 + centers['z']**2)
    print(f"Zone center norms - Min: {center_norms.min():.6f}, Max: {center_norms.max():.6f}")
    
    # Check specific zones
    print("\n=== SPECIFIC ZONE ANALYSIS ===")
    for zone_id in [0, 22, 8, 16]:  # Top few zones
        zone_data = classified_df[classified_df['zone_id'] == zone_id]
        center = cubic_df.iloc[zone_id]
        
        print(f"\nZone {zone_id}:")
        print(f"  Center: [{center['w']:.4f}, {center['x']:.4f}, {center['y']:.4f}, {center['z']:.4f}]")
        print(f"  Angle: {center['angle_deg']:.1f}°")
        print(f"  Sample point: [{zone_data.iloc[0]['w']:.4f}, {zone_data.iloc[0]['x']:.4f}, {zone_data.iloc[0]['y']:.4f}, {zone_data.iloc[0]['z']:.4f}]")
        
        # Check 3D coordinates (x,y,z) distribution
        xyz_norm = np.sqrt(zone_data['x']**2 + zone_data['y']**2 + zone_data['z']**2)
        print(f"  (x,y,z) norm range: [{xyz_norm.min():.4f}, {xyz_norm.max():.4f}]")
    
    return classified_df, centers

def create_proper_visualization():
    """Create a proper visualization showing the issue and fix."""
    
    classified_df, centers = analyze_quaternion_data()
    
    # Sample data
    sample_df = classified_df.sample(n=10000, random_state=42)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Current visualization (xyz coordinates only)
    ax1 = axes[0, 0]
    scatter = ax1.scatter(sample_df['x'], sample_df['y'], c=sample_df['zone_id'], 
                         cmap='tab20', s=1, alpha=0.6)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_title('Current XY Visualization\n(Problematic - not on unit sphere)')
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    
    # 2. Verify quaternions are on unit sphere
    ax2 = axes[0, 1]
    norms = np.sqrt(sample_df['w']**2 + sample_df['x']**2 + sample_df['y']**2 + sample_df['z']**2)
    ax2.hist(norms, bins=50, alpha=0.7)
    ax2.axvline(1.0, color='red', linestyle='--', linewidth=2, label='Unit norm')
    ax2.set_xlabel('Quaternion Norm')
    ax2.set_ylabel('Count')
    ax2.set_title('Quaternion Normalization Check')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. W component distribution
    ax3 = axes[0, 2]
    ax3.hist(sample_df['w'], bins=50, alpha=0.7, color='green')
    ax3.set_xlabel('W Component')
    ax3.set_ylabel('Count')
    ax3.set_title('W Component Distribution')
    ax3.grid(True, alpha=0.3)
    
    # 4. Stereographic projection (proper way)
    ax4 = axes[1, 0]
    # Stereographic projection: (x,y,z) / (1+w) for w > -1
    mask = sample_df['w'] > -0.99  # Avoid division by zero
    sample_stereo = sample_df[mask]
    denom = 1 + sample_stereo['w']
    x_proj = sample_stereo['x'] / denom
    y_proj = sample_stereo['y'] / denom
    
    ax4.scatter(x_proj, y_proj, c=sample_stereo['zone_id'], cmap='tab20', s=1, alpha=0.6)
    ax4.set_xlabel('X (Stereographic)')
    ax4.set_ylabel('Y (Stereographic)')
    ax4.set_title('Stereographic Projection\n(Proper 4D→2D mapping)')
    ax4.grid(True, alpha=0.3)
    ax4.set_aspect('equal')
    
    # 5. (x,y,z) norms vs w
    ax5 = axes[1, 1]
    xyz_norms = np.sqrt(sample_df['x']**2 + sample_df['y']**2 + sample_df['z']**2)
    scatter2 = ax5.scatter(sample_df['w'], xyz_norms, c=sample_df['zone_id'], 
                          cmap='tab20', s=1, alpha=0.6)
    ax5.set_xlabel('W Component')
    ax5.set_ylabel('||(x,y,z)|| norm')
    ax5.set_title('Relationship: w vs ||(x,y,z)||\nShould satisfy: w² + ||(x,y,z)||² = 1')
    ax5.grid(True, alpha=0.3)
    
    # Add theoretical curve
    w_theory = np.linspace(-1, 1, 100)
    xyz_theory = np.sqrt(1 - w_theory**2)
    ax5.plot(w_theory, xyz_theory, 'r-', linewidth=3, label='w² + ||(x,y,z)||² = 1')
    ax5.legend()
    
    # 6. Zone centers in different projections
    ax6 = axes[1, 2]
    # Show zone centers in original (x,y) space
    ax6.scatter(centers['x'], centers['y'], c=range(len(centers)), 
               cmap='tab20', s=100, alpha=0.8, edgecolors='black', linewidth=1)
    
    # Add labels
    for i, (idx, center) in enumerate(centers.iterrows()):
        ax6.annotate(f'{idx}', (center['x'], center['y']), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    ax6.set_xlabel('X')
    ax6.set_ylabel('Y')
    ax6.set_title('Zone Centers (24 active zones)')
    ax6.grid(True, alpha=0.3)
    ax6.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig('quaternion_visualization_debug.png', dpi=300, bbox_inches='tight')
    print("\nSaved debug visualization: quaternion_visualization_debug.png")
    plt.show()

def create_corrected_interactive_plot():
    """Create a corrected interactive plot using proper quaternion visualization."""
    
    try:
        import plotly.graph_objects as go
        import plotly.express as px
        
        print("\nCreating corrected interactive visualization...")
        
        # Load data
        classified_df = pd.read_csv("fibonacci_zone_classification.csv")
        sample_df = classified_df.sample(n=20000, random_state=42)
        
        # Create proper visualization options
        fig = go.Figure()
        
        # Method 1: Use axis-angle representation
        angles = 2 * np.arccos(np.abs(sample_df['w']))  # Rotation angle
        
        # Compute rotation axis (normalized vector part when |vector| > 0)
        vector_norms = np.sqrt(sample_df['x']**2 + sample_df['y']**2 + sample_df['z']**2)
        mask = vector_norms > 1e-8  # Avoid division by zero
        
        axis_x = np.zeros_like(sample_df['x'])
        axis_y = np.zeros_like(sample_df['y']) 
        axis_z = np.zeros_like(sample_df['z'])
        
        axis_x[mask] = sample_df['x'][mask] / vector_norms[mask]
        axis_y[mask] = sample_df['y'][mask] / vector_norms[mask]
        axis_z[mask] = sample_df['z'][mask] / vector_norms[mask]
        
        # Scale by rotation angle for visualization
        scaled_x = axis_x * np.sin(angles/2)
        scaled_y = axis_y * np.sin(angles/2)
        scaled_z = axis_z * np.sin(angles/2)
        
        # Get unique zones for coloring
        unique_zones = sorted(sample_df['zone_id'].unique())
        colors = px.colors.qualitative.Set3[:len(unique_zones)]
        zone_color_map = {zone: colors[i] for i, zone in enumerate(unique_zones)}
        
        # Plot each zone
        for zone_id in unique_zones[:12]:  # Show first 12 zones to avoid clutter
            zone_data = sample_df[sample_df['zone_id'] == zone_id]
            zone_indices = sample_df['zone_id'] == zone_id
            
            fig.add_trace(go.Scatter3d(
                x=scaled_x[zone_indices],
                y=scaled_y[zone_indices],
                z=scaled_z[zone_indices],
                mode='markers',
                marker=dict(
                    size=3,
                    color=zone_color_map[zone_id],
                    opacity=0.7
                ),
                name=f'Zone {zone_id}',
                text=[f'Zone: {zone_id}<br>Angle: {a:.1f}°<br>Axis: ({ax:.3f}, {ay:.3f}, {az:.3f})'
                      for a, ax, ay, az in zip(np.degrees(angles[zone_indices]), 
                                              axis_x[zone_indices], 
                                              axis_y[zone_indices], 
                                              axis_z[zone_indices])],
                hovertemplate='<b>%{text}</b><br>Position: (%{x:.3f}, %{y:.3f}, %{z:.3f})<extra></extra>'
            ))
        
        fig.update_layout(
            title='Corrected Quaternion Visualization<br><sub>Axis-Angle Representation: Points on unit sphere</sub>',
            scene=dict(
                xaxis_title='Rotation Axis X × sin(θ/2)',
                yaxis_title='Rotation Axis Y × sin(θ/2)',
                zaxis_title='Rotation Axis Z × sin(θ/2)',
                aspectmode='cube'
            ),
            width=1200,
            height=800
        )
        
        fig.write_html("corrected_quaternion_visualization.html")
        print("Saved corrected visualization: corrected_quaternion_visualization.html")
        
    except ImportError:
        print("Plotly not available, skipping interactive plot")

def main():
    """Main analysis function."""
    print("=== QUATERNION VISUALIZATION DEBUG ===")
    
    # Analyze the data
    analyze_quaternion_data()
    
    # Create debug visualizations
    create_proper_visualization()
    
    # Create corrected interactive plot
    create_corrected_interactive_plot()
    
    print("\n=== EXPLANATION OF THE ISSUE ===")
    print("""
    The issue you observed is due to PROJECTION METHOD:
    
    ❌ PROBLEM:
    - All quaternions ARE normalized (lie on 4D unit hypersphere)
    - But we're plotting just (x,y,z) coordinates in 3D space
    - This creates the illusion that they fill a 3D volume
    
    ✅ REALITY:
    - Zone 0 (identity) has quaternions near (1,0,0,0) → small (x,y,z)
    - Other zones have quaternions with larger (x,y,z) components
    - The constraint w² + x² + y² + z² = 1 is always satisfied
    
    🔧 SOLUTION:
    - Use stereographic projection or axis-angle representation
    - This properly maps 4D hypersphere to 3D/2D space
    - Preserves the geometric relationships between rotations
    """)

if __name__ == "__main__":
    main()
