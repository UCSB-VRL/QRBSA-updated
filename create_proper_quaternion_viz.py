#!/usr/bin/env python3
"""
Create a corrected interactive visualization that properly shows quaternion zones.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

def create_proper_quaternion_visualization():
    """Create a proper interactive quaternion visualization."""
    
    print("=== CREATING PROPER QUATERNION VISUALIZATION ===")
    
    # Load data
    classified_df = pd.read_csv("fibonacci_zone_classification.csv")
    distribution_df = pd.read_csv("fibonacci_zone_classification_distribution.csv")
    cubic_df = pd.read_csv("cubic_group_quaternions_48.csv")
    
    # Sample data for performance
    sample_df = classified_df.sample(n=25000, random_state=42)
    
    print(f"Sample size: {len(sample_df)} quaternions")
    print(f"Active zones: {len(distribution_df)}")
    
    # Verify the issue
    print("\n=== THE ISSUE EXPLAINED ===")
    print("Zone 0 (Identity rotation):")
    zone_0 = sample_df[sample_df['zone_id'] == 0]
    print(f"  Center: [1, 0, 0, 0] → Identity quaternion")
    print(f"  Sample points have w ≈ 0.86, small (x,y,z) → appear near origin in XYZ space")
    print(f"  (x,y,z) norm range: [{np.sqrt(zone_0['x']**2 + zone_0['y']**2 + zone_0['z']**2).min():.3f}, {np.sqrt(zone_0['x']**2 + zone_0['y']**2 + zone_0['z']**2).max():.3f}]")
    
    print("\nZone 22 (180° rotation):")
    zone_22 = sample_df[sample_df['zone_id'] == 22]
    print(f"  Center: [0, -0.707, 0.707, 0] → 180° rotation")
    print(f"  Sample points have w ≈ 0.02, large (x,y,z) → appear near sphere boundary")
    print(f"  (x,y,z) norm range: [{np.sqrt(zone_22['x']**2 + zone_22['y']**2 + zone_22['z']**2).min():.3f}, {np.sqrt(zone_22['x']**2 + zone_22['y']**2 + zone_22['z']**2).max():.3f}]")
    
    # Create visualization with multiple methods
    fig = go.Figure()
    
    # Get zone colors
    active_zones = sorted(distribution_df['zone_id'].astype(int).values)
    colors = px.colors.qualitative.Light24[:len(active_zones)]
    zone_colors = {zone: colors[i] for i, zone in enumerate(active_zones)}
    
    # Method 1: Show the "incorrect" visualization (what user sees)
    print("\n=== Creating visualization with explanation ===")
    
    for i, zone_id in enumerate(active_zones):
        zone_data = sample_df[sample_df['zone_id'] == zone_id]
        if len(zone_data) == 0:
            continue
            
        zone_info = distribution_df[distribution_df['zone_id'] == zone_id].iloc[0]
        center = cubic_df.iloc[zone_id]
        
        # Calculate rotation angle and axis for each point
        angles = 2 * np.arccos(np.clip(zone_data['w'], 0, 1))
        
        fig.add_trace(go.Scatter3d(
            x=zone_data['x'],
            y=zone_data['y'],
            z=zone_data['z'],
            mode='markers',
            marker=dict(
                size=3,
                color=zone_colors[zone_id],
                opacity=0.7
            ),
            name=f'Zone {zone_id} ({zone_info["angle_deg"]:.0f}°)',
            text=[f'Zone: {zone_id}<br>Rotation: {zone_info["angle_deg"]:.0f}°<br>w={w:.3f}<br>||(x,y,z)||={np.sqrt(x*x+y*y+z*z):.3f}<br>Quaternion norm: {np.sqrt(w*w+x*x+y*y+z*z):.3f}'
                  for w, x, y, z in zip(zone_data['w'], zone_data['x'], zone_data['y'], zone_data['z'])],
            hovertemplate='<b>%{text}</b><br>xyz: (%{x:.3f}, %{y:.3f}, %{z:.3f})<extra></extra>'
        ))
    
    # Add zone centers
    centers = cubic_df.iloc[active_zones]
    fig.add_trace(go.Scatter3d(
        x=centers['x'],
        y=centers['y'],
        z=centers['z'],
        mode='markers+text',
        marker=dict(
            size=12,
            color='black',
            symbol='diamond',
            line=dict(width=3, color='white'),
            opacity=1.0
        ),
        text=[f'{zone_id}' for zone_id in active_zones],
        textposition='middle center',
        textfont=dict(color='white', size=10, family='Arial Black'),
        name='Zone Centers',
        hovertemplate='<b>Zone Center %{text}</b><br>Quaternion: [w=%{customdata[0]:.3f}, x=%{x:.3f}, y=%{y:.3f}, z=%{z:.3f}]<br>Angle: %{customdata[1]:.1f}°<extra></extra>',
        customdata=[[centers.iloc[i]['w'], centers.iloc[i]['angle_deg']] for i in range(len(centers))]
    ))
    
    # Add unit sphere wireframe for reference
    u = np.linspace(0, 2 * np.pi, 20)
    v = np.linspace(0, np.pi, 10)
    x_sphere = np.outer(np.cos(u), np.sin(v))
    y_sphere = np.outer(np.sin(u), np.sin(v))
    z_sphere = np.outer(np.ones(np.size(u)), np.cos(v))
    
    fig.add_trace(go.Surface(
        x=x_sphere, y=y_sphere, z=z_sphere,
        opacity=0.1,
        colorscale='Greys',
        showscale=False,
        name='Unit Sphere Reference',
        hovertemplate='Unit sphere reference<extra></extra>'
    ))
    
    # Update layout
    fig.update_layout(
        title=dict(
            text='Quaternion Zone Visualization Explained<br><sub>🔍 Why zones appear at different distances from origin<br>All quaternions are normalized, but (x,y,z) components vary!</sub>',
            x=0.5,
            font=dict(size=16)
        ),
        scene=dict(
            xaxis_title='Quaternion X Component',
            yaxis_title='Quaternion Y Component',
            zaxis_title='Quaternion Z Component',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            ),
            aspectmode='cube',
            bgcolor='rgba(240,248,255,0.8)'
        ),
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.98,
            xanchor="left",
            x=1.02,
            font=dict(size=10),
            bgcolor='rgba(255,255,255,0.9)',
            bordercolor='rgba(0,0,0,0.2)',
            borderwidth=1
        ),
        annotations=[
            dict(
                text="<b>Key Insight:</b><br>• All quaternions have ||q|| = 1<br>• But ||(x,y,z)|| varies from 0 to 1<br>• Identity rotations: small (x,y,z)<br>• 180° rotations: large (x,y,z)<br>• This creates the distance effect!",
                showarrow=False,
                x=0.02, y=0.98,
                xref="paper", yref="paper",
                xanchor="left", yanchor="top",
                bgcolor="rgba(255,255,200,0.8)",
                bordercolor="orange",
                borderwidth=2,
                font=dict(size=12)
            )
        ],
        width=1400,
        height=900,
        margin=dict(r=250, l=0, t=120, b=0)
    )
    
    return fig

def create_axis_angle_visualization():
    """Create a proper axis-angle visualization."""
    
    print("\n=== Creating Axis-Angle Visualization ===")
    
    # Load data
    classified_df = pd.read_csv("fibonacci_zone_classification.csv")
    distribution_df = pd.read_csv("fibonacci_zone_classification_distribution.csv")
    
    # Sample data
    sample_df = classified_df.sample(n=20000, random_state=42)
    
    # Convert to axis-angle representation
    angles = 2 * np.arccos(np.clip(sample_df['w'], 0, 1))
    
    # Compute rotation axes
    vector_norms = np.sqrt(sample_df['x']**2 + sample_df['y']**2 + sample_df['z']**2)
    mask = vector_norms > 1e-8
    
    axis_x = np.zeros_like(sample_df['x'])
    axis_y = np.zeros_like(sample_df['y'])
    axis_z = np.zeros_like(sample_df['z'])
    
    axis_x[mask] = sample_df['x'][mask] / vector_norms[mask]
    axis_y[mask] = sample_df['y'][mask] / vector_norms[mask]
    axis_z[mask] = sample_df['z'][mask] / vector_norms[mask]
    
    # Create figure
    fig = go.Figure()
    
    # Color zones
    active_zones = sorted(distribution_df['zone_id'].astype(int).values)
    colors = px.colors.qualitative.Set3[:len(active_zones)]
    zone_colors = {zone: colors[i % len(colors)] for i, zone in enumerate(active_zones)}
    
    for zone_id in active_zones:
        zone_data = sample_df[sample_df['zone_id'] == zone_id]
        zone_mask = sample_df['zone_id'] == zone_id
        
        if len(zone_data) == 0:
            continue
        
        fig.add_trace(go.Scatter3d(
            x=axis_x[zone_mask],
            y=axis_y[zone_mask],
            z=axis_z[zone_mask],
            mode='markers',
            marker=dict(
                size=3,
                color=zone_colors[zone_id],
                opacity=0.7
            ),
            name=f'Zone {zone_id}',
            text=[f'Zone: {zone_id}<br>Angle: {a:.1f}°<br>Axis: ({ax:.3f}, {ay:.3f}, {az:.3f})'
                  for a, ax, ay, az in zip(np.degrees(angles[zone_mask]), 
                                          axis_x[zone_mask], 
                                          axis_y[zone_mask], 
                                          axis_z[zone_mask])],
            hovertemplate='<b>%{text}</b><extra></extra>'
        ))
    
    fig.update_layout(
        title='Quaternions as Rotation Axes<br><sub>Proper representation on unit sphere</sub>',
        scene=dict(
            xaxis_title='Rotation Axis X',
            yaxis_title='Rotation Axis Y', 
            zaxis_title='Rotation Axis Z',
            aspectmode='cube'
        ),
        width=1200,
        height=800
    )
    
    return fig

def main():
    """Main function."""
    
    # Create the explanatory visualization
    fig1 = create_proper_quaternion_visualization()
    fig1.write_html("quaternion_zones_explained.html")
    print("✅ Saved: quaternion_zones_explained.html")
    
    # Create axis-angle visualization  
    fig2 = create_axis_angle_visualization()
    fig2.write_html("quaternion_axis_angle_proper.html")
    print("✅ Saved: quaternion_axis_angle_proper.html")
    
    print(f"""
=== SUMMARY ===

✅ Your observation is CORRECT!
   • All quaternions ARE normalized (|q| = 1)
   • They lie on the 4D unit hypersphere

❌ The visualization ISSUE:
   • We plot only (x,y,z) components in 3D
   • Zone 0 (identity): w≈1, (x,y,z)≈0 → near origin
   • Zone 22 (180°): w≈0, ||(x,y,z)||≈1 → near sphere boundary

🔧 PROPER SOLUTIONS:
   1. quaternion_zones_explained.html - Shows the issue clearly
   2. quaternion_axis_angle_proper.html - Proper axis-angle representation
   
🎯 Key insight: The apparent "distance from origin" is NOT the quaternion
   lying at different positions, but the (x,y,z) COMPONENTS having different
   magnitudes while maintaining w² + x² + y² + z² = 1!
""")

if __name__ == "__main__":
    main()
