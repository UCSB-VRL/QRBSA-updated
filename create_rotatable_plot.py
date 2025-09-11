#!/usr/bin/env python3
"""
Simple script to create and display a single interactive 3D plot of quaternion zones.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import webbrowser
import os

def create_simple_interactive_plot():
    """Create a simple interactive 3D plot with zone colors."""
    print("Creating simple interactive 3D plot...")
    
    # Load data
    classified_df = pd.read_csv("fibonacci_zone_classification.csv")
    distribution_df = pd.read_csv("fibonacci_zone_classification_distribution.csv")
    cubic_df = pd.read_csv("cubic_group_quaternions_48.csv")
    
    # Sample data for performance (30k points)
    sample_df = classified_df.sample(n=30000, random_state=42)
    
    # Create color mapping
    active_zones = sorted(distribution_df['zone_id'].astype(int).values)
    colors = px.colors.qualitative.Set3[:len(active_zones)]  # Use Set3 color palette
    zone_colors = {zone_id: colors[i] for i, zone_id in enumerate(active_zones)}
    
    # Create the figure
    fig = go.Figure()
    
    # Add quaternion points for each zone
    for zone_id in active_zones:
        zone_data = sample_df[sample_df['zone_id'] == zone_id]
        if len(zone_data) > 0:
            # Get zone info
            zone_info = distribution_df[distribution_df['zone_id'] == zone_id].iloc[0]
            center_info = cubic_df.iloc[zone_id]
            
            fig.add_trace(go.Scatter3d(
                x=zone_data['x'],
                y=zone_data['y'],
                z=zone_data['z'],
                mode='markers',
                marker=dict(
                    size=3,
                    color=zone_colors[zone_id],
                    opacity=0.8
                ),
                name=f'Zone {zone_id} ({zone_info["angle_deg"]:.0f}°)',
                text=[f'Zone: {zone_id}<br>Rotation: {zone_info["angle_deg"]:.0f}°<br>Count: {zone_info["count"]:,}<br>Distance: {d:.4f}' 
                      for d in zone_data['min_distance']],
                hovertemplate='<b>%{text}</b><br>Position: (%{x:.3f}, %{y:.3f}, %{z:.3f})<extra></extra>'
            ))
    
    # Add zone centers as black diamonds
    centers = cubic_df.iloc[active_zones]
    fig.add_trace(go.Scatter3d(
        x=centers['x'],
        y=centers['y'],
        z=centers['z'],
        mode='markers+text',
        marker=dict(
            size=10,
            color='black',
            symbol='diamond',
            line=dict(width=2, color='white'),
            opacity=1.0
        ),
        text=[str(zone_id) for zone_id in active_zones],
        textposition='middle center',
        textfont=dict(color='white', size=10),
        name='Zone Centers',
        hovertemplate='<b>Zone Center %{text}</b><br>Quaternion: (%{x:.3f}, %{y:.3f}, %{z:.3f})<extra></extra>'
    ))
    
    # Update layout for better interaction
    fig.update_layout(
        title=dict(
            text='🔄 Interactive Quaternion Zone Visualization<br><sub>🖱️ Drag to rotate • 🔍 Scroll to zoom • 👆 Click legend to show/hide zones</sub>',
            x=0.5,
            font=dict(size=18),
            pad=dict(t=20)
        ),
        scene=dict(
            xaxis_title=dict(text='Quaternion X', font=dict(size=14)),
            yaxis_title=dict(text='Quaternion Y', font=dict(size=14)),
            zaxis_title=dict(text='Quaternion Z', font=dict(size=14)),
            camera=dict(
                eye=dict(x=1.3, y=1.3, z=1.3),
                center=dict(x=0, y=0, z=0)
            ),
            aspectmode='cube',
            bgcolor='rgba(240,240,240,0.1)'
        ),
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.98,
            xanchor="left",
            x=1.02,
            font=dict(size=11),
            itemclick="toggle",
            itemdoubleclick="toggleothers",
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='rgba(0,0,0,0.2)',
            borderwidth=1
        ),
        width=1300,
        height=800,
        margin=dict(r=250, l=50, t=120, b=50),
        font=dict(family="Arial, sans-serif", size=12)
    )
    
    return fig

def save_and_open_plot():
    """Create, save, and optionally open the interactive plot."""
    try:
        # Create the plot
        fig = create_simple_interactive_plot()
        
        # Save as HTML
        filename = "rotatable_quaternion_zones.html"
        fig.write_html(filename)
        print(f"✅ Saved interactive plot: {filename}")
        
        # Get file info
        file_size = os.path.getsize(filename) / (1024 * 1024)  # MB
        full_path = os.path.abspath(filename)
        
        print(f"📁 File location: {full_path}")
        print(f"📊 File size: {file_size:.1f} MB")
        print(f"🎯 Points plotted: 30,000 quaternions + 24 zone centers")
        
        print("\n🎮 INTERACTION GUIDE:")
        print("   • Drag with mouse to rotate the 3D view")
        print("   • Scroll to zoom in/out")
        print("   • Click legend items to show/hide zones") 
        print("   • Double-click legend to show only that zone")
        print("   • Hover over points for details")
        print("   • Use the toolbar (top-right) for more options")
        
        print(f"\n🌐 To view: Open '{filename}' in your web browser")
        
        return filename
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    print("=== ROTATABLE QUATERNION ZONE VISUALIZATION ===")
    filename = save_and_open_plot()
    
    if filename:
        print(f"\n✨ SUCCESS! Your interactive plot is ready!")
        print(f"   File: {filename}")
        print("   Open this file in your web browser to explore the zones!")

if __name__ == "__main__":
    main()
