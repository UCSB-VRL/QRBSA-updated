#!/usr/bin/env python3
"""
Create interactive 3D visualizations of quaternion zones that can be rotated and explored.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo

def load_data():
    """Load the classification data."""
    print("Loading classification data...")
    classified_df = pd.read_csv("fibonacci_zone_classification.csv")
    distribution_df = pd.read_csv("fibonacci_zone_classification_distribution.csv")
    cubic_df = pd.read_csv("cubic_group_quaternions_48.csv")
    return classified_df, distribution_df, cubic_df

def create_zone_color_mapping(distribution_df):
    """Create consistent colors for zones."""
    active_zones = sorted(distribution_df['zone_id'].astype(int).values)
    
    # Use plotly color scales for better distinction
    angle_groups = distribution_df.groupby('angle_deg')['zone_id'].apply(list).to_dict()
    
    # Color schemes by rotation angle
    color_schemes = {
        0.0: px.colors.qualitative.Set1[0:1],  # Red for identity
        90.0: px.colors.sequential.Blues[2:8],  # Blues for 90° (6 zones)
        120.0: px.colors.sequential.Greens[2:10],  # Greens for 120° (8 zones) 
        180.0: px.colors.sequential.Oranges[2:11]  # Oranges for 180° (9 zones)
    }
    
    zone_colors = {}
    zone_info = {}
    
    for angle, zones in angle_groups.items():
        colors = color_schemes[angle]
        for i, zone in enumerate(sorted(zones)):
            zone_colors[int(zone)] = colors[i % len(colors)]
            zone_info[int(zone)] = {
                'angle': angle,
                'color': colors[i % len(colors)],
                'type': get_rotation_type(angle)
            }
    
    return zone_colors, zone_info

def get_rotation_type(angle):
    """Get rotation type description."""
    if angle == 0.0:
        return "Identity"
    elif angle == 90.0:
        return "Face rotation"
    elif angle == 120.0:
        return "Edge rotation"
    elif angle == 180.0:
        return "Vertex rotation"
    else:
        return f"{angle}° rotation"

def create_interactive_3d_zones(sample_size=20000):
    """Create interactive 3D visualization of quaternion zones."""
    print(f"Creating interactive 3D visualization with {sample_size} points...")
    
    # Load data
    classified_df, distribution_df, cubic_df = load_data()
    zone_colors, zone_info = create_zone_color_mapping(distribution_df)
    
    # Sample data for performance
    if sample_size < len(classified_df):
        sample_df = classified_df.sample(n=sample_size, random_state=42)
    else:
        sample_df = classified_df.copy()
    
    # Create the main 3D scatter plot
    fig = go.Figure()
    
    # Add points for each zone
    for zone_id, color in zone_colors.items():
        zone_data = sample_df[sample_df['zone_id'] == zone_id]
        if len(zone_data) > 0:
            info = zone_info[zone_id]
            fig.add_trace(go.Scatter3d(
                x=zone_data['x'],
                y=zone_data['y'], 
                z=zone_data['z'],
                mode='markers',
                marker=dict(
                    size=2,
                    color=color,
                    opacity=0.7
                ),
                name=f'Zone {zone_id} ({info["type"]})',
                text=[f'Zone: {zone_id}<br>Type: {info["type"]}<br>Angle: {info["angle"]:.1f}°<br>Distance: {d:.4f}' 
                      for d in zone_data['min_distance']],
                hovertemplate='<b>%{text}</b><br>X: %{x:.4f}<br>Y: %{y:.4f}<br>Z: %{z:.4f}<extra></extra>'
            ))
    
    # Add zone centers as larger markers
    active_zones = sorted(zone_colors.keys())
    centers = cubic_df.iloc[active_zones]
    
    fig.add_trace(go.Scatter3d(
        x=centers['x'],
        y=centers['y'],
        z=centers['z'],
        mode='markers',
        marker=dict(
            size=8,
            color='black',
            symbol='diamond',
            line=dict(width=2, color='white'),
            opacity=1.0
        ),
        name='Zone Centers',
        text=[f'Zone Center {zone_id}<br>Type: {zone_info[zone_id]["type"]}<br>Angle: {zone_info[zone_id]["angle"]:.1f}°'
              for zone_id in active_zones],
        hovertemplate='<b>%{text}</b><br>X: %{x:.4f}<br>Y: %{y:.4f}<br>Z: %{z:.4f}<extra></extra>'
    ))
    
    # Update layout
    fig.update_layout(
        title=dict(
            text='Interactive Quaternion Zone Visualization<br><sub>Rotate, zoom, and hover to explore the 24 zones</sub>',
            x=0.5,
            font=dict(size=16)
        ),
        scene=dict(
            xaxis_title='Quaternion X',
            yaxis_title='Quaternion Y',
            zaxis_title='Quaternion Z',
            camera=dict(
                eye=dict(x=1.2, y=1.2, z=1.2)
            ),
            aspectmode='cube'
        ),
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left", 
            x=1.02,
            itemsizing='constant',
            font=dict(size=10)
        ),
        width=1200,
        height=800,
        margin=dict(r=200, l=0, t=100, b=0)
    )
    
    return fig

def create_interactive_projections(sample_size=15000):
    """Create interactive 2D projections."""
    print(f"Creating interactive 2D projections with {sample_size} points...")
    
    # Load data
    classified_df, distribution_df, cubic_df = load_data()
    zone_colors, zone_info = create_zone_color_mapping(distribution_df)
    
    # Sample data
    sample_df = classified_df.sample(n=sample_size, random_state=42)
    
    # Create subplot figure
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('XY Projection', 'XZ Projection', 'YZ Projection', 'Zone Statistics'),
        specs=[[{'type': 'scatter'}, {'type': 'scatter'}],
               [{'type': 'scatter'}, {'type': 'bar'}]]
    )
    
    # XY Projection
    for zone_id, color in zone_colors.items():
        zone_data = sample_df[sample_df['zone_id'] == zone_id]
        if len(zone_data) > 0:
            info = zone_info[zone_id]
            fig.add_trace(
                go.Scatter(
                    x=zone_data['x'],
                    y=zone_data['y'],
                    mode='markers',
                    marker=dict(size=3, color=color, opacity=0.7),
                    name=f'Zone {zone_id}',
                    text=[f'Zone: {zone_id}<br>Type: {info["type"]}<br>Angle: {info["angle"]:.1f}°' for _ in range(len(zone_data))],
                    hovertemplate='<b>%{text}</b><br>X: %{x:.4f}<br>Y: %{y:.4f}<extra></extra>',
                    showlegend=False
                ),
                row=1, col=1
            )
    
    # XZ Projection  
    for zone_id, color in zone_colors.items():
        zone_data = sample_df[sample_df['zone_id'] == zone_id]
        if len(zone_data) > 0:
            fig.add_trace(
                go.Scatter(
                    x=zone_data['x'],
                    y=zone_data['z'], 
                    mode='markers',
                    marker=dict(size=3, color=color, opacity=0.7),
                    name=f'Zone {zone_id}',
                    showlegend=False,
                    hovertemplate='Zone: %{customdata}<br>X: %{x:.4f}<br>Z: %{y:.4f}<extra></extra>',
                    customdata=zone_data['zone_id']
                ),
                row=1, col=2
            )
    
    # YZ Projection
    for zone_id, color in zone_colors.items():
        zone_data = sample_df[sample_df['zone_id'] == zone_id]
        if len(zone_data) > 0:
            fig.add_trace(
                go.Scatter(
                    x=zone_data['y'],
                    y=zone_data['z'],
                    mode='markers', 
                    marker=dict(size=3, color=color, opacity=0.7),
                    name=f'Zone {zone_id}',
                    showlegend=False,
                    hovertemplate='Zone: %{customdata}<br>Y: %{x:.4f}<br>Z: %{y:.4f}<extra></extra>',
                    customdata=zone_data['zone_id']
                ),
                row=2, col=1
            )
    
    # Zone statistics bar chart
    zone_counts = distribution_df.sort_values('zone_id')
    fig.add_trace(
        go.Bar(
            x=zone_counts['zone_id'],
            y=zone_counts['count'],
            marker=dict(
                color=[zone_colors[int(z)] for z in zone_counts['zone_id']],
                opacity=0.8
            ),
            name='Zone Population',
            text=[f'{int(c):,}' for c in zone_counts['count']],
            textposition='outside',
            hovertemplate='Zone %{x}: %{y:,} quaternions<extra></extra>',
            showlegend=False
        ),
        row=2, col=2
    )
    
    # Update layout
    fig.update_layout(
        title=dict(
            text='Interactive Quaternion Zone Projections',
            x=0.5,
            font=dict(size=16)
        ),
        height=800,
        width=1400
    )
    
    # Update axes labels
    fig.update_xaxes(title_text="X", row=1, col=1)
    fig.update_yaxes(title_text="Y", row=1, col=1)
    fig.update_xaxes(title_text="X", row=1, col=2)
    fig.update_yaxes(title_text="Z", row=1, col=2)
    fig.update_xaxes(title_text="Y", row=2, col=1)
    fig.update_yaxes(title_text="Z", row=2, col=1)
    fig.update_xaxes(title_text="Zone ID", row=2, col=2)
    fig.update_yaxes(title_text="Count", row=2, col=2)
    
    return fig

def create_zone_explorer():
    """Create an interactive zone explorer with detailed information."""
    print("Creating interactive zone explorer...")
    
    # Load data
    classified_df, distribution_df, cubic_df = load_data()
    zone_colors, zone_info = create_zone_color_mapping(distribution_df)
    
    # Create dropdown options
    zone_options = []
    for zone_id in sorted(zone_colors.keys()):
        info = zone_info[zone_id]
        zone_count = distribution_df[distribution_df['zone_id'] == zone_id]['count'].values[0]
        zone_options.append({
            'label': f'Zone {zone_id} - {info["type"]} ({zone_count:,} points)',
            'value': zone_id
        })
    
    # Create the figure with all zones initially
    fig = go.Figure()
    
    # Add all zones (initially visible)
    for zone_id, color in zone_colors.items():
        zone_data = classified_df[classified_df['zone_id'] == zone_id].sample(n=min(2000, len(classified_df[classified_df['zone_id'] == zone_id])), random_state=42)
        info = zone_info[zone_id]
        
        fig.add_trace(go.Scatter3d(
            x=zone_data['x'],
            y=zone_data['y'],
            z=zone_data['z'],
            mode='markers',
            marker=dict(
                size=3,
                color=color,
                opacity=0.8
            ),
            name=f'Zone {zone_id} ({info["type"]})',
            text=[f'Zone: {zone_id}<br>Type: {info["type"]}<br>Angle: {info["angle"]:.1f}°<br>Distance: {d:.4f}' 
                  for d in zone_data['min_distance']],
            hovertemplate='<b>%{text}</b><br>X: %{x:.4f}<br>Y: %{y:.4f}<br>Z: %{z:.4f}<extra></extra>',
            visible=True
        ))
    
    # Update layout with interactive controls
    fig.update_layout(
        title=dict(
            text='Quaternion Zone Explorer<br><sub>Use the legend to show/hide zones • Rotate and zoom to explore</sub>',
            x=0.5,
            font=dict(size=16)
        ),
        scene=dict(
            xaxis_title='Quaternion X',
            yaxis_title='Quaternion Y',
            zaxis_title='Quaternion Z',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            ),
            aspectmode='cube'
        ),
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02,
            itemsizing='constant',
            font=dict(size=10),
            itemclick="toggle",  # Allow clicking to show/hide traces
            itemdoubleclick="toggleothers"  # Double-click to show only that trace
        ),
        width=1400,
        height=900,
        margin=dict(r=300, l=0, t=100, b=0)
    )
    
    return fig

def save_interactive_plots():
    """Save all interactive plots as HTML files."""
    print("Creating and saving interactive plots...")
    
    try:
        # Create the main 3D interactive plot
        print("1. Creating main 3D visualization...")
        fig_3d = create_interactive_3d_zones(sample_size=25000)
        fig_3d.write_html("interactive_quaternion_zones_3d.html")
        print("   ✓ Saved: interactive_quaternion_zones_3d.html")
        
        # Create 2D projections
        print("2. Creating 2D projections...")
        fig_2d = create_interactive_projections(sample_size=20000)
        fig_2d.write_html("interactive_quaternion_projections.html") 
        print("   ✓ Saved: interactive_quaternion_projections.html")
        
        # Create zone explorer
        print("3. Creating zone explorer...")
        fig_explorer = create_zone_explorer()
        fig_explorer.write_html("interactive_zone_explorer.html")
        print("   ✓ Saved: interactive_zone_explorer.html")
        
        print("\n🎉 SUCCESS! Created 3 interactive HTML files:")
        print("   • interactive_quaternion_zones_3d.html - Main 3D rotatable plot")
        print("   • interactive_quaternion_projections.html - 2D projections + stats")
        print("   • interactive_zone_explorer.html - Zone explorer with show/hide")
        print("\nOpen these files in your web browser to interact with them!")
        
    except Exception as e:
        print(f"❌ Error creating interactive plots: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main function."""
    print("=== INTERACTIVE QUATERNION ZONE VISUALIZATION ===")
    save_interactive_plots()

if __name__ == "__main__":
    main()
