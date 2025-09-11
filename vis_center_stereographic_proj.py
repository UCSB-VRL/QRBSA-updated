# Create stereographic projection visualization
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import plotly.express as px

def stereographic_projection_3d(quaternions):
    """
    Project quaternions from 4D hypersphere to 3D space using stereographic projection.
    Projects from north pole (0,0,0,1) to the hyperplane w=0.
    
    For quaternion q = (w,x,y,z), the projection is:
    (X,Y,Z) = (x,y,z) / (1-w)  if w != 1
    (X,Y,Z) = (0,0,0) if w = 1 (north pole maps to origin)
    """
    projected = []
    for q in quaternions:
        w, x, y, z = q
        if abs(w - 1.0) < 1e-10:  # Near north pole (identity quaternion)
            # Map to origin
            projected.append([0.0, 0.0, 0.0])
        else:
            # Standard stereographic projection
            denom = 1.0 - w
            X = x / denom
            Y = y / denom 
            Z = z / denom
            projected.append([X, Y, Z])
    
    return np.array(projected)

def inverse_stereographic_projection_3d(points):
    """
    Map 3D points back to 4D hypersphere using inverse stereographic projection.
    """
    quaternions = []
    for p in points:
        X, Y, Z = p
        r_squared = X**2 + Y**2 + Z**2
        
        if r_squared == 0:
            # Origin maps to north pole
            quaternions.append([1.0, 0.0, 0.0, 0.0])
        else:
            # Inverse stereographic formula
            denom = 1.0 + r_squared
            w = (r_squared - 1.0) / denom
            x = 2.0 * X / denom
            y = 2.0 * Y / denom
            z = 2.0 * Z / denom
            quaternions.append([w, x, y, z])
    
    return np.array(quaternions)

# Load the zone centers
zone_centers = pd.read_csv('cubic_group_quaternions_48.csv')
print(f"Loaded {len(zone_centers)} zone centers")

# Extract quaternion components
quaternions = zone_centers[['w', 'x', 'y', 'z']].values

# Apply stereographic projection
projected_centers = stereographic_projection_3d(quaternions)

print("\nStereographic Projection Results:")
print("Original vs Projected coordinates:")
print("Zone | Original (w,x,y,z) | Projected (X,Y,Z) | Distance from origin")
print("-" * 80)

for i in range(min(10, len(quaternions))):
    orig = quaternions[i]
    proj = projected_centers[i]
    dist = np.linalg.norm(proj)
    print(f"{i:4d} | ({orig[0]:6.3f},{orig[1]:6.3f},{orig[2]:6.3f},{orig[3]:6.3f}) | "
          f"({proj[0]:8.3f},{proj[1]:8.3f},{proj[2]:8.3f}) | {dist:8.3f}")

# Create visualization
fig = plt.figure(figsize=(20, 15))

# 1. Original quaternion space (x,y,z components only)
ax1 = fig.add_subplot(231, projection='3d')
colors = plt.cm.tab20(np.arange(len(quaternions)) % 20)
ax1.scatter(quaternions[:, 1], quaternions[:, 2], quaternions[:, 3], 
           c=colors, s=100, alpha=0.7)
ax1.set_xlabel('X')
ax1.set_ylabel('Y') 
ax1.set_zlabel('Z')
ax1.set_title('Original Quaternions\n(x,y,z components)')

# 2. Stereographic projection
ax2 = fig.add_subplot(232, projection='3d')
ax2.scatter(projected_centers[:, 0], projected_centers[:, 1], projected_centers[:, 2],
           c=colors, s=100, alpha=0.7)
ax2.set_xlabel('X (projected)')
ax2.set_ylabel('Y (projected)')
ax2.set_zlabel('Z (projected)')
ax2.set_title('Stereographic Projection\nof Zone Centers')

# 3. Distance comparison
ax3 = fig.add_subplot(233)
original_distances = np.linalg.norm(quaternions[:, 1:4], axis=1)
projected_distances = np.linalg.norm(projected_centers, axis=1)

ax3.scatter(original_distances, projected_distances, c=colors, s=50, alpha=0.7)
ax3.set_xlabel('Original distance from origin')
ax3.set_ylabel('Projected distance from origin')
ax3.set_title('Distance Comparison')
ax3.plot([0, 1], [0, np.inf], 'r--', alpha=0.5, label='y = x/(1-x²)')

# 4. W component vs projected distance
ax4 = fig.add_subplot(234)
w_components = quaternions[:, 0]
ax4.scatter(w_components, projected_distances, c=colors, s=50, alpha=0.7)
ax4.set_xlabel('W component')
ax4.set_ylabel('Projected distance from origin')
ax4.set_title('W vs Projected Distance')

# 5. Rotation angles vs distances
ax5 = fig.add_subplot(235)
angles = zone_centers['angle_deg'].values
ax5.scatter(angles, projected_distances, c=colors, s=50, alpha=0.7)
ax5.set_xlabel('Rotation Angle (degrees)')
ax5.set_ylabel('Projected Distance')
ax5.set_title('Rotation Angle vs Projected Distance')

# 6. Special points analysis
ax6 = fig.add_subplot(236)
# Categorize points by rotation type
identity_mask = angles == 0
face_mask = (angles == 90) | (angles == 270)
edge_mask = (angles == 120) | (angles == 240) 
vertex_mask = angles == 180

ax6.scatter(projected_distances[identity_mask], [0]*sum(identity_mask), 
           c='red', s=100, label='Identity (0°)', alpha=0.8)
ax6.scatter(projected_distances[face_mask], [1]*sum(face_mask), 
           c='blue', s=100, label='Face (90°/270°)', alpha=0.8)
ax6.scatter(projected_distances[edge_mask], [2]*sum(edge_mask), 
           c='green', s=100, label='Edge (120°/240°)', alpha=0.8)
ax6.scatter(projected_distances[vertex_mask], [3]*sum(vertex_mask), 
           c='orange', s=100, label='Vertex (180°)', alpha=0.8)

ax6.set_xlabel('Projected Distance from Origin')
ax6.set_ylabel('Rotation Type')
ax6.set_title('Stereographic Distances by Rotation Type')
ax6.set_yticks([0, 1, 2, 3])
ax6.set_yticklabels(['Identity', 'Face', 'Edge', 'Vertex'])
ax6.legend()
ax6.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('stereographic_projection_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# Create interactive stereographic projection
fig_interactive = go.Figure()

# Add zone centers with different symbols for different rotation types
rotation_types = {
    0: ('Identity', 'red', 'circle'),
    90: ('Face 90°', 'blue', 'square'),
    270: ('Face 270°', 'lightblue', 'square-open'),
    120: ('Edge 120°', 'green', 'diamond'),
    240: ('Edge 240°', 'lightgreen', 'diamond-open'),
    180: ('Vertex 180°', 'orange', 'x')
}

for angle in rotation_types.keys():
    mask = zone_centers['angle_deg'] == angle
    if sum(mask) > 0:
        label, color, symbol = rotation_types[angle]
        indices = np.where(mask)[0]
        
        fig_interactive.add_trace(go.Scatter3d(
            x=projected_centers[mask, 0],
            y=projected_centers[mask, 1], 
            z=projected_centers[mask, 2],
            mode='markers',
            marker=dict(
                size=12,
                color=color,
                symbol=symbol,
                line=dict(width=2, color='black')
            ),
            name=label,
            text=[f'Zone {i}<br>Angle: {angle}°<br>Projected: ({projected_centers[i,0]:.3f}, {projected_centers[i,1]:.3f}, {projected_centers[i,2]:.3f})'
                  for i in indices],
            hovertemplate='<b>%{text}</b><extra></extra>'
        ))

fig_interactive.update_layout(
    title='Interactive Stereographic Projection of Quaternion Zone Centers',
    scene=dict(
        xaxis_title='X (Stereographic)',
        yaxis_title='Y (Stereographic)', 
        zaxis_title='Z (Stereographic)',
        camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
    ),
    width=1000,
    height=800
)

fig_interactive.write_html('stereographic_projection_interactive.html')
print("\nSaved interactive plot: stereographic_projection_interactive.html")

# Summary statistics
print(f"\n" + "="*60)
print("STEREOGRAPHIC PROJECTION ANALYSIS SUMMARY")
print("="*60)
print(f"Total zone centers: {len(quaternions)}")
print(f"\nProjected distances:")
print(f"  Minimum: {projected_distances.min():.6f}")
print(f"  Maximum: {projected_distances.max():.6f}")
print(f"  Mean: {projected_distances.mean():.6f}")
print(f"  Std: {projected_distances.std():.6f}")

print(f"\nBy rotation type:")
for angle in [0, 90, 120, 180, 270, 240]:
    mask = zone_centers['angle_deg'] == angle
    if sum(mask) > 0:
        distances = projected_distances[mask]
        print(f"  {angle:3d}°: {len(distances):2d} zones, distances = {distances.min():.3f} to {distances.max():.3f}")

print(f"\nSpecial cases:")
identity_indices = np.where(zone_centers['angle_deg'] == 0)[0]
vertex_indices = np.where(zone_centers['angle_deg'] == 180)[0]
print(f"  Identity (0°): Projects to origin (distance ≈ 0)")
print(f"  Vertex (180°): Projects to infinity (large distances)")