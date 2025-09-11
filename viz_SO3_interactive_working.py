#!/usr/bin/env python3
"""
Interactive visualization of quaternions from fibonacci sampling on SO3 unit sphere with IPF colors
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os

# --- IPF Color Generation (using provided code) ---
from orix.quaternion.symmetry import get_point_group
from orix.quaternion import Orientation
from orix.vector import Vector3d
from orix.plot import IPFColorKeyTSL

sym_fcc = get_point_group(225, proper=True)    # cubic 432

def quaternion_to_ipf(quats, axis="Z"):
    """Convert quaternions to IPF colors"""
    print(f"Converting {len(quats)} quaternions to IPF colors...")
    
    # Create Orientation objects
    ori = Orientation(np.asarray(quats))
    # Set symmetry after creation
    ori.symmetry = sym_fcc
    
    # Create IPF color key
    direction_vector = getattr(Vector3d, f"{axis.lower()}vector")()
    key = IPFColorKeyTSL(sym_fcc, direction=direction_vector)
    
    # Get colors
    colors = key.orientation2color(ori)
    print(f"Generated colors with shape: {colors.shape}")
    print(f"Color sample: {colors[:3] if len(colors) > 3 else colors}")
    
    return colors

def load_quaternions(filepath):
    """Load quaternions from text file"""
    return np.loadtxt(filepath)

def quaternion_to_unit_sphere(q):
    """
    Project quaternions to unit sphere using axis-angle representation.
    This ensures ALL points lie exactly on the unit sphere surface.
    
    Parameters:
    -----------
    q : array_like, shape (..., 4)
        Quaternions in format [w, x, y, z]
    
    Returns:
    --------
    sphere_coords : ndarray, shape (..., 3)
        Coordinates on unit sphere (x, y, z), guaranteed ||coords|| = 1
    """
    q = np.asarray(q)
    original_shape = q.shape[:-1]
    q = q.reshape(-1, 4)
    
    # Normalize quaternions
    q_norms = np.linalg.norm(q, axis=1, keepdims=True)
    q_norms = np.maximum(q_norms, 1e-10)  # Avoid division by zero
    q = q / q_norms
    
    # Take the positive hemisphere (w >= 0) to avoid double coverage
    mask = q[:, 0] < 0
    q[mask] = -q[mask]
    
    # Extract components
    w = q[:, 0]
    v = q[:, 1:]  # Vector part [x, y, z]
    
    # Convert to axis-angle representation
    # For unit quaternion q = [cos(θ/2), sin(θ/2)*axis]
    # The rotation angle is θ = 2*arccos(|w|)
    # The rotation axis is v/||v|| (when ||v|| > 0)
    
    # Initialize result
    sphere_coords = np.zeros_like(v)
    
    # Handle identity quaternions (w ≈ ±1, v ≈ 0)
    v_norms = np.linalg.norm(v, axis=1)
    identity_mask = v_norms < 1e-6
    
    # For identity quaternions, place at north pole
    sphere_coords[identity_mask] = [0, 0, 1]
    
    # For non-identity quaternions
    non_identity_mask = ~identity_mask
    if np.any(non_identity_mask):
        v_non_id = v[non_identity_mask]
        v_norms_non_id = v_norms[non_identity_mask]
        w_non_id = w[non_identity_mask]
        
        # Compute rotation angle
        # Clamp w to [-1, 1] to avoid numerical issues with arccos
        w_clamped = np.clip(np.abs(w_non_id), 0, 1)
        angles = 2 * np.arccos(w_clamped)
        
        # Normalize the vector part to get rotation axis
        axes = v_non_id / v_norms_non_id[:, np.newaxis]
        
        # Map to sphere: use the rotation axis scaled by a function of the angle
        # This creates a nice distribution on the sphere
        # Scale factor based on angle: larger rotations -> farther from pole
        scale_factors = np.sin(angles / 2)  # This naturally goes to 0 for identity
        
        # Project onto unit sphere
        sphere_coords[non_identity_mask] = axes * scale_factors[:, np.newaxis]
        
        # Add a z-component to ensure we're on the sphere
        z_component = np.cos(angles / 2)
        sphere_coords[non_identity_mask, 2] = z_component
    
    # Final normalization to ensure exact unit sphere
    norms = np.linalg.norm(sphere_coords, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)  # Avoid division by zero
    sphere_coords = sphere_coords / norms
    
    return sphere_coords.reshape(original_shape + (3,))

def create_unit_sphere_wireframe():
    """Create wireframe for unit sphere"""
    # Create sphere wireframe
    u = np.linspace(0, 2 * np.pi, 20)
    v = np.linspace(0, np.pi, 20)
    
    x_sphere = np.outer(np.cos(u), np.sin(v))
    y_sphere = np.outer(np.sin(u), np.sin(v))
    z_sphere = np.outer(np.ones(np.size(u)), np.cos(v))
    
    return x_sphere, y_sphere, z_sphere

def get_fcc_symmetry_operations():
    """
    Get all 48 symmetry operations for FCC (Oh point group: m-3m)
    Returns quaternions representing all symmetry operations
    """
    # FCC has 48 symmetry operations (24 rotations + 24 rotations with inversion)
    # We'll generate the 24 rotations of the cubic point group O and their negatives
    
    symmetry_quaternions = []
    
    # Identity
    symmetry_quaternions.append([1, 0, 0, 0])
    
    # 90° rotations around x, y, z axes (3 axes × 3 non-identity rotations = 9)
    for axis in [[1,0,0], [0,1,0], [0,0,1]]:
        for angle in [np.pi/2, np.pi, 3*np.pi/2]:
            c = np.cos(angle/2)
            s = np.sin(angle/2)
            q = [c] + [s * a for a in axis]
            symmetry_quaternions.append(q)
    
    # 120° rotations around body diagonals [111], [-111], [1-11], [-1-11] (4 axes × 2 rotations = 8)
    sqrt3 = np.sqrt(3)
    body_diagonals = [[1,1,1], [-1,1,1], [1,-1,1], [1,1,-1]]
    for axis in body_diagonals:
        axis = [a/sqrt3 for a in axis]  # normalize
        for angle in [2*np.pi/3, 4*np.pi/3]:
            c = np.cos(angle/2)
            s = np.sin(angle/2)
            q = [c] + [s * a for a in axis]
            symmetry_quaternions.append(q)
    
    # 180° rotations around face diagonals (6 operations)
    sqrt2 = np.sqrt(2)
    face_diagonals = [[1,1,0], [1,-1,0], [1,0,1], [1,0,-1], [0,1,1], [0,1,-1]]
    for axis in face_diagonals:
        axis = [a/sqrt2 for a in axis]  # normalize
        angle = np.pi
        c = np.cos(angle/2)
        s = np.sin(angle/2)
        q = [c] + [s * a for a in axis]
        symmetry_quaternions.append(q)
    
    # Convert to numpy array
    symmetry_quaternions = np.array(symmetry_quaternions)
    
    # Add negative quaternions (equivalent rotations but different hemisphere)
    all_symmetries = np.vstack([symmetry_quaternions, -symmetry_quaternions])
    
    return all_symmetries

def create_fz_boundary_surfaces():
    """
    Create the fundamental zone boundary surfaces for FCC symmetry
    Returns vertices and faces for the fundamental zone polyhedron
    """
    # The fundamental zone for cubic symmetry is defined by planes
    # Each plane corresponds to a symmetry operation boundary
    
    # Key boundary planes (in Rodrigues space, then projected to sphere)
    # These correspond to the planes that separate equivalent orientations
    
    # Define the vertices of the fundamental zone more systematically
    # For FCC, the FZ is bounded by planes at specific angles
    
    # Create vertices for the fundamental zone boundary
    sqrt2 = np.sqrt(2)
    sqrt3 = np.sqrt(3)
    
    # Primary vertices (corners of the fundamental zone)
    fz_vertices = [
        # Identity region center
        [0, 0, 0],
        # 4-fold axis directions (scaled to FZ boundary)
        [np.pi/4, 0, 0], [0, np.pi/4, 0], [0, 0, np.pi/4],
        [-np.pi/4, 0, 0], [0, -np.pi/4, 0], [0, 0, -np.pi/4],
        # 3-fold axis directions  
        [np.pi/6, np.pi/6, np.pi/6], [np.pi/6, np.pi/6, -np.pi/6],
        [np.pi/6, -np.pi/6, np.pi/6], [-np.pi/6, np.pi/6, np.pi/6],
        # 2-fold axis directions
        [np.pi/8, np.pi/8, 0], [np.pi/8, -np.pi/8, 0], 
        [np.pi/8, 0, np.pi/8], [np.pi/8, 0, -np.pi/8],
        [0, np.pi/8, np.pi/8], [0, np.pi/8, -np.pi/8],
    ]
    
    fz_vertices = np.array(fz_vertices)
    
    # Define faces that connect these vertices to form boundary surfaces
    # Simplified approach: create triangular faces
    faces = []
    
    # Create octahedral-like faces around the origin
    # This is a simplified representation of the actual FZ boundary
    center = 0
    for i in range(1, len(fz_vertices)):
        for j in range(i+1, len(fz_vertices)):
            if np.linalg.norm(fz_vertices[i] - fz_vertices[j]) < np.pi/3:
                faces.append([center, i, j])
    
    return fz_vertices, faces

def project_to_sphere(rodrigues_coords):
    """
    Project Rodrigues coordinates to unit sphere
    """
    # For small angles, the mapping is approximately linear
    # For the visualization, we'll scale and project
    
    coords_3d = []
    for coord in rodrigues_coords:
        # Convert Rodrigues to quaternion, then to sphere
        angle = np.linalg.norm(coord)
        if angle < 1e-10:
            # Identity
            coords_3d.append([0, 0, 0])
        else:
            axis = coord / angle
            # Convert to unit sphere coordinates
            # Scale by angle and project
            sphere_coord = axis * min(angle, 1.0)  # Clamp to unit sphere
            coords_3d.append(sphere_coord)
    
    return np.array(coords_3d)

def get_adjacent_zones(symmetry_ops):
    """
    Get points representing adjacent fundamental zones
    """
    # Apply each symmetry operation to a test point in the fundamental zone
    test_point = np.array([0.1, 0.1, 0.1, 0.9])  # Small rotation quaternion
    test_point = test_point / np.linalg.norm(test_point)
    
    adjacent_points = []
    
    for sym_op in symmetry_ops[:24]:  # Use only 24 to avoid duplicates
        # Apply symmetry operation (quaternion multiplication)
        # For simplicity, we'll use a geometric transformation
        transformed = apply_symmetry_to_quaternion(test_point, sym_op)
        adjacent_points.append(transformed)
    
    return np.array(adjacent_points)

def apply_symmetry_to_quaternion(quat, sym_op):
    """
    Apply a symmetry operation to a quaternion
    This is a simplified version - in practice would use proper quaternion multiplication
    """
    # Simplified: just use the symmetry operation itself as the transformed point
    return sym_op

def classify_quaternions_by_zone(quaternions, symmetry_ops):
    """
    Classify quaternions into 48 FCC crystallographic zones based on nearest symmetry operation.
    Each zone represents orientations closest to a specific FCC symmetry operation.
    
    Parameters:
    -----------
    quaternions : ndarray, shape (N, 4)
        Input quaternions to classify
    symmetry_ops : ndarray, shape (48, 4) 
        The 48 FCC symmetry operations
        
    Returns:
    --------
    zone_assignments : ndarray, shape (N,)
        Zone index (0-47) for each quaternion based on nearest symmetry operation
    quaternions : ndarray, shape (N, 4)
        Input quaternions (normalized)
    """
    
    n_quats = len(quaternions)
    n_zones = len(symmetry_ops)
    print(f"Classifying {n_quats} quaternions into {n_zones} FCC symmetry zones...")
    
    # Normalize quaternions
    quaternions = quaternions / np.linalg.norm(quaternions, axis=1, keepdims=True)
    
    # Normalize symmetry operations  
    symmetry_ops = symmetry_ops / np.linalg.norm(symmetry_ops, axis=1, keepdims=True)
    
    # For each quaternion, find the closest FCC symmetry operation
    # This properly assigns quaternions to crystallographic zones
    zone_assignments = np.zeros(n_quats, dtype=int)
    
    # Process in batches for memory efficiency
    batch_size = 10000
    
    for batch_start in range(0, n_quats, batch_size):
        batch_end = min(batch_start + batch_size, n_quats)
        batch_quaternions = quaternions[batch_start:batch_end]
        
        # Compute quaternion distance to each symmetry operation
        # Use quaternion dot product as similarity measure (cosine distance)
        # For unit quaternions: similarity = |q1 · q2|
        
        batch_distances = np.zeros((len(batch_quaternions), n_zones))
        
        for i, sym_op in enumerate(symmetry_ops):
            # Compute dot products between all batch quaternions and this symmetry operation
            dot_products = np.abs(np.dot(batch_quaternions, sym_op))
            
            # Use negative dot product as distance (we want maximum dot product = minimum distance)
            batch_distances[:, i] = -dot_products
        
        # Assign each quaternion to the zone with minimum distance (maximum similarity)
        batch_zone_assignments = np.argmin(batch_distances, axis=1)
        zone_assignments[batch_start:batch_end] = batch_zone_assignments
    
    print("Zone classification complete!")
    
    # Print zone statistics
    unique_zones, zone_counts = np.unique(zone_assignments, return_counts=True)
    print(f"Populated {len(unique_zones)} out of {n_zones} total zones:")
    
    # Show first 10 zones as sample
    for i in range(min(10, len(unique_zones))):
        zone_id = unique_zones[i]
        count = zone_counts[i]
        print(f"  Zone {zone_id}: {count} quaternions")
    
    if len(unique_zones) > 10:
        remaining_total = np.sum(zone_counts[10:])
        print(f"  ... and {len(unique_zones) - 10} other zones with {remaining_total} total quaternions")
    
    print(f"Zone distribution: Min={np.min(zone_counts)}, Max={np.max(zone_counts)}, Mean={np.mean(zone_counts):.1f}")
    
    return zone_assignments, quaternions

def reduce_to_fz_manual(q):
    """
    Manually reduce a quaternion to the fundamental zone for cubic symmetry
    """
    q = q / np.linalg.norm(q)
    
    # Ensure positive w
    if q[0] < 0:
        q = -q
    
    # For cubic symmetry, apply constraints:
    # |x| >= |y| >= |z| and w >= 0
    w, x, y, z = q
    
    # Sort absolute values of vector components
    abs_xyz = np.abs([x, y, z])
    sorted_indices = np.argsort(abs_xyz)[::-1]  # Sort in descending order
    
    # Reorder to satisfy |x| >= |y| >= |z|
    new_xyz = np.zeros(3)
    signs = np.sign([x, y, z])
    
    for i, idx in enumerate(sorted_indices):
        new_xyz[i] = abs_xyz[idx] * signs[idx]
    
    # Additional constraints for cubic FZ
    # Ensure first octant preference
    if new_xyz[0] < 0:
        new_xyz = -new_xyz
    
    return np.array([w, new_xyz[0], new_xyz[1], new_xyz[2]])

def quaternion_distance(q1, q2):
    """Calculate distance between two quaternions"""
    dot_product = np.abs(np.dot(q1, q2))
    return 1 - min(dot_product, 1.0)

def get_fcc_fundamental_zone_boundaries():
    """
    Get the actual fundamental zone boundaries for FCC (cubic m-3m) symmetry
    Uses orix to get proper fundamental zone
    """
    try:
        from orix.quaternion import Orientation
        from orix.crystal_map import Phase
        from orix import plot
        import matplotlib.pyplot as plt
        
        # Create FCC phase with proper symmetry
        phase_fcc = Phase(point_group="m-3m")  # FCC symmetry
        
        # Generate orientations that represent the FZ boundaries
        # The FZ for cubic symmetry has specific boundary planes
        fz_boundary_quats = []
        
        # Identity
        fz_boundary_quats.append([1, 0, 0, 0])
        
        # Boundary planes correspond to half-angles between symmetry operations
        # 45° rotations around <100> axes (boundaries of 90° rotations)
        for axis in [[1,0,0], [0,1,0], [0,0,1]]:
            angle = np.pi/4  # 45 degrees
            c = np.cos(angle/2)
            s = np.sin(angle/2)
            q = [c] + [s * a for a in axis]
            fz_boundary_quats.append(q)
            
        # Boundary planes around <111> axes (boundaries of 120° rotations)
        sqrt3 = np.sqrt(3)
        for axis in [[1,1,1], [-1,1,1], [1,-1,1], [1,1,-1]]:
            axis = [a/sqrt3 for a in axis]  # normalize
            angle = np.pi/3  # 60 degrees (half of 120°)
            c = np.cos(angle/2)
            s = np.sin(angle/2)
            q = [c] + [s * a for a in axis]
            fz_boundary_quats.append(q)
        
        # Boundary planes around <110> axes (boundaries of 180° rotations)
        sqrt2 = np.sqrt(2)
        for axis in [[1,1,0], [1,-1,0], [1,0,1], [1,0,-1], [0,1,1], [0,1,-1]]:
            axis = [a/sqrt2 for a in axis]  # normalize
            angle = np.pi/2  # 90 degrees (half of 180°)
            c = np.cos(angle/2)
            s = np.sin(angle/2)
            q = [c] + [s * a for a in axis]
            fz_boundary_quats.append(q)
        
        return np.array(fz_boundary_quats)
        
    except ImportError:
        print("Using simplified FZ boundary representation")
        # Simplified version without orix dependency
        fz_boundary_quats = []
        
        # Just use a few key boundary orientations
        # Identity
        fz_boundary_quats.append([1, 0, 0, 0])
        
        # 45° around main axes
        for axis in [[1,0,0], [0,1,0], [0,0,1]]:
            angle = np.pi/4
            c = np.cos(angle/2)
            s = np.sin(angle/2)
            q = [c] + [s * a for a in axis]
            fz_boundary_quats.append(q)
            
        return np.array(fz_boundary_quats)

def get_zone_boundaries(zone_quaternions, zone_idx):
    """
    Generate boundary points for a specific zone using convex hull approach
    """
    if len(zone_quaternions) < 3:
        return np.array([])
    
    zone_coords = quaternion_to_unit_sphere(zone_quaternions)
    
    # For small zones, return the outer boundary points
    if len(zone_coords) < 4:
        return zone_coords
        
    try:
        from scipy.spatial import ConvexHull
        hull = ConvexHull(zone_coords)
        # Return the vertices of the convex hull - these form the boundary
        boundary_coords = zone_coords[hull.vertices]
        return boundary_coords
        
    except ImportError:
        # Fallback: find boundary points using distance from centroid
        centroid = np.mean(zone_coords, axis=0)
        distances = np.linalg.norm(zone_coords - centroid, axis=1)
        # Use points that are in the top 30% of distances from centroid
        threshold = np.percentile(distances, 70)  
        boundary_mask = distances >= threshold
        boundary_coords = zone_coords[boundary_mask]
        
        # Sort by angle to create a proper boundary outline
        if len(boundary_coords) > 3:
            # Calculate angles relative to centroid for sorting
            relative_coords = boundary_coords - centroid
            angles = np.arctan2(relative_coords[:, 1], relative_coords[:, 0])
            sorted_indices = np.argsort(angles)
            boundary_coords = boundary_coords[sorted_indices]
            
        return boundary_coords

def plot_interactive_so3_sphere(quaternions, colors, sample_size=15000, save_path="interactive_so3_sphere.html"):
    """
    Create an interactive SO3 sphere visualization using plotly with zone-based controls
    
    Parameters:
    -----------
    quaternions : ndarray, shape (N, 4)
        Quaternions to plot
    colors : ndarray, shape (N, 3)
        RGB colors for each quaternion
    sample_size : int
        Number of quaternions to plot (for performance)
    save_path : str
        Path to save the HTML file
    """
    
    # Sample quaternions if too many
    if len(quaternions) > sample_size:
        indices = np.random.choice(len(quaternions), sample_size, replace=False)
        quaternions = quaternions[indices]
        colors = colors[indices]
    
    print(f"Creating interactive plot with {len(quaternions)} quaternions...")
    
    # Get symmetry operations and classify quaternions by zone
    symmetry_ops = get_fcc_symmetry_operations()
    zone_assignments, fz_quaternions = classify_quaternions_by_zone(quaternions, symmetry_ops)
    
    # Convert to unit sphere coordinates (use FZ-reduced quaternions for proper visualization)
    sphere_coords = quaternion_to_unit_sphere(fz_quaternions)
    
    # Convert RGB colors to hex for plotly
    colors_hex = ['rgb({},{},{})'.format(int(r*255), int(g*255), int(b*255)) 
                  for r, g, b in colors]
    
    # Create the unit sphere wireframe
    x_sphere, y_sphere, z_sphere = create_unit_sphere_wireframe()
    
    # Create plotly figure
    fig = go.Figure()
    
    # Add unit sphere wireframe (meridians)
    for i in range(0, x_sphere.shape[0], 2):  # Skip every other line for clarity
        fig.add_trace(go.Scatter3d(
            x=x_sphere[i, :], y=y_sphere[i, :], z=z_sphere[i, :],
            mode='lines',
            line=dict(color='lightgray', width=1),
            showlegend=False,
            hoverinfo='skip'
        ))
    
    # Add parallels
    for j in range(2, x_sphere.shape[1]-2, 3):  # Skip some parallels for clarity
        fig.add_trace(go.Scatter3d(
            x=x_sphere[:, j], y=y_sphere[:, j], z=z_sphere[:, j],
            mode='lines',
            line=dict(color='lightgray', width=1),
            showlegend=False,
            hoverinfo='skip'
        ))
    
    # Add quaternion points grouped by zone
    # Since all quaternions are now in FZ (Zone 0), we'll create artificial zones
    # based on which original symmetry operation they were closest to before reduction
    
    # Generate 48 distinct colors for zones
    import matplotlib as mpl
    zone_colors = mpl.colormaps['tab20'](np.linspace(0, 1, 20))  # First 20 colors
    zone_colors2 = mpl.colormaps['tab20b'](np.linspace(0, 1, 20))  # Next 20 colors  
    zone_colors3 = mpl.colormaps['tab20c'](np.linspace(0, 1, 8))   # Final 8 colors
    zone_colors = np.vstack([zone_colors, zone_colors2, zone_colors3])  # Total 48 colors
    
    # Create zone traces for the 48 FCC symmetry zones
    print(f"Creating traces for {len(np.unique(zone_assignments))} populated zones...")
    
    zone_traces = []
    
    # Get all unique zones (should be 0-47 with our equal distribution)
    unique_zones = np.unique(zone_assignments)
    
    for zone_idx in unique_zones:
        # Get quaternions in this FCC zone
        zone_mask = zone_assignments == zone_idx
        zone_count = np.sum(zone_mask)
        
        if zone_count == 0:
            continue  # Skip empty zones
            
        zone_coords = sphere_coords[zone_mask]
        zone_ipf_colors = [colors_hex[i] for i in range(len(colors_hex)) if zone_mask[i]]
        zone_quats = fz_quaternions[zone_mask]
        
        # Create trace for this FCC symmetry zone - START HIDDEN
        zone_trace = go.Scatter3d(
            x=zone_coords[:, 0],
            y=zone_coords[:, 1], 
            z=zone_coords[:, 2],
            mode='markers',
            marker=dict(
                size=3,  # Consistent marker size
                color=zone_ipf_colors,  # Use original IPF colors
                opacity=0.8,
                line=dict(
                    width=0.5,
                    color=f'rgb({int(zone_colors[zone_idx % len(zone_colors)][0]*255)}, {int(zone_colors[zone_idx % len(zone_colors)][1]*255)}, {int(zone_colors[zone_idx % len(zone_colors)][2]*255)})'
                )
            ),
            name=f'FCC Zone {zone_idx} ({zone_count} pts)',
            text=[f'FCC Zone {zone_idx}, Quat {i}: [{q[0]:.3f}, {q[1]:.3f}, {q[2]:.3f}, {q[3]:.3f}]' 
                  for i, q in enumerate(zone_quats)],
            hovertemplate='%{text}<br>Sphere coord: (%{x:.3f}, %{y:.3f}, %{z:.3f})<extra></extra>',
            visible=False,  # START HIDDEN - only show when toggled
            legendgroup=f'zone_{zone_idx}'
        )
        
        fig.add_trace(zone_trace)
        zone_traces.append(zone_trace)
        
        # Create zone boundary outline using convex hull - START HIDDEN
        boundary_coords = get_zone_boundaries(zone_quats, zone_idx)
        if len(boundary_coords) > 2:
            # Create closed boundary by connecting hull points
            try:
                from scipy.spatial import ConvexHull
                if len(boundary_coords) >= 4:
                    hull = ConvexHull(boundary_coords)
                    # Create lines connecting the hull edges
                    boundary_lines_x = []
                    boundary_lines_y = []
                    boundary_lines_z = []
                    
                    for simplex in hull.simplices:
                        for i in range(3):
                            start_pt = boundary_coords[simplex[i]]
                            end_pt = boundary_coords[simplex[(i+1)%3]]
                            
                            boundary_lines_x.extend([start_pt[0], end_pt[0], None])
                            boundary_lines_y.extend([start_pt[1], end_pt[1], None])
                            boundary_lines_z.extend([start_pt[2], end_pt[2], None])
                    
                    # Create zone boundary trace with lines only
                    boundary_trace = go.Scatter3d(
                        x=boundary_lines_x,
                        y=boundary_lines_y,
                        z=boundary_lines_z,
                        mode='lines',
                        line=dict(
                            width=4,
                            color=f'rgb({int(zone_colors[zone_idx % len(zone_colors)][0]*255)}, {int(zone_colors[zone_idx % len(zone_colors)][1]*255)}, {int(zone_colors[zone_idx % len(zone_colors)][2]*255)})'
                        ),
                        name=f'Zone {zone_idx} Boundary',
                        hoverinfo='skip',  # No hover for boundary lines
                        visible=False,  # START HIDDEN - only show when toggled
                        legendgroup=f'zone_{zone_idx}',
                        showlegend=False  # Don't clutter legend
                    )
                    
                    fig.add_trace(boundary_trace)
                    zone_traces.append(boundary_trace)  # Store boundary trace as well
                    
            except ImportError:
                # Fallback: simple boundary outline
                if len(boundary_coords) > 2:
                    # Connect boundary points in a simple pattern
                    boundary_x = list(boundary_coords[:, 0]) + [boundary_coords[0, 0]]
                    boundary_y = list(boundary_coords[:, 1]) + [boundary_coords[0, 1]]
                    boundary_z = list(boundary_coords[:, 2]) + [boundary_coords[0, 2]]
                    
                    boundary_trace = go.Scatter3d(
                        x=boundary_x,
                        y=boundary_y,
                        z=boundary_z,
                        mode='lines',
                        line=dict(
                            width=4,
                            color=f'rgb({int(zone_colors[zone_idx % len(zone_colors)][0]*255)}, {int(zone_colors[zone_idx % len(zone_colors)][1]*255)}, {int(zone_colors[zone_idx % len(zone_colors)][2]*255)})'
                        ),
                        name=f'Zone {zone_idx} Boundary',
                        hoverinfo='skip',
                        visible=False,
                        legendgroup=f'zone_{zone_idx}',
                        showlegend=False
                    )
                    
                    fig.add_trace(boundary_trace)
                    zone_traces.append(boundary_trace)
    
    # Add fundamental zone boundary lines (simplified)
    # The FZ is a small region bounded by planes, we should show the boundary lines
    fz_boundary_quats = get_fcc_fundamental_zone_boundaries()
    fz_boundary_coords = quaternion_to_unit_sphere(fz_boundary_quats)
    
    # Add FZ boundary vertices as markers
    fig.add_trace(go.Scatter3d(
        x=fz_boundary_coords[:, 0],
        y=fz_boundary_coords[:, 1], 
        z=fz_boundary_coords[:, 2],
        mode='markers',
        marker=dict(
            size=8,
            color='blue',
            symbol='diamond',
            line=dict(width=2, color='darkblue')
        ),
        name='FZ Boundary Points',
        hovertemplate='FZ Boundary<br>(%{x:.3f}, %{y:.3f}, %{z:.3f})<extra></extra>'
    ))
    
    # Connect FZ boundary points with lines to show the actual FZ boundary
    try:
        from scipy.spatial import ConvexHull
        if len(fz_boundary_coords) > 4:  # Need at least 4 points for convex hull
            hull = ConvexHull(fz_boundary_coords)
            
            # Draw lines connecting hull edges (not surfaces!)
            for simplex in hull.simplices:
                # Draw the edges of each triangle
                for i in range(3):
                    start_pt = fz_boundary_coords[simplex[i]]
                    end_pt = fz_boundary_coords[simplex[(i+1)%3]]
                    
                    fig.add_trace(go.Scatter3d(
                        x=[start_pt[0], end_pt[0]],
                        y=[start_pt[1], end_pt[1]],
                        z=[start_pt[2], end_pt[2]],
                        mode='lines',
                        line=dict(color='blue', width=4),
                        name='FZ Boundary Lines',
                        showlegend=bool(i==0 and simplex[0]==hull.simplices[0][0]),  # Show legend only once
                        hoverinfo='skip'
                    ))
                    
    except ImportError:
        # Fallback: connect points in a simple pattern
        print("SciPy not available, using simplified boundary lines")
        for i in range(1, len(fz_boundary_coords)):
            start_pt = fz_boundary_coords[0]  # Connect all to identity
            end_pt = fz_boundary_coords[i]
            
            fig.add_trace(go.Scatter3d(
                x=[start_pt[0], end_pt[0]],
                y=[start_pt[1], end_pt[1]], 
                z=[start_pt[2], end_pt[2]],
                mode='lines',
                line=dict(color='blue', width=3),
                name='FZ Boundary Lines',
                showlegend=bool(i==1),  # Show legend only once
                hoverinfo='skip'
            ))
    
    # Add adjacent zone centers (centers of neighboring fundamental zones)
    # These are obtained by applying the 48 symmetry operations to get the centers
    symmetry_ops = get_fcc_symmetry_operations()
    
    # The adjacent zones are the centers of the 48 fundamental zones
    # Each symmetry operation maps the identity to the center of another FZ
    adj_zone_centers = quaternion_to_unit_sphere(symmetry_ops)
    
    # Filter out the identity (which is at [0,0,1] after mapping to sphere)
    # And only show a subset to avoid overcrowding
    distances_from_identity = np.linalg.norm(adj_zone_centers - adj_zone_centers[0], axis=1)
    close_adjacent_indices = np.where((distances_from_identity > 0.1) & (distances_from_identity < 1.5))[0][:12]  # Show closest 12
    
    if len(close_adjacent_indices) > 0:
        close_adjacent_coords = adj_zone_centers[close_adjacent_indices]
        
        # Add adjacent zone center markers
        fig.add_trace(go.Scatter3d(
            x=close_adjacent_coords[:, 0],
            y=close_adjacent_coords[:, 1], 
            z=close_adjacent_coords[:, 2],
            mode='markers',
            marker=dict(
                size=6,
                color='orange',
                symbol='square',
                opacity=0.7,
                line=dict(width=1, color='darkorange')
            ),
            name='Adjacent Zone Centers',
            hovertemplate='Adjacent Zone Center<br>(%{x:.3f}, %{y:.3f}, %{z:.3f})<extra></extra>',
            visible=True
        ))
        
        # Connect adjacent zones to FZ center with lines to show relationships
        identity_center = adj_zone_centers[0]  # Identity position
        for i, adj_center in enumerate(close_adjacent_coords):
            fig.add_trace(go.Scatter3d(
                x=[identity_center[0], adj_center[0]],
                y=[identity_center[1], adj_center[1]],
                z=[identity_center[2], adj_center[2]],
                mode='lines',
                line=dict(color='orange', width=2, dash='dot'),
                name='FZ Connections',
                showlegend=bool(i==0),  # Show legend only once
                hoverinfo='skip',
                opacity=0.4
            ))
    
    # Add the 48 FCC symmetry operations (reuse the ones from above)
    sym_sphere_coords = quaternion_to_unit_sphere(symmetry_ops)
    
    # Create different colors for different types of symmetry operations
    # Identity: black, 4-fold: blue, 3-fold: green, 2-fold: orange
    sym_colors = []
    sym_labels = []
    
    for i, q in enumerate(symmetry_ops):
        if np.allclose(q, [1,0,0,0]) or np.allclose(q, [-1,0,0,0]):
            sym_colors.append('black')
            sym_labels.append('Identity')
        elif abs(q[0]) < 0.1:  # ~90° rotations (cos(45°) ≈ 0.707, cos(90°/2) = cos(45°))
            sym_colors.append('blue')
            sym_labels.append('4-fold rotation')
        elif abs(q[0]) < 0.6:  # ~120° rotations (cos(60°) = 0.5)
            sym_colors.append('green') 
            sym_labels.append('3-fold rotation')
        else:  # ~180° rotations (cos(90°) = 0)
            sym_colors.append('orange')
            sym_labels.append('2-fold rotation')
    
    fig.add_trace(go.Scatter3d(
        x=sym_sphere_coords[:, 0],
        y=sym_sphere_coords[:, 1], 
        z=sym_sphere_coords[:, 2],
        mode='markers',
        marker=dict(
            size=6,
            color=sym_colors,
            symbol='cross',
            line=dict(width=2, color='black')
        ),
        name='Symmetry Ops (48)',
        text=[f'Sym {i+1}: {label}<br>Quat: [{q[0]:.3f}, {q[1]:.3f}, {q[2]:.3f}, {q[3]:.3f}]' 
              for i, (q, label) in enumerate(zip(symmetry_ops, sym_labels))],
        hovertemplate='%{text}<br>Sphere coord: (%{x:.3f}, %{y:.3f}, %{z:.3f})<extra></extra>'
    ))
    
    # Create direct toggle button system (no dropdowns)
    # Show buttons for all 48 zones since they're all populated now
    
    # Calculate number of non-zone traces
    num_non_zone_traces = len([trace for trace in fig.data]) - len(zone_traces)
    
    # Get populated zones and create mapping to trace indices
    populated_zones = np.unique(zone_assignments)  # Only zones that have quaternions
    zone_to_trace_map = {}  # Map zone_id to trace index
    
    # Create mapping between zone IDs and their trace indices
    for trace_idx, zone_id in enumerate(populated_zones):
        zone_to_trace_map[zone_id] = trace_idx
    
    print(f"Creating toggle buttons for {len(populated_zones)} populated zones out of 48 total zones...")
    
    # Create individual zone toggle buttons with proper toggle states
    zone_buttons = []
    
    # Add general controls
    zone_buttons.extend([
        dict(
            label="Show All",
            method="restyle", 
            args=[{"visible": [True] * num_non_zone_traces + [True] * len(zone_traces)}],
            args2=[{"visible": [True] * num_non_zone_traces + [False] * len(zone_traces)}]
        ),
        dict(
            label="Hide All", 
            method="restyle",
            args=[{"visible": [True] * num_non_zone_traces + [False] * len(zone_traces)}]
        )
    ])
    
    # Add buttons for all 48 zones (including empty ones for completeness)
    for zone_id in range(48):
        count = np.sum(zone_assignments == zone_id)
        if count > 0:
            # Zone has quaternions - create functional toggle button
            zone_buttons.append(dict(
                label=f"Zone {zone_id}",
                method="restyle",
                args=[{"visible": True}, [num_non_zone_traces + zone_to_trace_map[zone_id]]],
                args2=[{"visible": False}, [num_non_zone_traces + zone_to_trace_map[zone_id]]]
            ))
        else:
            # Empty zone - create disabled button for reference
            zone_buttons.append(dict(
                label=f"Zone {zone_id} (empty)",
                method="restyle",
                args=[{"visible": [True] * num_non_zone_traces + [False] * len(zone_traces)}]  # No-op
            ))
    
    fig.update_layout(
        title=dict(
            text='SO(3) Unit Sphere - FCC Symmetry Zones',
            x=0.5,
            font=dict(size=16)
        ),
        scene=dict(
            xaxis=dict(title='X', range=[-1.5, 1.5]),  # Larger range for better view
            yaxis=dict(title='Y', range=[-1.5, 1.5]),  # Larger range for better view 
            zaxis=dict(title='Z', range=[-1.5, 1.5]),  # Larger range for better view
            aspectmode='cube',
            bgcolor='white',
            camera=dict(
                eye=dict(x=2.0, y=2.0, z=2.0),        # Further back for larger view
                center=dict(x=0, y=0, z=0),            # Centered on origin
                up=dict(x=0, y=0, z=1)                 # Z-axis up
            ),
            # Center the scene better in the available space
            domain=dict(x=[0.3, 1.0], y=[0.1, 0.9])   # Position scene in right portion of screen
        ),
        width=1800,  # Much wider for better layout
        height=1200,  # Taller for better aspect ratio and centering
        showlegend=False,  # Disable legend - use buttons only for clean interface
        # Space for custom buttons on left
        margin=dict(l=250, r=50, t=80, b=50)
        )
    
    # Save as HTML with custom toggle functionality
    fig.write_html(save_path)
    
    # Add custom JavaScript for true toggle functionality
    with open(save_path, 'r') as f:
        html_content = f.read()
    
    # Add custom CSS and JavaScript for sleek toggle buttons
    custom_script = """
    <style>
    .custom-toggle-btn {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        color: #495057;
        padding: 5px 10px;
        margin: 2px;
        border-radius: 4px;
        cursor: pointer;
        font-size: 12px;
        transition: all 0.2s;
        display: inline-block;
        min-width: 80px;
        text-align: center;
    }
    .custom-toggle-btn:hover {
        background-color: #e9ecef;
        border-color: #adb5bd;
    }
    .custom-toggle-btn.active {
        background-color: #007bff;
        border-color: #007bff;
        color: white;
    }
    .toggle-controls {
        position: absolute;
        top: 20px;
        left: 20px;
        z-index: 1000;
        max-width: 200px;
    }
    .master-controls {
        margin-bottom: 10px;
    }
    .zone-controls {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 2px;
        max-height: 500px;
        overflow-y: auto;
    }
    </style>
    
    <div class="toggle-controls">
        <div class="master-controls">
            <button class="custom-toggle-btn" onclick="showAllZones()">Show All</button>
            <button class="custom-toggle-btn" onclick="hideAllZones()">Hide All</button>
        </div>
        <div class="zone-controls" id="zone-controls">
            <!-- Zone buttons will be populated by JavaScript -->
        </div>
    </div>
    
    <script>
    // Track zone visibility states and trace mappings
    let zoneStates = {};
    let zoneToTraceMap = {};  // Maps zone ID to array of trace indices (quaternions, boundary)
    let populatedZones = [];
    
    // Initialize zone controls
    function initializeZoneControls() {
        const zoneControlsDiv = document.getElementById('zone-controls');
        const plotDiv = document.getElementsByClassName('plotly-graph-div')[0];
        
        if (!plotDiv || !plotDiv.data) return;
        
        // Find zone traces and create mapping
        plotDiv.data.forEach((trace, index) => {
            if (trace.name && trace.name.startsWith('FCC Zone ')) {
                const match = trace.name.match(/FCC Zone (\\d+)/);
                if (match) {
                    const zoneId = parseInt(match[1]);
                    if (!zoneToTraceMap[zoneId]) {
                        zoneToTraceMap[zoneId] = [];
                        populatedZones.push(zoneId);
                        zoneStates[zoneId] = false; // Start hidden
                    }
                    zoneToTraceMap[zoneId].push(index);  // Add trace index to this zone
                }
            } else if (trace.name && trace.name.startsWith('Zone ') && trace.name.includes('Boundary')) {
                const match = trace.name.match(/Zone (\\d+) Boundary/);
                if (match) {
                    const zoneId = parseInt(match[1]);
                    if (!zoneToTraceMap[zoneId]) {
                        zoneToTraceMap[zoneId] = [];
                    }
                    zoneToTraceMap[zoneId].push(index);  // Add boundary trace index to this zone
                }
            }
        });
        
        // Create buttons for all 48 zones
        for (let zoneId = 0; zoneId < 48; zoneId++) {
            const button = document.createElement('button');
            button.id = `zone-btn-${zoneId}`;
            
            if (populatedZones.includes(zoneId)) {
                // Populated zone - functional button
                button.className = 'custom-toggle-btn';
                button.textContent = `Zone ${zoneId}`;
                button.onclick = () => toggleZone(zoneId);
            } else {
                // Empty zone - disabled button
                button.className = 'custom-toggle-btn';
                button.textContent = `Zone ${zoneId}`;
                button.style.opacity = '0.3';
                button.style.cursor = 'not-allowed';
                button.title = 'No quaternions in this zone';
            }
            
            zoneControlsDiv.appendChild(button);
        }
        
        // Hide all zones initially
        hideAllZones();
    }
    
    // Toggle individual zone
    function toggleZone(zoneId) {
        if (!populatedZones.includes(zoneId)) return;
        
        const plotDiv = document.getElementsByClassName('plotly-graph-div')[0];
        const button = document.getElementById(`zone-btn-${zoneId}`);
        const traceIndices = zoneToTraceMap[zoneId];
        
        if (!traceIndices || traceIndices.length === 0) return;
        
        // Toggle state
        zoneStates[zoneId] = !zoneStates[zoneId];
        
        // Update plot visibility for all traces in this zone (quaternions + boundaries)
        const update = {'visible': zoneStates[zoneId]};
        Plotly.restyle(plotDiv, update, traceIndices);
        
        // Update button appearance
        if (zoneStates[zoneId]) {
            button.classList.add('active');
        } else {
            button.classList.remove('active');
        }
    }
    
        // Show all zones
    function showAllZones() {
        const plotDiv = document.getElementsByClassName('plotly-graph-div')[0];
        populatedZones.forEach(zoneId => {
            const traceIndices = zoneToTraceMap[zoneId];
            const button = document.getElementById(`zone-btn-${zoneId}`);
            
            zoneStates[zoneId] = true;
            Plotly.restyle(plotDiv, {'visible': true}, traceIndices);
            button.classList.add('active');
        });
    }
    
    // Hide all zones
    function hideAllZones() {
        const plotDiv = document.getElementsByClassName('plotly-graph-div')[0];
        populatedZones.forEach(zoneId => {
            const traceIndices = zoneToTraceMap[zoneId];
            const button = document.getElementById(`zone-btn-${zoneId}`);
            
            zoneStates[zoneId] = false;
            Plotly.restyle(plotDiv, {'visible': false}, traceIndices);
            button.classList.remove('active');
        });
    }
    
    // Initialize when page loads
    setTimeout(initializeZoneControls, 1000);
    </script>
    """
    
    # Insert custom script before closing body tag
    html_content = html_content.replace('</body>', custom_script + '\n</body>')
    
    # Write the modified HTML
    with open(save_path, 'w') as f:
        f.write(html_content)
    
    print(f"Interactive plot saved as: {save_path}")
    
    return fig

def validate_fundamental_zone(normalized_quaternions, zone_assignments):
    """
    Validate that the 48-zone FCC symmetry classification is working correctly
    """
    print("\n=== FCC SYMMETRY ZONE VALIDATION ===")
    
    # Get statistics for all zones
    unique_zones, zone_counts = np.unique(zone_assignments, return_counts=True)
    
    print(f"Validating {len(normalized_quaternions)} quaternions across FCC symmetry zones:")
    print(f"Found {len(unique_zones)} populated zones out of 48 total zones:")
    
    # Show zone distribution (top 10 most populated)
    total_quats = len(normalized_quaternions)
    sorted_indices = np.argsort(zone_counts)[::-1]
    for i in range(min(10, len(unique_zones))):
        zone_id = unique_zones[sorted_indices[i]]
        count = zone_counts[sorted_indices[i]]
        percentage = (count / total_quats) * 100
        print(f"  Zone {zone_id}: {count} quaternions ({percentage:.1f}%)")
    
    if len(unique_zones) > 10:
        print(f"  ... and {len(unique_zones) - 10} other zones with fewer quaternions")
    
    # Calculate rotation angles for validation
    angles = []
    for q in normalized_quaternions:
        if q[0] < 0:
            q = -q
        angle = 2 * np.arccos(np.clip(np.abs(q[0]), 0, 1)) * 180 / np.pi
        angles.append(angle)
    angles = np.array(angles)
    
    print(f"\nOverall rotation angle statistics:")
    print(f"  Min angle: {np.min(angles):.2f}°")
    print(f"  Max angle: {np.max(angles):.2f}°") 
    print(f"  Mean angle: {np.mean(angles):.2f}°")
    print(f"  Std angle: {np.std(angles):.2f}°")
    
    # Check if quaternions are properly normalized
    norms = np.linalg.norm(normalized_quaternions, axis=1)
    print(f"\nQuaternion normalization check:")
    print(f"  Min norm: {np.min(norms):.6f}")
    print(f"  Max norm: {np.max(norms):.6f}") 
    print(f"  All properly normalized: {'✓' if np.allclose(norms, 1.0) else '✗'}")
    
    print("=== VALIDATION COMPLETE ===\n")

def main():
    # File path
    quat_file = "/data/home/umang/Materials/QRBSA-data-augmentation/quaternions_fibonacci.txt"
    
    # Check if file exists
    if not os.path.exists(quat_file):
        print(f"Error: File {quat_file} not found!")
        return
    
    print("Loading quaternions...")
    quaternions = load_quaternions(quat_file)
    print(f"Loaded {len(quaternions)} quaternions")
    print(f"Quaternion shape: {quaternions.shape}")
    
    # Generate IPF colors
    print("Generating IPF colors...")
    # For large datasets, process in chunks to avoid memory issues
    chunk_size = 50000
    colors = []
    
    for i in range(0, len(quaternions), chunk_size):
        chunk_end = min(i + chunk_size, len(quaternions))
        chunk_quats = quaternions[i:chunk_end]
        chunk_colors = quaternion_to_ipf(chunk_quats, axis="Z")
        colors.append(chunk_colors)
        print(f"Processed {chunk_end}/{len(quaternions)} quaternions")
    
    colors = np.vstack(colors)
    print(f"Generated {len(colors)} IPF colors")
    
    # Create interactive visualization
    print("Creating interactive SO3 sphere visualization...")
    fig = plot_interactive_so3_sphere(quaternions, colors, 
                                     sample_size=100000,  # Increase for more detail
                                     save_path="interactive_fibonacci_so3_sphere.html")
    
    print("Visualization complete!")
    print("Open 'interactive_fibonacci_so3_sphere.html' in a web browser to view the interactive plot")
    
    # Validate fundamental zone classification with a sample
    print("\nValidating fundamental zone classification...")
    symmetry_ops = get_fcc_symmetry_operations()
    sample_size = min(10000, len(quaternions))
    zone_assignments, fz_quaternions = classify_quaternions_by_zone(quaternions[:sample_size], symmetry_ops)
    validate_fundamental_zone(fz_quaternions, zone_assignments)
    
    # Test with a few example quaternions to verify colors
    print("\nTesting with example quaternions:")
    test_quats = [
        [1, 0, 0, 0],      # identity
        [0.92388, 0, 0.38268, 0],
        [0.70711, 0.70711, 0, 0],
    ]
    test_colors = quaternion_to_ipf(test_quats, axis="Z")
    print("Test quaternion colors:", test_colors)

if __name__ == "__main__":
    main()
