#!/usr/bin/env python3
"""
Debug script to analyze zone classification issues
"""

import numpy as np
import matplotlib.pyplot as plt
from viz_SO3_interactive import (
    get_fcc_symmetry_operations, 
    quaternion_to_unit_sphere,
    classify_quaternions_by_zone,
    load_quaternions
)

def debug_symmetry_operations():
    """Debug the 48 symmetry operations"""
    print("=== DEBUGGING SYMMETRY OPERATIONS ===")
    
    symmetry_ops = get_fcc_symmetry_operations()
    print(f"Number of symmetry operations: {len(symmetry_ops)}")
    
    # Check if all are unit quaternions
    norms = np.linalg.norm(symmetry_ops, axis=1)
    print(f"Quaternion norms - Min: {norms.min():.6f}, Max: {norms.max():.6f}")
    print(f"All unit quaternions? {np.allclose(norms, 1.0)}")
    
    # Convert to sphere coordinates
    sphere_coords = quaternion_to_unit_sphere(symmetry_ops)
    sphere_norms = np.linalg.norm(sphere_coords, axis=1)
    print(f"Sphere coordinate norms - Min: {sphere_norms.min():.6f}, Max: {sphere_norms.max():.6f}")
    print(f"All on unit sphere? {np.allclose(sphere_norms, 1.0)}")
    
    # Check for duplicates (accounting for ±q equivalence)
    unique_coords = []
    for coord in sphere_coords:
        is_duplicate = False
        for unique_coord in unique_coords:
            if np.allclose(coord, unique_coord, atol=1e-6) or np.allclose(coord, -unique_coord, atol=1e-6):
                is_duplicate = True
                break
        if not is_duplicate:
            unique_coords.append(coord)
    
    print(f"Unique sphere coordinates: {len(unique_coords)}")
    
    return symmetry_ops, sphere_coords

def debug_fundamental_zone():
    """Debug fundamental zone classification"""
    print("\n=== DEBUGGING FUNDAMENTAL ZONE ===")
    
    # Test with known quaternions that should be in FZ
    test_quats = np.array([
        [1, 0, 0, 0],                    # Identity - definitely in FZ
        [0.9659, 0.2588, 0, 0],         # 30° around X - should be in FZ
        [0.7071, 0.7071, 0, 0],         # 90° around X - boundary of FZ
        [0.5, 0.5, 0.5, 0.5],           # 120° around [111] - boundary of FZ  
        [0.9239, 0.3827, 0, 0],         # 45° around X - boundary of FZ
    ])
    
    print("Test quaternions:")
    for i, q in enumerate(test_quats):
        angle = 2 * np.arccos(min(1.0, abs(q[0])))
        print(f"  Q{i}: {q} -> {np.degrees(angle):.1f}°")
    
    # Classify using current method
    symmetry_ops = get_fcc_symmetry_operations()
    zone_assignments, fz_quats = classify_quaternions_by_zone(test_quats, symmetry_ops)
    
    print("\nClassification results:")
    for i, (original, fz_reduced, zone) in enumerate(zip(test_quats, fz_quats, zone_assignments)):
        orig_angle = 2 * np.arccos(min(1.0, abs(original[0])))
        fz_angle = 2 * np.arccos(min(1.0, abs(fz_reduced[0])))
        print(f"  Q{i}: Zone {zone}")
        print(f"    Original: {original} ({np.degrees(orig_angle):.1f}°)")
        print(f"    FZ-reduced: {fz_reduced} ({np.degrees(fz_angle):.1f}°)")

def debug_fz_boundaries():
    """Check what the actual FZ boundaries should be"""
    print("\n=== FUNDAMENTAL ZONE BOUNDARIES ===")
    
    # For cubic (m-3m) symmetry, the FZ boundaries are:
    # - 45° around <100> axes (π/4 radians)
    # - 54.74° around <111> axes (π/3 radians) 
    # - 90° around <110> axes (π/2 radians)
    
    print("Expected FZ boundary angles:")
    print(f"  <100> directions: 45.0° ({np.degrees(np.pi/4):.1f}°)")
    print(f"  <111> directions: 54.7° ({np.degrees(np.arccos(np.sqrt(2/3))):.1f}°)")
    print(f"  <110> directions: 90.0° ({np.degrees(np.pi/2):.1f}°)")
    
    # The fundamental zone should be bounded by these angles
    max_fz_angle = 54.74  # Degrees
    print(f"\nMaximum FZ angle should be ~{max_fz_angle:.1f}°")

if __name__ == "__main__":
    debug_symmetry_operations()
    debug_fundamental_zone() 
    debug_fz_boundaries()
