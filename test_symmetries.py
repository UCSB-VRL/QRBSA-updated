#!/usr/bin/env python3
"""
Quick test to verify the 48 FCC symmetry operations
"""

import numpy as np
from viz_SO3_interactive import get_fcc_symmetry_operations, quaternion_to_unit_sphere

def test_symmetry_operations():
    """Test that we have the correct number and types of symmetry operations"""
    
    # Get all symmetry operations
    sym_ops = get_fcc_symmetry_operations()
    
    print(f"Total symmetry operations: {len(sym_ops)}")
    print(f"Expected: 48 (24 rotations + 24 inverses)")
    
    # Check for unique operations (within numerical precision)
    unique_ops = []
    tolerance = 1e-10
    
    for op in sym_ops:
        is_unique = True
        for unique_op in unique_ops:
            if np.allclose(op, unique_op, atol=tolerance) or np.allclose(op, -unique_op, atol=tolerance):
                is_unique = False
                break
        if is_unique:
            unique_ops.append(op)
    
    print(f"Unique operations (considering q and -q as same): {len(unique_ops)}")
    
    # Categorize operations by rotation angle
    categories = {"Identity": 0, "90° (4-fold)": 0, "120° (3-fold)": 0, "180° (2-fold)": 0}
    
    for q in sym_ops:
        # Normalize
        q = q / np.linalg.norm(q)
        w = abs(q[0])  # Take absolute value since q and -q represent same rotation
        
        if w > 0.999:  # cos(0°/2) = 1
            categories["Identity"] += 1
        elif w < 0.1:  # cos(90°/2) = cos(45°) ≈ 0.707, but this catches ~90°
            categories["90° (4-fold)"] += 1
        elif w < 0.6:  # cos(60°) = 0.5, catches 120° rotations
            categories["120° (3-fold)"] += 1
        else:
            categories["180° (2-fold)"] += 1
    
    print("\nOperation types:")
    for category, count in categories.items():
        print(f"  {category}: {count}")
    
    # Test a few specific operations
    print(f"\nFirst few operations:")
    for i, op in enumerate(sym_ops[:8]):
        sphere_coord = quaternion_to_unit_sphere(op.reshape(1, -1))[0]
        print(f"  Op {i+1}: quat={op} -> sphere=({sphere_coord[0]:.3f}, {sphere_coord[1]:.3f}, {sphere_coord[2]:.3f})")

if __name__ == "__main__":
    test_symmetry_operations()
