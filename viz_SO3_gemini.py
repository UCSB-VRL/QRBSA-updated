# Generate the full 24 rotation matrices of the cubic (octahedral) group and convert to quaternions.
import numpy as np
import itertools
import pandas as pd

def mat_to_quat(M):
    """Convert rotation matrix to quaternion (w, x, y, z) -- numerically stable method."""
    m = np.asarray(M, dtype=float)
    trace = m[0,0] + m[1,1] + m[2,2]
    if trace > 0.0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (m[2,1] - m[1,2]) * s
        y = (m[0,2] - m[2,0]) * s
        z = (m[1,0] - m[0,1]) * s
    else:
        # Find the largest diagonal element
        if (m[0,0] > m[1,1]) and (m[0,0] > m[2,2]):
            s = 2.0 * np.sqrt(1.0 + m[0,0] - m[1,1] - m[2,2])
            w = (m[2,1] - m[1,2]) / s
            x = 0.25 * s
            y = (m[0,1] + m[1,0]) / s
            z = (m[0,2] + m[2,0]) / s
        elif m[1,1] > m[2,2]:
            s = 2.0 * np.sqrt(1.0 + m[1,1] - m[0,0] - m[2,2])
            w = (m[0,2] - m[2,0]) / s
            x = (m[0,1] + m[1,0]) / s
            y = 0.25 * s
            z = (m[1,2] + m[2,1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + m[2,2] - m[0,0] - m[1,1])
            w = (m[1,0] - m[0,1]) / s
            x = (m[0,2] + m[2,0]) / s
            y = (m[1,2] + m[2,1]) / s
            z = 0.25 * s
    q = np.array([w,x,y,z], dtype=float)
    # normalize
    q /= np.linalg.norm(q)
    # canonicalize to w>=0
    if q[0] < 0:
        q = -q
    return q

def generate_cubic_rotations():
    """Generate the 24 rotation matrices of the cube (proper rotations = octahedral group)."""
    mats = []
    basis = np.eye(3)
    perms = list(itertools.permutations([0,1,2]))
    signs = list(itertools.product([1,-1], repeat=3))
    for p in perms:
        P = np.array([basis[i] for i in p]).T  # columns are permuted basis vectors
        for s in signs:
            S = np.diag(s)
            M = P @ S
            if np.linalg.det(M) > 0.5:  # det == +1 (numerical tol)
                # ensure proper rotation (orthonormal)
                if np.allclose(np.dot(M.T, M), np.eye(3), atol=1e-8):
                    mats.append(M)
    # Deduplicate (some constructions can produce duplicates due to permutations)
    uniq = []
    for M in mats:
        found = False
        for U in uniq:
            if np.allclose(M, U, atol=1e-8):
                found = True
                break
        if not found:
            uniq.append(M)
    return uniq

mats = generate_cubic_rotations()
len(mats), mats[0].shape if mats else None

# Convert to quaternions
rows = []
for i, M in enumerate(mats):
    q = mat_to_quat(M)  # canonical w>=0
    # store both q and -q as distinct quaternion representations in quaternion space
    for sign in [1, -1]:
        qq = sign * q
        # axis-angle for readability
        w, x, y, z = qq
        theta = 2 * np.arccos(np.clip(w, -1.0, 1.0))
        s = np.sqrt(max(0.0, 1 - w*w))
        if s < 1e-8:
            axis = np.array([0.0, 0.0, 0.0])
        else:
            axis = np.array([x, y, z]) / s
        rows.append({
            "mat_index": i,
            "w": float(qq[0]),
            "x": float(qq[1]),
            "y": float(qq[2]),
            "z": float(qq[3]),
            "angle_rad": float(theta),
            "angle_deg": float(np.degrees(theta)),
            "axis_x": float(axis[0]),
            "axis_y": float(axis[1]),
            "axis_z": float(axis[2])
        })

df = pd.DataFrame(rows)
# sort for stable presentation: by mat_index then w desc
df = df.sort_values(by=["mat_index", "w"], ascending=[True, False]).reset_index(drop=True)

# Show counts and a preview
print(f"Generated {len(df)} quaternions from {len(mats)} rotation matrices")
print("\nFirst few quaternions:")
print(df.head(10))

print(f"\nDataFrame shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

# Save to local directory instead of /mnt/data/
output_file = "cubic_group_quaternions_48.csv"
df.to_csv(output_file, index=False)
print(f"\nSaved quaternions to: {output_file}")

# Show summary statistics
print(f"\nSummary:")
print(f"Number of rotation matrices: {len(mats)}")
print(f"Number of quaternions (including ±q): {len(df)}")
print(f"Unique angles (degrees): {sorted(df['angle_deg'].unique())}")
