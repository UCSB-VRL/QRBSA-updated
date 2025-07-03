# # Write a code that take quaternions in .points file format and convert them each quaternion to fundamental zone

# # readf the .points file
# import numpy as np
# import os
# import sys
# import math
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from mpl_toolkits.mplot3d import Axes3D

# def read_points_file(filename):

#     filename = os.path.join(os.getcwd(), filename)
#     quaternions = []
#     with open(filename, 'r') as f:
#         lines = f.readlines()
#         for line in lines:
#             if line.startswith('#'):
#                 continue
#             else:
#                 quaternions.append([float(x) for x in line.split()])
#     return np.array(quaternions)

# def quaternion_to_fundamental_zone(quaternion):
#     # Normalize the quaternion
#     norm = np.linalg.norm(quaternion)
#     if norm == 0:
#         raise ValueError("Zero quaternion cannot be normalized.")
#     quaternion = quaternion / norm

#     # Convert quaternion to Euler angles
#     w, x, y, z = quaternion
#     phi = math.atan2(2*(w*x + y*z), 1 - 2*(x**2 + y**2))
#     theta = math.asin(2*(w*y - z*x))
#     psi = math.atan2(2*(w*z + x*y), 1 - 2*(y**2 + z**2))

#     return np.array([phi, theta, psi])

# def main():
#     if len(sys.argv) != 2:
#         print("Usage: python quat2fz.py <filename>")
#         sys.exit(1)

#     filename = sys.argv[1]
#     quaternions = read_points_file(filename)
#     fundamental_zones = []

#     for quaternion in quaternions:
#         fz = quaternion_to_fundamental_zone(quaternion)
#         fundamental_zones.append(fz)

#     # Save the fundamental zones to a new file
#     output_filename = os.path.splitext(filename)[0] + '_fz.txt'
#     np.savetxt(output_filename, fundamental_zones, fmt='%.6f')
#     print(f"Fundamental zones saved to {output_filename}")

import numpy as np
from mat_sci_torch_quats.symmetries import fcc_syms
import torch
# def hamilton_product(q1, q2):
#     """Quaternion multiplication: q1 ⊗ q2"""
#     w1, x1, y1, z1 = q1
#     w2, x2, y2, z2 = q2
#     return np.array([
#         w1*w2 - x1*x2 - y1*y2 - z1*z2,
#         w1*x2 + x1*w2 + y1*z2 - z1*y2,
#         w1*y2 - x1*z2 + y1*w2 + z1*x2,
#         w1*z2 + x1*y2 - y1*x2 + z1*w2
#     ])

# ---- Step 1: Generate uniform quaternions using Fibonacci sampling ----
phi = np.sqrt(2.0)
psi = 1.533751168755204288118041
n = 1000000
Q = np.empty((n, 4), dtype=float)

for i in range(n):
    s = i + 0.5
    r = np.sqrt(s / n)
    R = np.sqrt(1.0 - s / n)
    alpha = 2.0 * np.pi * s / phi
    beta = 2.0 * np.pi * s / psi
    Q[i] = [
        r * np.sin(alpha),
        r * np.cos(alpha),
        R * np.sin(beta),
        R * np.cos(beta),
    ]
    Q[i] /= np.linalg.norm(Q[i])  # ensure unit quaternion

# Save original quaternions
np.savetxt("quaternions_fibonacci.txt", Q, fmt="%.6f")

def hamilton_product(q1, q2):
    """
    Vectorized Hamilton product between q1 and q2.
    q1.shape and q2.shape can broadcast to a common shape (..., 4).
    Returns an array of shape (..., 4) after broadcasting.
    """
    # Each quaternion is [qr, qx, qy, qz].
    # We'll manually broadcast each component, then compute out = q1 * q2.
    
    # Split into components:
    r1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    r2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]

    # Determine the final broadcast shape, excluding last dimension 4:
    out_shape = np.broadcast_shapes(q1.shape[:-1], q2.shape[:-1]) + (4,)
    out = np.empty(out_shape, dtype=q1.dtype)

    # Broadcast each component up to out_shape[:-1]
    r1b = np.broadcast_to(r1, out_shape[:-1])
    x1b = np.broadcast_to(x1, out_shape[:-1])
    y1b = np.broadcast_to(y1, out_shape[:-1])
    z1b = np.broadcast_to(z1, out_shape[:-1])

    r2b = np.broadcast_to(r2, out_shape[:-1])
    x2b = np.broadcast_to(x2, out_shape[:-1])
    y2b = np.broadcast_to(y2, out_shape[:-1])
    z2b = np.broadcast_to(z2, out_shape[:-1])

    # Perform the Hamilton product component-wise
    out[..., 0] = r1b * r2b - x1b * x2b - y1b * y2b - z1b * z2b
    out[..., 1] = r1b * x2b + x1b * r2b + y1b * z2b - z1b * y2b
    out[..., 2] = r1b * y2b - x1b * z2b + y1b * r2b + z1b * x2b
    out[..., 3] = r1b * z2b + x1b * y2b - y1b * x2b + z1b * r2b

    return out

def fz_reduce_torch(Q, syms):
    """
    For all quaternions in Q (shape = (N,4)), find the best quaternion 
    in the FCC FZ (closest to the identity [1,0,0,0]).

    Returns best_quat (N,4).

    Arguments:
      - Q:    torch.Tensor of shape (N,4)
      - syms: torch.Tensor of shape (24,4) for FCC symmetry operators
    """
    # 1) Normalize input quaternions => shape (N,4)
    norms_Q = torch.norm(Q, dim=1, keepdim=True)
    Q_norm = Q / (norms_Q + 1e-12)

    # 2) Expand for broadcasting:
    #    syms => (1,24,4)
    #    Q_norm => (N,1,4)
    syms_ext = syms.unsqueeze(0)      # => shape (1,24,4)
    Q_ext = Q_norm.unsqueeze(1)       # => shape (N,1,4)

    # 3) Apply each FCC symmetry => shape (N,24,4)
    quat_sym = hamilton_product_torch(syms_ext, Q_ext)

    # 4) Normalize each symmetrical result => shape (N,24,4)
    norms_sym = torch.norm(quat_sym, dim=2, keepdim=True)
    quat_sym = quat_sym / (norms_sym + 1e-12)

    # 5) Dot with identity => real part is index 0
    #    => shape (N,24)
    dot_vals = quat_sym[..., 0].clamp_(-1.0, 1.0)

    # 6) Misorientation angles => 2 * arccos(real_part)
    angles = 2.0 * torch.acos(dot_vals)  # shape => (N,24)

    # 7) For each of the N quaternions, find the symmetry op giving min angle
    best_idx = torch.argmin(angles, dim=1)  # shape (N,)

    # Optionally check if min angle > 90 degrees
    best_angles = angles[torch.arange(angles.size(0)), best_idx]
    best_angles_deg = best_angles * 180.0 / math.pi
    if (best_angles_deg > 90).any():
        print("Warning: minimum angle > 90° found!")
        bad_idx = (best_angles_deg > 90).nonzero(as_tuple=True)[0]
        print(f"Bad indices: {bad_idx}")
        print(f"Bad angles (deg): {best_angles_deg[bad_idx]}")
        import pdb; pdb.set_trace()

    # 8) Gather best quaternions => shape (N,4)
    best_quat = quat_sym[torch.arange(Q.size(0)), best_idx, :]

    return best_quat

def fz_reduce(Q):
    """
    For all quaternions in Q (shape = (N,4)), find the best quaternion in the FCC FZ.
    Returns best_quat shape = (N,4).
    """
    # Normalize input
    Q_norm = Q / np.linalg.norm(Q, axis=1, keepdims=True)  # (N,4)

    # Broadcast:
    # syms -> (1,24,4)
    # Q_norm -> (N,1,4)
    # => hamilton_product -> (N,24,4)
    Q_ext = Q_norm[:, None, :]   # (N,1,4)
    syms_ext = syms[None, :, :]  # (1,24,4)

    quat_sym = hamilton_product(syms_ext, Q_ext)  # (N,24,4)

    # Normalize each (N,24,4)
    denom = np.linalg.norm(quat_sym, axis=2, keepdims=True)  # (N,24,1)
    quat_sym = quat_sym / denom

    # Dot with identity quaternion => just the real part (index 0)
    # shape => (N,24)
    dot_vals = np.clip(quat_sym[..., 0], -1.0, 1.0)

    # Convert to misorientation angles => 2 * arccos( dot ) 
    angles = 2 * np.arccos(dot_vals)  # shape => (N,24)

    # Find index of minimum angle for each of the N quaternions
    best_idx = np.argmin(angles, axis=1)  # shape => (N,)

    best_angles = angles[np.arange(angles.shape[0]), best_idx]  # shape => (N,)
    best_angles_deg = np.degrees(best_angles)  # shape => (N,)
    # Check for angles > 90 degrees
    if np.any(best_angles_deg > 90):
        print("Warning: minimum angle > 90°")

        # find which index is > 90
        bad_idx = np.where(best_angles_deg > 90)[0]
        print(f"Bad indices: {bad_idx}")
        print(f"Bad angles: {best_angles_deg[bad_idx]}")
        # print(f"Bad quats: {quat_sym[bad_idx]}")
        import pdb; pdb.set_trace()

    # Gather best quats
    best_quat = quat_sym[np.arange(Q.shape[0]), best_idx, :]  # shape => (N,4)

    return best_quat

# for i, quat in enumerate(Q):
#     min_angle = np.inf
#     best_quat = quat

#     for sym in syms:
#         quat_sym = hamilton_product(sym, quat)
#         quat_sym /= np.linalg.norm(quat_sym)
#         dot_val = np.clip(np.dot(quat_sym, [1, 0, 0, 0]), -1.0, 1.0)
#         angle = 2*np.arccos(dot_val)

#         if angle < min_angle:
#             min_angle = angle
#             best_quat = quat_sym

#     deg_angle = np.degrees(min_angle)
#     if deg_angle > 90:
#         print(f"Warning: minimum angle > 90° = {deg_angle:.2f}")
#         import pdb; pdb.set_trace()

#     Q_fz[i] = best_quat

#      # --- If min_angle > 90°, then break into pdb ---
    
# ---- Step 2: Reduce to fundamental zone using FCC symmetry operators ----
syms = np.asarray(fcc_syms)  # shape: (24, 4)
# expand syms by appending negative of syms
syms = np.concatenate((syms, -syms), axis=0)  # shape: (48, 4)

# fz reduce with original function.
from mat_sci_torch_quats.quats_old import fz_reduce

device = torch.device('cuda:0')  # Change to 'cuda:0' if you want to use GPU

# convert Q to torch tensor
Q_tensor = torch.tensor(Q, dtype=torch.float32, device=device)
# convert syms to torch tensor
syms_tensor = torch.tensor(fcc_syms, dtype=torch.float32, device=device)
Q_fz= fz_reduce(Q_tensor, syms_tensor)

# convert back to numpy array
Q_fz = Q_fz.detach().cpu().numpy()
# ---- Step 3: Save reduced quaternions ----
#np.savetxt("quaternions_fz.txt", Q_fz, fmt="%.6f")

print("Saved:")
print(" - quaternions_fibonacci.txt (raw samples)")
print(" - quaternions_fz.txt (symmetry-reduced)")

# ---- Step 4: Compute average angle to [1, 0, 0, 0] ----
identity = np.array([1.0, 0.0, 0.0, 0.0])
angles_rad = []

for q in Q_fz:
    dot_val = np.clip(np.dot(q, identity), -1.0, 1.0)
    angle = 2*np.arccos(dot_val)  # angle in radians
    angles_rad.append(angle)

angles_rad = np.array(angles_rad)
angles_deg = np.degrees(angles_rad)
# save all the quaternions which have angles in degrees> 62 degrees
edge_quats = Q_fz[angles_deg > 62]
print(f"Number of quaternions with angle > 62 degrees: {len(edge_quats)}")
np.savetxt("quaternions_edge_fz.txt", edge_quats, fmt="%.6f")

avg_angle_rad = np.mean(angles_rad)
avg_angle_deg = np.degrees(avg_angle_rad)

# calculate median and mode of angles
median_angle_rad = np.median(angles_rad)
median_angle_deg = np.degrees(median_angle_rad)

mode_angle_rad = np.unique(angles_rad, return_counts=True)
mode_angle_rad = mode_angle_rad[0][np.argmax(mode_angle_rad[1])]
mode_angle_deg = np.degrees(mode_angle_rad)

# calculate max and min of angles
max_angle_rad = np.max(angles_rad)
max_angle_deg = np.degrees(max_angle_rad)
min_angle_rad = np.min(angles_rad)
min_angle_deg = np.degrees(min_angle_rad)


print(f"Average angle to identity quaternion:")
print(f"  - Radians: {avg_angle_rad:.6f}")
print(f"  - Degrees: {avg_angle_deg:.6f}")

print(f"Median angle to identity quaternion:")
print(f"  - Radians: {median_angle_rad:.6f}")
print(f"  - Degrees: {median_angle_deg:.6f}")

print(f"Mode angle to identity quaternion:")
print(f"  - Radians: {mode_angle_rad:.6f}")
print(f"  - Degrees: {mode_angle_deg:.6f}")

print(f"Max angle to identity quaternion:")
print(f"  - Radians: {max_angle_rad:.6f}")
print(f"  - Degrees: {max_angle_deg:.6f}")

print(f"Min angle to identity quaternion:")
print(f"  - Radians: {min_angle_rad:.6f}")
print(f"  - Degrees: {min_angle_deg:.6f}")

# get histogram of angles
import matplotlib.pyplot as plt
plt.hist(np.degrees(angles_rad), bins=100)
plt.xlabel("Angle (degrees)")
plt.ylabel("Frequency")
plt.title("Histogram of angles to identity quaternion")
plt.grid()
plt.savefig("angle_histogram.png")
plt.show()


# NOW DO IT FOR EDGE QUATS.
# Calculate statistics for edge_quats
angles_rad_edge = []
for q in edge_quats:
    dot_val = np.clip(np.dot(q, identity), -1.0, 1.0)
    angle = 2 * np.arccos(dot_val)
    angles_rad_edge.append(angle)
angles_rad_edge = np.array(angles_rad_edge)

avg_angle_rad_edge = np.mean(angles_rad_edge)
avg_angle_deg_edge = np.degrees(avg_angle_rad_edge)

# calculate median and mode of angles
median_angle_rad_edge = np.median(angles_rad_edge)
median_angle_deg_edge = np.degrees(median_angle_rad_edge)

mode_angle_rad_edge = np.unique(angles_rad_edge, return_counts=True)
mode_angle_rad_edge = mode_angle_rad_edge[0][np.argmax(mode_angle_rad_edge[1])]
mode_angle_deg_edge = np.degrees(mode_angle_rad_edge)

# calculate max and min of angles
max_angle_rad_edge = np.max(angles_rad_edge)
max_angle_deg_edge = np.degrees(max_angle_rad_edge)
min_angle_rad_edge = np.min(angles_rad_edge)
min_angle_deg_edge = np.degrees(min_angle_rad_edge)

print(f"Average angle to identity quaternion (edge_quats):")
print(f"  - Radians: {avg_angle_rad_edge:.6f}")
print(f"  - Degrees: {avg_angle_deg_edge:.6f}")

print(f"Median angle to identity quaternion (edge_quats):")
print(f"  - Radians: {median_angle_rad_edge:.6f}")
print(f"  - Degrees: {median_angle_deg_edge:.6f}")

print(f"Mode angle to identity quaternion (edge_quats):")
print(f"  - Radians: {mode_angle_rad_edge:.6f}")
print(f"  - Degrees: {mode_angle_deg_edge:.6f}")

print(f"Max angle to identity quaternion (edge_quats):")
print(f"  - Radians: {max_angle_rad_edge:.6f}")
print(f"  - Degrees: {max_angle_deg_edge:.6f}")

print(f"Min angle to identity quaternion (edge_quats):")
print(f"  - Radians: {min_angle_rad_edge:.6f}")
print(f"  - Degrees: {min_angle_deg_edge:.6f}")

# get histogram of angles for edge_quats
import matplotlib.pyplot as plt
plt.hist(np.degrees(angles_rad_edge))
plt.xlabel("Angle (degrees)")
plt.ylabel("Frequency")
plt.title("Histogram of angles to identity quaternion (edge_quats)")
plt.grid()
plt.savefig("angle_histogram_edge_quats.png")
plt.show()


# Calculate for original Q
angles_rad_orig = []
for q in Q:
    dot_val = np.clip(np.dot(q, identity), -1.0, 1.0)
    angle = 2*np.arccos(dot_val)  # angle in radians
    angles_rad_orig.append(angle)
angles_rad_orig = np.array(angles_rad_orig)
avg_angle_rad_orig = np.mean(angles_rad_orig)
avg_angle_deg_orig = np.degrees(avg_angle_rad_orig)
median_angle_rad_orig = np.median(angles_rad_orig)
median_angle_deg_orig = np.degrees(median_angle_rad_orig)

mode_angle_rad_orig = np.unique(angles_rad_orig, return_counts=True)
mode_angle_rad_orig = mode_angle_rad_orig[0][np.argmax(mode_angle_rad_orig[1])]
mode_angle_deg_orig = np.degrees(mode_angle_rad_orig)
# print(f"Average angle to identity quaternion (original):")
# print(f"  - Radians: {avg_angle_rad_orig:.6f}")
# print(f"  - Degrees: {avg_angle_deg_orig:.6f}")
# print(f"Median angle to identity quaternion (original):")
# print(f"  - Radians: {median_angle_rad_orig:.6f}")

# print(f"  - Degrees: {median_angle_deg_orig:.6f}")
# print(f"Mode angle to identity quaternion (original):")
# print(f"  - Radians: {mode_angle_rad_orig:.6f}")
# print(f"  - Degrees: {mode_angle_deg_orig:.6f}")
